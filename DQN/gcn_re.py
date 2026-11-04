from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from DQN.dqngnn_re import DQNGNNAgentRefactored, DQNGNNShapeConfig, QNetwork


class GraphEncoderGCN(nn.Module):
    def __init__(self, node_in: int, hidden: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_channels = node_in if layer_idx == 0 else hidden
            self.layers.append(GCNConv(in_channels, hidden))
            self.norms.append(nn.LayerNorm(hidden))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = F.relu(conv(h, edge_index))
            if residual.shape == h.shape:
                h = h + residual
            h = norm(h)
        return h


class GCNGNNNetworks(nn.Module):
    def __init__(self, shape: DQNGNNShapeConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_encoder = GraphEncoderGCN(shape.node_in, shape.gnn_hidden)
        self.target_encoder = GraphEncoderGCN(shape.node_in, shape.gnn_hidden)
        self.eval_q_net = QNetwork(shape.dqn_input_dim, shape.max_neighbors)
        self.target_q_net = QNetwork(shape.dqn_input_dim, shape.max_neighbors)
        self.to(self.device)
        self.sync_target(hard=True)

    @torch.no_grad()
    def sync_target(self, hard: bool = False, tau: float = 0.003) -> None:
        self._sync_module(self.target_q_net, self.eval_q_net, hard=hard, tau=tau)
        self._sync_module(self.target_encoder, self.eval_encoder, hard=hard, tau=tau)

    @staticmethod
    @torch.no_grad()
    def _sync_module(target: nn.Module, source: nn.Module, hard: bool, tau: float) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            if hard:
                target_param.data.copy_(source_param.data)
            else:
                target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def encode_action_input(
        self,
        *,
        encoder: GraphEncoderGCN,
        data,
        source: int,
        target: int,
        neighbors: list[int],
        tls_state: torch.Tensor,
    ) -> torch.Tensor:
        data = data.clone().to(self.device)
        node_embeddings = encoder(data.x, data.edge_index, data.edge_attr)
        neighbor_idx = self._padded_neighbor_tensor(neighbors)
        graph_embedding = self._flatten_local_graph_embedding(node_embeddings, source, target, neighbor_idx)
        return torch.cat([graph_embedding, tls_state.to(self.device)], dim=-1)

    def batch_encode_action_input(
        self,
        *,
        encoder: GraphEncoderGCN,
        data_list: list,
        bag_list: list[tuple[int, int, list[int]]],
        tls_state_batch: torch.Tensor,
    ) -> torch.Tensor:
        from torch_geometric.data import Batch

        data_list = [data.clone().to(self.device) for data in data_list]
        batch = Batch.from_data_list(data_list)
        all_embeddings = encoder(batch.x, batch.edge_index, batch.edge_attr)
        inputs = []
        for idx, bag in enumerate(bag_list):
            source, target, neighbors = bag[:3]
            graph_embeddings = all_embeddings[batch.ptr[idx] : batch.ptr[idx + 1]]
            neighbor_idx = self._padded_neighbor_tensor(neighbors)
            local_embedding = self._flatten_local_graph_embedding(graph_embeddings, source, target, neighbor_idx)
            inputs.append(torch.cat([local_embedding, tls_state_batch[idx].to(self.device)], dim=-1))
        return torch.stack(inputs, dim=0)

    def eval_q_values(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.eval_q_net(model_input)

    @torch.no_grad()
    def target_q_values(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.target_q_net(model_input)

    def _padded_neighbor_tensor(self, neighbors: list[int]) -> torch.Tensor:
        if not neighbors:
            return torch.zeros(self.shape.max_neighbors, dtype=torch.long, device=self.device)
        padded = neighbors + [neighbors[0]] * (self.shape.max_neighbors - len(neighbors))
        return torch.tensor(padded[: self.shape.max_neighbors], dtype=torch.long, device=self.device)

    def _flatten_local_graph_embedding(
        self,
        node_embeddings: torch.Tensor,
        source: int,
        target: int,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        source_embedding = node_embeddings[source].unsqueeze(0)
        target_embedding = node_embeddings[target].unsqueeze(0)
        neighbor_embeddings = node_embeddings.index_select(0, neighbor_idx)
        return torch.cat([source_embedding, target_embedding, neighbor_embeddings], dim=0).flatten()


class GCNGNNAgentRefactored(DQNGNNAgentRefactored):
    """
    GCN variant of the refactored GNN agent.

    It keeps the same DQN-side mechanisms as DQNGNNAgentRefactored:
    - Double DQN target
    - Dueling QNetwork
    - PER

    The only intended architectural difference is the graph encoder:
    GINEConv(edge_attr-aware) -> GCNConv(edge_attr-agnostic).
    """

    def __init__(
        self,
        shape: DQNGNNShapeConfig,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        memory_capacity: int = 10000,
        target_tau: float = 0.003,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            shape,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            memory_capacity=memory_capacity,
            target_tau=target_tau,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
            per_beta_frames=per_beta_frames,
            device=device,
        )
        self.networks = GCNGNNNetworks(shape, device=self.device)
        self.optimizer = torch.optim.Adam(
            list(self.networks.eval_encoder.parameters()) + list(self.networks.eval_q_net.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )
        self.encoder = self.networks.eval_encoder
        self.eval_net = self.networks.eval_q_net
        self.target_net = self.networks.target_q_net

    def save_tmp_file(self, ep: int, filename: str = "tmp_file.pt") -> None:
        torch.save(
            {
                "episode": ep,
                "eval_q_net": self.networks.eval_q_net.state_dict(),
                "target_q_net": self.networks.target_q_net.state_dict(),
                "eval_encoder": self.networks.eval_encoder.state_dict(),
                "target_encoder": self.networks.target_encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "shape": self.shape,
                "agent_type": "gcn_gnn_refactored",
            },
            filename,
        )
