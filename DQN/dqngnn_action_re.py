from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from DQN.dqngnn_re import GraphEncoderGINE, PrioritizedReplayMemoryNumpy


@dataclass(frozen=True)
class DQNGNNActionShapeConfig:
    num_nodes: int
    max_neighbors: int
    node_in: int
    edge_in: int
    gnn_hidden: int = 128
    action_feature_dim: int = 15

    @property
    def tls_state_dim(self) -> int:
        return self.max_neighbors * 2

    @property
    def action_input_dim(self) -> int:
        return self.gnn_hidden * 3 + self.action_feature_dim


class ActionQNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x).squeeze(-1)


class DQNGNNActionNetworks(nn.Module):
    def __init__(self, shape: DQNGNNActionShapeConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_encoder = GraphEncoderGINE(shape.node_in, shape.edge_in, shape.gnn_hidden)
        self.target_encoder = GraphEncoderGINE(shape.node_in, shape.edge_in, shape.gnn_hidden)
        self.eval_q_net = ActionQNetwork(shape.action_input_dim)
        self.target_q_net = ActionQNetwork(shape.action_input_dim)
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

    def q_values_for_sample(
        self,
        *,
        encoder: GraphEncoderGINE,
        q_net: ActionQNetwork,
        data: Any,
        bag: tuple[int, int, list[int], torch.Tensor],
    ) -> torch.Tensor:
        source, target, neighbors, action_features = bag
        data = data.clone().to(self.device)
        node_embeddings = encoder(data.x, data.edge_index, data.edge_attr)
        return self._q_values_from_embeddings(q_net, node_embeddings, source, target, neighbors, action_features)

    def batch_q_values(
        self,
        *,
        encoder: GraphEncoderGINE,
        q_net: ActionQNetwork,
        data_list: list[Any],
        bag_list: list[tuple[int, int, list[int], torch.Tensor]],
    ) -> torch.Tensor:
        data_list = [data.clone().to(self.device) for data in data_list]
        batch = Batch.from_data_list(data_list)
        all_embeddings = encoder(batch.x, batch.edge_index, batch.edge_attr)
        rows = []
        for idx, bag in enumerate(bag_list):
            source, target, neighbors, action_features = bag
            node_embeddings = all_embeddings[batch.ptr[idx] : batch.ptr[idx + 1]]
            rows.append(self._q_values_from_embeddings(q_net, node_embeddings, source, target, neighbors, action_features))
        return torch.stack(rows, dim=0)

    def _q_values_from_embeddings(
        self,
        q_net: ActionQNetwork,
        node_embeddings: torch.Tensor,
        source: int,
        target: int,
        neighbors: list[int],
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        q_values = torch.full((self.shape.max_neighbors,), -1e9, device=self.device)
        if not neighbors:
            return q_values

        source_embedding = node_embeddings[source]
        target_embedding = node_embeddings[target]
        candidate_count = min(len(neighbors), self.shape.max_neighbors)
        action_features = action_features.to(device=self.device, dtype=node_embeddings.dtype)

        action_inputs = []
        for idx in range(candidate_count):
            candidate_embedding = node_embeddings[neighbors[idx]]
            action_input = torch.cat(
                [source_embedding, target_embedding, candidate_embedding, action_features[idx]],
                dim=0,
            )
            action_inputs.append(action_input)

        action_inputs_tensor = torch.stack(action_inputs, dim=0)
        q_values[:candidate_count] = q_net(action_inputs_tensor)
        return q_values


class DQNGNNActionAgentRefactored:
    """
    Action-centric DQNGNN agent.

    Instead of representing each action only by its neighbor node embedding,
    this agent directly builds Q(s, a) from:
    - source node embedding
    - target node embedding
    - candidate neighbor embedding
    - candidate action features
    """

    def __init__(
        self,
        shape: DQNGNNActionShapeConfig,
        *,
        edge_feature_mode: str = "all",
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
        self.shape = shape
        self.edge_feature_mode = edge_feature_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networks = DQNGNNActionNetworks(shape, device=self.device)
        self.optimizer = torch.optim.Adam(
            list(self.networks.eval_encoder.parameters()) + list(self.networks.eval_q_net.parameters()),
            lr=lr,
            weight_decay=1e-5,
        )
        self.loss_func = nn.MSELoss(reduction="none")
        self.memory = PrioritizedReplayMemoryNumpy(
            memory_capacity,
            shape.tls_state_dim,
            1,
            shape.max_neighbors,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )
        self.a_size = shape.max_neighbors
        self.num_nodes = shape.num_nodes
        self.use_gnn = True
        self.EPSILON = 0.9
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.gamma = gamma
        self.BATCH_SIZE = batch_size
        self.target_tau = target_tau
        self.learn_step_counter = 0
        self.show = True

        self.encoder = self.networks.eval_encoder
        self.eval_net = self.networks.eval_q_net
        self.target_net = self.networks.target_q_net

    def _pad_neighbors(self, neighbors: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not neighbors:
            return (
                torch.zeros(self.a_size, dtype=torch.long, device=self.device),
                torch.zeros(self.a_size, dtype=torch.bool, device=self.device),
            )
        padded = neighbors + [neighbors[0]] * (self.a_size - len(neighbors))
        neighbor_idx = torch.tensor(padded[: self.a_size], dtype=torch.long, device=self.device)
        mask = torch.zeros(self.a_size, dtype=torch.bool, device=self.device)
        mask[: min(len(neighbors), self.a_size)] = True
        return neighbor_idx, mask

    def _pad_feat2(self, feat_list: list[list[float]] | None, feat_dim: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.zeros(self.a_size, feat_dim, device=self.device, dtype=torch.float32)
        mask = torch.zeros(self.a_size, device=self.device, dtype=torch.bool)
        if feat_list is None:
            return values, mask
        count = min(len(feat_list), self.a_size)
        if count:
            values[:count] = torch.tensor(feat_list[:count], device=self.device, dtype=torch.float32)
            mask[:count] = True
        return values, mask

    def concat_state_tls(self, state: torch.Tensor, tls: list[list[float]]) -> torch.Tensor:
        tls_state, _ = self._pad_feat2(tls)
        return tls_state.flatten()

    @torch.no_grad()
    def choose_action_gnn(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor,
        neighbors: list[int],
        source: int,
        target: int,
        data_t: Any,
        evaluate: bool = False,
    ) -> int:
        if not neighbors:
            raise ValueError("choose_action_gnn requires at least one neighbor")
        _, neighbor_mask = self._pad_neighbors(neighbors)
        if random.random() <= self.EPSILON and not evaluate:
            return random.choice(neighbors)

        action_features = data_t.action_features if hasattr(data_t, "action_features") else None
        if action_features is None:
            raise ValueError("Action-centric agent requires data_t.action_features")
        q_values = self.networks.q_values_for_sample(
            encoder=self.networks.eval_encoder,
            q_net=self.networks.eval_q_net,
            data=data_t,
            bag=(source, target, neighbors, action_features),
        )
        q_values = q_values.masked_fill(~neighbor_mask, -1e9)
        return neighbors[int(q_values.argmax())]

    def learn_dqn_neighbor(self) -> float | None:
        if len(self.memory) < self.BATCH_SIZE:
            return None
        if self.show:
            print("learning..")
            self.show = False

        self.networks.eval_encoder.train()
        self.networks.eval_q_net.train()
        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            done_batch,
            next_mask,
            data_t_list,
            next_data_t_list,
            bag_list,
            next_bag_list,
            sample_indices,
            sample_weights,
        ) = self.memory.sample(self.BATCH_SIZE)

        action_batch = action_batch.to(self.device).to(torch.int64)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        next_mask = next_mask.to(self.device).bool()
        sample_weights = sample_weights.to(self.device)

        state_q_values = self.networks.batch_q_values(
            encoder=self.networks.eval_encoder,
            q_net=self.networks.eval_q_net,
            data_list=data_t_list,
            bag_list=bag_list,
        )
        state_action_values = state_q_values.gather(1, action_batch)

        with torch.no_grad():
            next_eval_q = self.networks.batch_q_values(
                encoder=self.networks.eval_encoder,
                q_net=self.networks.eval_q_net,
                data_list=next_data_t_list,
                bag_list=next_bag_list,
            )
            next_eval_q = next_eval_q.masked_fill(~next_mask, -1e9)
            next_action = next_eval_q.argmax(dim=1, keepdim=True)

            next_target_q = self.networks.batch_q_values(
                encoder=self.networks.target_encoder,
                q_net=self.networks.target_q_net,
                data_list=next_data_t_list,
                bag_list=next_bag_list,
            )
            next_target_q = next_target_q.masked_fill(~next_mask, -1e9)
            selected_next_q = next_target_q.gather(1, next_action)
            target = reward_batch + (1.0 - done_batch) * self.gamma * selected_next_q

        td_errors = state_action_values - target
        loss = (self.loss_func(state_action_values, target) * sample_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.networks.eval_encoder.parameters()) + list(self.networks.eval_q_net.parameters()),
            max_norm=10.0,
        )
        self.optimizer.step()
        self.memory.update_priorities(sample_indices, td_errors)
        self.networks.sync_target(hard=False, tau=self.target_tau)
        self.learn_step_counter += 1
        return float(loss.item())

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
                "edge_feature_mode": self.edge_feature_mode,
                "agent_type": "dqngnn_action_refactored",
            },
            filename,
        )

    def load_full_model(
        self,
        gnn: bool = True,
        filename: str = "dqngnn_action_refactored.pt",
        strict: bool = True,
        load_optimizer: bool = True,
        buffer_name: str | None = None,
    ) -> int:
        checkpoint = torch.load(filename, map_location=self.device)
        self.networks.eval_q_net.load_state_dict(checkpoint["eval_q_net"], strict=strict)
        self.networks.target_q_net.load_state_dict(checkpoint["target_q_net"], strict=strict)
        self.networks.eval_encoder.load_state_dict(checkpoint["eval_encoder"], strict=strict)
        self.networks.target_encoder.load_state_dict(checkpoint["target_encoder"], strict=strict)
        if load_optimizer and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as exc:
                print(f"[warn] optimizer state could not be loaded: {exc}")
        return int(checkpoint.get("episode", 0))
