from __future__ import annotations

from dataclasses import dataclass
import random
from collections import namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv


Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "next_state",
        "reward",
        "done",
        "next_action_mask",
        "data_t",
        "next_data_t",
        "bag",
        "next_bag",
    ),
)


@dataclass(frozen=True)
class DQNGNNShapeConfig:
    num_nodes: int
    max_neighbors: int
    node_in: int
    edge_in: int
    gnn_hidden: int = 128

    @property
    def tls_state_dim(self) -> int:
        return self.max_neighbors * 2

    @property
    def graph_embedding_dim(self) -> int:
        return self.gnn_hidden * (self.max_neighbors + 2)

    @property
    def dqn_input_dim(self) -> int:
        return self.graph_embedding_dim + self.tls_state_dim


class GraphEncoderGINE(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_channels = node_in if layer_idx == 0 else hidden
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=edge_in))
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = F.relu(conv(h, edge_index, edge_attr))
            if residual.shape == h.shape:
                h = h + residual
            h = norm(h)
        return h


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class PrioritizedReplayMemoryNumpy:
    def __init__(
        self,
        capacity: int,
        state_shape: int | tuple[int, ...],
        action_shape: int | tuple[int, ...],
        action_mask_shape: int,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.sample_count = 0
        self.memory = np.zeros(
            (capacity,),
            dtype=[
                ("state", np.float32, self._subarray_shape(state_shape)),
                ("action", np.int64, self._subarray_shape(action_shape)),
                ("next_state", np.float32, self._subarray_shape(state_shape)),
                ("reward", np.float32, (1,)),
                ("done", np.float32, (1,)),
                ("next_action_mask", np.bool_, (action_mask_shape,)),
                ("data_t", object),
                ("next_data_t", object),
                ("bag", object),
                ("next_bag", object),
            ],
        )
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.full = False

    @staticmethod
    def _subarray_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
        # NumPy's integer 1 dtype syntax has meant either scalar or length-one
        # subarray across versions. Always request an explicit feature axis.
        return (int(shape),) if isinstance(shape, (int, np.integer)) else tuple(shape)

    def push(self, *args: Any) -> None:
        self.memory[self.position] = Transition(*args)
        max_priority = self.priorities.max() if len(self) else 1.0
        self.priorities[self.position] = max(max_priority, 1.0)
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Any], list[Any], list[Any], list[Any], np.ndarray, torch.Tensor]:
        max_index = self.capacity if self.full else self.position
        priorities = self.priorities[:max_index]
        if priorities.sum() <= 0:
            probabilities = np.full(max_index, 1.0 / max_index, dtype=np.float32)
        else:
            scaled = priorities ** self.alpha
            probabilities = scaled / scaled.sum()
        indices = np.random.choice(max_index, batch_size, replace=False, p=probabilities)
        batch = self.memory[indices]
        self.sample_count += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.sample_count / self.beta_frames)
        weights = (max_index * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        def tensor_field(name: str) -> torch.Tensor:
            values = np.stack(batch[name])
            # Full training-state files may restore the old scalar-field dtype.
            # Normalize at sampling time so they can resume without losing replay.
            if values.ndim == 1:
                values = values.reshape(batch_size, 1)
            return torch.tensor(values)

        return (
            tensor_field("state"),
            tensor_field("action"),
            tensor_field("next_state"),
            tensor_field("reward"),
            tensor_field("done"),
            tensor_field("next_action_mask"),
            [item for item in batch["data_t"]],
            [item for item in batch["next_data_t"]],
            [item for item in batch["bag"]],
            [item for item in batch["next_bag"]],
            indices,
            torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        errors = td_errors.detach().abs().flatten().cpu().numpy()
        for idx, error in zip(indices, errors):
            self.priorities[idx] = float(error + self.epsilon)

    def __len__(self) -> int:
        return self.capacity if self.full else self.position


class DQNGNNNetworks(nn.Module):
    """
    Architecture scaffold for the next DQNGNN migration step.

    This file is intentionally not wired into the current training loop yet.
    It documents and validates the cleaner split:
    - eval_encoder + eval_q_net for current Q(s, a)
    - target_encoder + target_q_net for target Q(s', a')
    """

    def __init__(self, shape: DQNGNNShapeConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_encoder = GraphEncoderGINE(shape.node_in, shape.edge_in, shape.gnn_hidden)
        self.target_encoder = GraphEncoderGINE(shape.node_in, shape.edge_in, shape.gnn_hidden)
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
        encoder: GraphEncoderGINE,
        data: Any,
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
        encoder: GraphEncoderGINE,
        data_list: list[Any],
        bag_list: list[tuple[int, int, list[int]]],
        tls_state_batch: torch.Tensor,
    ) -> torch.Tensor:
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


class DQNGNNAgentRefactored:
    """
    Interface-compatible DQNGNN agent for the refactored episode runner.

    It is not wired into route_planning_main.py by default yet.
    The public methods intentionally mirror the current DQN class:
    - choose_action_gnn(...)
    - learn_dqn_neighbor()
    - concat_state_tls(...)
    - save_tmp_file(...)
    - load_full_model(...)
    """

    def __init__(
        self,
        shape: DQNGNNShapeConfig,
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
        self.networks = DQNGNNNetworks(shape, device=self.device)
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

        # Keep these aliases so existing eval/train mode code still works.
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

        model_input = self.networks.encode_action_input(
            encoder=self.networks.eval_encoder,
            data=data_t,
            source=source,
            target=target,
            neighbors=neighbors,
            tls_state=state,
        )
        q_values = self.networks.eval_q_values(model_input)
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

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device).to(torch.int64)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        next_mask = next_mask.to(self.device).bool()
        sample_weights = sample_weights.to(self.device)

        state_input = self.networks.batch_encode_action_input(
            encoder=self.networks.eval_encoder,
            data_list=data_t_list,
            bag_list=bag_list,
            tls_state_batch=state_batch,
        )
        state_action_values = self.networks.eval_q_values(state_input).gather(1, action_batch)

        with torch.no_grad():
            next_eval_input = self.networks.batch_encode_action_input(
                encoder=self.networks.eval_encoder,
                data_list=next_data_t_list,
                bag_list=next_bag_list,
                tls_state_batch=next_state_batch,
            )
            next_eval_q = self.networks.eval_q_values(next_eval_input)
            next_eval_q = next_eval_q.masked_fill(~next_mask, -1e9)
            next_action = next_eval_q.argmax(dim=1, keepdim=True)

            next_target_input = self.networks.batch_encode_action_input(
                encoder=self.networks.target_encoder,
                data_list=next_data_t_list,
                bag_list=next_bag_list,
                tls_state_batch=next_state_batch,
            )
            next_target_q = self.networks.target_q_values(next_target_input)
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
            },
            filename,
        )

    def load_full_model(
        self,
        gnn: bool = True,
        filename: str = "dqngnn_refactored.pt",
        strict: bool = True,
        load_optimizer: bool = True,
        buffer_name: str | None = None,
    ) -> int:
        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location=self.device)

        if "eval_q_net" in checkpoint:
            self._load_q_network_state(self.networks.eval_q_net, checkpoint["eval_q_net"], strict=strict)
            self._load_q_network_state(self.networks.target_q_net, checkpoint["target_q_net"], strict=strict)
            self.networks.eval_encoder.load_state_dict(checkpoint["eval_encoder"], strict=strict)
            self.networks.target_encoder.load_state_dict(checkpoint["target_encoder"], strict=strict)
        else:
            # Compatibility path for checkpoints produced by the current DQN class.
            self._load_q_network_state(self.networks.eval_q_net, checkpoint["dqn"], strict=False)
            self._load_q_network_state(self.networks.target_q_net, checkpoint["dqn_target"], strict=False)
            if gnn and "gnn" in checkpoint:
                self.networks.eval_encoder.load_state_dict(checkpoint["gnn"], strict=strict)
                self.networks.target_encoder.load_state_dict(checkpoint.get("gnn_target", checkpoint["gnn"]), strict=strict)

        if load_optimizer and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as exc:
                print(f"[warn] optimizer state could not be loaded: {exc}")
        return int(checkpoint.get("episode", 0))

    @staticmethod
    def _load_q_network_state(network: QNetwork, state_dict: dict[str, torch.Tensor], strict: bool) -> None:
        current = network.state_dict()
        compatible = {key: value for key, value in state_dict.items() if key in current and current[key].shape == value.shape}
        missing = sorted(set(current) - set(compatible))
        unexpected = sorted(set(state_dict) - set(current))
        network.load_state_dict(compatible, strict=False)
        if (missing or unexpected) and strict:
            raise RuntimeError(f"QNetwork checkpoint mismatch. missing={missing}, unexpected={unexpected}")
        if missing or unexpected:
            print(f"[warn] partially loaded QNetwork state. missing={missing}, unexpected={unexpected}")
