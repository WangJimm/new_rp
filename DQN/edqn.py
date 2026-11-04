from __future__ import annotations

import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
class EDQNShapeConfig:
    num_nodes: int
    max_neighbors: int
    time_normalizer_sec: float = 3600.0

    @property
    def tls_state_dim(self) -> int:
        return 0

    @property
    def current_time_dim(self) -> int:
        return 1

    @property
    def dqn_input_dim(self) -> int:
        return self.num_nodes * 2 + self.current_time_dim


class UniformReplayMemoryNumpy:
    """Uniform replay buffer with the same sample interface used by the runner.

    Jiang et al.'s eDQN uses experience replay with mini-batch sampling. This
    buffer intentionally does not apply PER so that the baseline reflects the
    eDQN-side changes: Double target, competitive/dueling Q network, and Huber
    loss.
    """

    def __init__(
        self,
        capacity: int,
        state_shape: int | tuple[int, ...],
        action_shape: int | tuple[int, ...],
        action_mask_shape: int,
    ) -> None:
        self.capacity = capacity
        self.memory = np.zeros(
            (capacity,),
            dtype=[
                ("state", np.float32, state_shape),
                ("action", np.int64, action_shape),
                ("next_state", np.float32, state_shape),
                ("reward", np.float32, 1),
                ("done", np.float32, 1),
                ("next_action_mask", np.bool_, (action_mask_shape,)),
                ("data_t", object),
                ("next_data_t", object),
                ("bag", object),
                ("next_bag", object),
            ],
        )
        self.position = 0
        self.full = False

    def push(self, *args: Any) -> None:
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

    def sample(
        self,
        batch_size: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[Any],
        list[Any],
        list[Any],
        list[Any],
        np.ndarray,
        torch.Tensor,
    ]:
        max_index = len(self)
        indices = np.random.choice(max_index, batch_size, replace=False)
        batch = self.memory[indices]
        weights = torch.ones(batch_size, 1, dtype=torch.float32)
        return (
            torch.tensor(np.stack(batch["state"])),
            torch.tensor(np.stack(batch["action"])),
            torch.tensor(np.stack(batch["next_state"])),
            torch.tensor(np.stack(batch["reward"])),
            torch.tensor(np.stack(batch["done"])),
            torch.tensor(np.stack(batch["next_action_mask"])),
            [item for item in batch["data_t"]],
            [item for item in batch["next_data_t"]],
            [item for item in batch["bag"]],
            [item for item in batch["next_bag"]],
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        return None


class CompetitiveQNetwork(nn.Module):
    """Competitive/dueling Q network used by eDQN.

    The shared feature extractor follows the HEDQN paper's four hidden-layer
    setting. The two output branches estimate V(s) and A(s,a), then combine them
    as Q(s,a)=V(s)+A(s,a)-mean_a A(s,a).
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, int, int, int] = (256, 128, 64, 16),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.feature = nn.Sequential(*layers)
        self.value = nn.Linear(prev_dim, 1)
        self.advantage = nn.Linear(prev_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        value = self.value(h)
        advantage = self.advantage(h)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class EDQNNetworks(nn.Module):
    def __init__(self, shape: EDQNShapeConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_q_net = CompetitiveQNetwork(shape.dqn_input_dim, shape.max_neighbors)
        self.target_q_net = CompetitiveQNetwork(shape.dqn_input_dim, shape.max_neighbors)
        self.to(self.device)
        self.sync_target()

    @torch.no_grad()
    def sync_target(self) -> None:
        self.target_q_net.load_state_dict(self.eval_q_net.state_dict())

    def eval_q_values(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.eval_q_net(model_input)

    @torch.no_grad()
    def target_q_values(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.target_q_net(model_input)


class EDQNAgentRefactored:
    """Lower-level route-planning eDQN baseline.

    This agent keeps the existing RP runner interface while matching the eDQN
    mechanisms described in Jiang et al.: deep double-Q target, competitive
    Q-network structure, and Huber loss. It intentionally remains non-GNN.
    """

    def __init__(
        self,
        shape: EDQNShapeConfig,
        *,
        lr: float = 1e-5,
        gamma: float = 0.95,
        batch_size: int = 128,
        memory_capacity: int = 20000,
        target_update_interval: int = 200,
        huber_delta: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networks = EDQNNetworks(shape, device=self.device)
        self.optimizer = torch.optim.Adam(self.networks.eval_q_net.parameters(), lr=lr)
        self.memory = UniformReplayMemoryNumpy(
            memory_capacity,
            shape.dqn_input_dim,
            1,
            shape.max_neighbors,
        )
        self.a_size = shape.max_neighbors
        self.num_nodes = shape.num_nodes
        self.use_gnn = False
        self.EPSILON = 0.9
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.gamma = gamma
        self.BATCH_SIZE = batch_size
        self.target_update_interval = target_update_interval
        self.huber_delta = huber_delta
        self.learn_step_counter = 0
        self.show = True

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

    def concat_state_time(self, state: torch.Tensor, current_time_sec: float) -> torch.Tensor:
        normalized_time = float(current_time_sec) / max(float(self.shape.time_normalizer_sec), 1.0)
        time_tensor = torch.tensor([normalized_time], device=self.device, dtype=torch.float32)
        return torch.cat([state.to(self.device), time_tensor], dim=-1)

    def concat_state_tls(self, state: torch.Tensor, tls: list[list[float]]) -> torch.Tensor:
        return self.concat_state_time(state, 0.0)

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

        q_values = self.networks.eval_q_values(state.to(self.device))
        q_values = q_values.masked_fill(~neighbor_mask, -1e9)
        return neighbors[int(q_values.argmax())]

    def learn_dqn_neighbor(self) -> float | None:
        if len(self.memory) < self.BATCH_SIZE:
            return None
        if self.show:
            print("learning..")
            self.show = False

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

        state_action_values = self.networks.eval_q_values(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_eval_q = self.networks.eval_q_values(next_state_batch)
            next_eval_q = next_eval_q.masked_fill(~next_mask, -1e9)
            next_action = next_eval_q.argmax(dim=1, keepdim=True)

            next_target_q = self.networks.target_q_values(next_state_batch)
            next_target_q = next_target_q.masked_fill(~next_mask, -1e9)
            selected_next_q = next_target_q.gather(1, next_action)
            target = reward_batch + (1.0 - done_batch) * self.gamma * selected_next_q

        elementwise_loss = F.smooth_l1_loss(
            state_action_values,
            target,
            reduction="none",
            beta=self.huber_delta,
        )
        loss = (elementwise_loss * sample_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks.eval_q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_interval == 0:
            self.networks.sync_target()
        return float(loss.item())

    def save_tmp_file(self, ep: int, filename: str = "tmp_file.pt") -> None:
        torch.save(
            {
                "episode": ep,
                "eval_q_net": self.networks.eval_q_net.state_dict(),
                "target_q_net": self.networks.target_q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "shape": self.shape,
                "agent_type": "edqn_refactored",
            },
            filename,
        )

    def load_full_model(
        self,
        gnn: bool = False,
        filename: str = "edqn_refactored.pt",
        strict: bool = True,
        load_optimizer: bool = True,
        buffer_name: str | None = None,
    ) -> int:
        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location=self.device)

        self.networks.eval_q_net.load_state_dict(checkpoint["eval_q_net"], strict=strict)
        self.networks.target_q_net.load_state_dict(checkpoint["target_q_net"], strict=strict)
        if load_optimizer and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as exc:
                print(f"[warn] optimizer state could not be loaded: {exc}")
        return int(checkpoint.get("episode", 0))
