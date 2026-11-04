from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from DQN.dqngnn_re import PrioritizedReplayMemoryNumpy, QNetwork


@dataclass(frozen=True)
class DQNShapeConfig:
    num_nodes: int
    max_neighbors: int

    @property
    def tls_state_dim(self) -> int:
        return self.max_neighbors * 2

    @property
    def dqn_input_dim(self) -> int:
        return self.num_nodes * 2 + self.tls_state_dim


class DQNNetworks(nn.Module):
    def __init__(self, shape: DQNShapeConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_q_net = QNetwork(shape.dqn_input_dim, shape.max_neighbors)
        self.target_q_net = QNetwork(shape.dqn_input_dim, shape.max_neighbors)
        self.to(self.device)
        self.sync_target(hard=True)

    @torch.no_grad()
    def sync_target(self, hard: bool = False, tau: float = 0.003) -> None:
        for target_param, source_param in zip(self.target_q_net.parameters(), self.eval_q_net.parameters()):
            if hard:
                target_param.data.copy_(source_param.data)
            else:
                target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def eval_q_values(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.eval_q_net(model_input)

    @torch.no_grad()
    def target_q_values(self, model_input: torch.Tensor) -> torch.Tensor:
        return self.target_q_net(model_input)


class DQNAgentRefactored:
    """
    Non-GNN DQN agent compatible with the current refactored episode runner.

    It keeps the same training improvements as the refactored DQNGNN agent:
    - Double DQN target
    - Dueling QNetwork
    - Prioritized Experience Replay
    """

    def __init__(
        self,
        shape: DQNShapeConfig,
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
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networks = DQNNetworks(shape, device=self.device)
        self.optimizer = torch.optim.Adam(self.networks.eval_q_net.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_func = nn.MSELoss(reduction="none")
        self.memory = PrioritizedReplayMemoryNumpy(
            memory_capacity,
            shape.dqn_input_dim,
            1,
            shape.max_neighbors,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
        )
        self.a_size = shape.max_neighbors
        self.num_nodes = shape.num_nodes
        self.use_gnn = False
        self.EPSILON = 0.9
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.gamma = gamma
        self.BATCH_SIZE = batch_size
        self.target_tau = target_tau
        self.learn_step_counter = 0
        self.show = True

        # Keep aliases expected by shared eval/train and checkpoint helpers.
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
        return torch.cat([state.to(self.device), tls_state.flatten()], dim=-1)

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

        td_errors = state_action_values - target
        loss = (self.loss_func(state_action_values, target) * sample_weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks.eval_q_net.parameters(), max_norm=10.0)
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
                "optimizer": self.optimizer.state_dict(),
                "shape": self.shape,
                "agent_type": "dqn_refactored",
            },
            filename,
        )

    def load_full_model(
        self,
        gnn: bool = False,
        filename: str = "dqn_refactored.pt",
        strict: bool = True,
        load_optimizer: bool = True,
        buffer_name: str | None = None,
    ) -> int:
        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location=self.device)

        if "eval_q_net" in checkpoint:
            self.networks.eval_q_net.load_state_dict(checkpoint["eval_q_net"], strict=strict)
            self.networks.target_q_net.load_state_dict(checkpoint["target_q_net"], strict=strict)
        else:
            self._load_q_network_state(self.networks.eval_q_net, checkpoint["dqn"], strict=False)
            self._load_q_network_state(self.networks.target_q_net, checkpoint["dqn_target"], strict=False)

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
