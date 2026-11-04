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
        "action_mask",
        "next_action_mask",
        "data_t",
        "next_data_t",
        "bag",
        "next_bag",
    ),
)


@dataclass(frozen=True)
class SACDShapeConfig:
    num_nodes: int
    max_neighbors: int
    time_normalizer_sec: float = 3600.0
    action_feature_dim: int = 0

    @property
    def current_time_dim(self) -> int:
        return 1

    @property
    def action_feature_state_dim(self) -> int:
        return self.max_neighbors * self.action_feature_dim

    @property
    def state_dim(self) -> int:
        return self.num_nodes * 2 + self.current_time_dim + self.action_feature_state_dim


class SACDPrioritizedReplayMemoryNumpy:
    """Prioritized replay buffer for masked discrete SAC.

    The runner pushes the same transition tuple used by DQN/eDQN. This buffer
    additionally records the current action mask from the bag so SAC can compute
    policy and entropy terms over valid outgoing-neighbor actions only.
    """

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
        self.capacity = int(capacity)
        self.action_dim = int(action_mask_shape)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.epsilon = float(epsilon)
        self.sample_count = 0
        self.memory = np.zeros(
            (capacity,),
            dtype=[
                ("state", np.float32, state_shape),
                ("action", np.int64, action_shape),
                ("next_state", np.float32, state_shape),
                ("reward", np.float32, 1),
                ("done", np.float32, 1),
                ("action_mask", np.bool_, (action_mask_shape,)),
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

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_action_mask: np.ndarray,
        data_t: Any = None,
        next_data_t: Any = None,
        bag: Any = None,
        next_bag: Any = None,
    ) -> None:
        action_index = int(np.asarray(action).reshape(-1)[0])
        current_mask = self._mask_from_bag(bag, fallback_action=action_index)
        write_index = self.position
        self.memory[write_index] = Transition(
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(next_state, dtype=np.float32),
            np.asarray(reward, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            current_mask,
            np.asarray(next_action_mask, dtype=np.bool_),
            None,
            None,
            None,
            None,
        )
        max_priority = self.priorities.max() if len(self) else 1.0
        self.priorities[write_index] = max(float(max_priority), 1.0)
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
        torch.Tensor,
        np.ndarray,
        torch.Tensor,
    ]:
        max_index = len(self)
        priorities = self.priorities[:max_index]
        if priorities.sum() <= 0:
            probabilities = np.full(max_index, 1.0 / max_index, dtype=np.float32)
        else:
            scaled = priorities ** self.alpha
            probabilities = scaled / scaled.sum()
        indices = np.random.choice(max_index, batch_size, replace=False, p=probabilities)
        batch = self.memory[indices]
        self.sample_count += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.sample_count / max(self.beta_frames, 1))
        weights = (max_index * probabilities[indices]) ** (-beta)
        weights = weights / max(float(weights.max()), 1e-8)
        return (
            torch.tensor(np.stack(batch["state"])),
            torch.tensor(np.stack(batch["action"])),
            torch.tensor(np.stack(batch["next_state"])),
            torch.tensor(np.stack(batch["reward"])),
            torch.tensor(np.stack(batch["done"])),
            torch.tensor(np.stack(batch["action_mask"])),
            torch.tensor(np.stack(batch["next_action_mask"])),
            indices,
            torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        errors = td_errors.detach().abs().flatten().cpu().numpy()
        for idx, error in zip(indices, errors):
            self.priorities[int(idx)] = float(error + self.epsilon)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "action_dim": self.action_dim,
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_frames": self.beta_frames,
            "epsilon": self.epsilon,
            "sample_count": self.sample_count,
            "memory": self.memory,
            "priorities": self.priorities,
            "position": self.position,
            "full": self.full,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.action_dim = int(state.get("action_dim", self.action_dim))
        self.alpha = float(state.get("alpha", self.alpha))
        self.beta_start = float(state.get("beta_start", self.beta_start))
        self.beta_frames = int(state.get("beta_frames", self.beta_frames))
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.sample_count = int(state.get("sample_count", self.sample_count))
        memory = state.get("memory")
        if isinstance(memory, np.ndarray):
            if "action_mask" not in (memory.dtype.names or ()): 
                raise ValueError("SACD replay state is missing action_mask; start a fresh SACD run.")
            self.memory = memory
        self.position = int(state.get("position", self.position))
        self.full = bool(state.get("full", self.full))
        priorities = state.get("priorities")
        if isinstance(priorities, np.ndarray) and priorities.shape[0] == self.memory.shape[0]:
            self.priorities = priorities.astype(np.float32, copy=False)
        else:
            self.priorities = np.zeros((self.capacity,), dtype=np.float32)
            self.priorities[: len(self)] = 1.0

    def _mask_from_bag(self, bag: Any, fallback_action: int) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.bool_)
        neighbors = None
        if isinstance(bag, tuple) and len(bag) >= 3:
            neighbors = bag[2]
        if neighbors:
            mask[: min(len(neighbors), self.action_dim)] = True
        if not mask.any() and 0 <= fallback_action < self.action_dim:
            mask[fallback_action] = True
        if not mask.any():
            mask[0] = True
        return mask

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (256, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SACDNetworks(nn.Module):
    def __init__(self, shape: SACDShapeConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = MLP(shape.state_dim, shape.max_neighbors)
        self.q1 = MLP(shape.state_dim, shape.max_neighbors)
        self.q2 = MLP(shape.state_dim, shape.max_neighbors)
        self.target_q1 = MLP(shape.state_dim, shape.max_neighbors)
        self.target_q2 = MLP(shape.state_dim, shape.max_neighbors)
        self.to(self.device)
        self.sync_targets(hard=True)

    @torch.no_grad()
    def sync_targets(self, *, hard: bool = False, tau: float = 0.005) -> None:
        for target, source in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                if hard:
                    target_param.data.copy_(source_param.data)
                else:
                    target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


class _NoopTrainEval:
    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None


class SACDAgentRefactored:
    """Masked discrete Soft Actor-Critic baseline for the RP runner.

    State: current-node one-hot + target-node one-hot + normalized SUMO time
    + padded outgoing-action traffic features.
    Action: masked outgoing-neighbor index.
    Reward: SACD-specific route reward emitted by the shared episode runner.
    """

    def __init__(
        self,
        shape: SACDShapeConfig,
        *,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        memory_capacity: int = 50000,
        initial_alpha: float = 0.05,
        auto_alpha: bool = True,
        target_entropy_ratio: float = 0.30,
        replay_alpha: float = 0.0,
        use_epsilon_exploration: bool = False,
        epsilon_cap_schedule: tuple[tuple[int | None, float], ...] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.networks = SACDNetworks(shape, device=self.device)
        self.actor_optimizer = torch.optim.Adam(self.networks.actor.parameters(), lr=actor_lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.networks.q1.parameters()) + list(self.networks.q2.parameters()),
            lr=critic_lr,
            weight_decay=1e-5,
        )
        self.log_alpha = torch.tensor(np.log(max(initial_alpha, 1e-8)), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.memory = SACDPrioritizedReplayMemoryNumpy(
            memory_capacity,
            shape.state_dim,
            1,
            shape.max_neighbors,
            alpha=replay_alpha,
        )
        self.a_size = shape.max_neighbors
        self.num_nodes = shape.num_nodes
        self.use_gnn = False
        self.EPSILON = 0.9
        self.max_epsilon = 1.0
        self.min_epsilon = 0.0
        self.gamma = gamma
        self.tau = tau
        self.BATCH_SIZE = batch_size
        self.auto_alpha = auto_alpha
        self.target_entropy_ratio = target_entropy_ratio
        self.use_epsilon_exploration = bool(use_epsilon_exploration)
        self.epsilon_cap_schedule = epsilon_cap_schedule
        self.training_episode = 0
        self.learn_step_counter = 0
        self.show = True
        self.eval_net = _NoopTrainEval()
        self.target_net = self.eval_net

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def set_training_episode(self, episode: int) -> None:
        self.training_episode = int(episode)

    def effective_epsilon(self) -> float:
        if not self.use_epsilon_exploration:
            return 0.0
        epsilon = float(self.EPSILON)
        if not self.epsilon_cap_schedule:
            return epsilon
        for episode_end, cap in self.epsilon_cap_schedule:
            if episode_end is None or self.training_episode < int(episode_end):
                return min(epsilon, float(cap))
        return epsilon

    def concat_state_time(self, state: torch.Tensor, current_time_sec: float) -> torch.Tensor:
        normalized_time = float(current_time_sec) / max(float(self.shape.time_normalizer_sec), 1.0)
        time_tensor = torch.tensor([normalized_time], device=self.device, dtype=torch.float32)
        return torch.cat([state.to(self.device), time_tensor], dim=-1)

    def concat_state_action_features(self, state: torch.Tensor, action_features: torch.Tensor | None = None) -> torch.Tensor:
        feature_dim = int(self.shape.action_feature_dim)
        if feature_dim <= 0:
            return state.to(self.device)
        padded = torch.zeros((self.shape.max_neighbors, feature_dim), device=self.device, dtype=torch.float32)
        if action_features is not None and action_features.numel() > 0:
            features = action_features.to(self.device, dtype=torch.float32)
            rows = min(features.size(0), self.shape.max_neighbors)
            cols = min(features.size(1), feature_dim)
            padded[:rows, :cols] = features[:rows, :cols]
        return torch.cat([state.to(self.device), padded.reshape(-1)], dim=-1)

    def concat_state_tls(self, state: torch.Tensor, tls: list[list[float]]) -> torch.Tensor:
        return self.concat_state_time(state, 0.0)

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
        if random.random() <= self.effective_epsilon() and not evaluate:
            return random.choice(neighbors)

        logits = self.networks.actor(state.to(self.device).unsqueeze(0)).squeeze(0)
        masked_logits = logits.masked_fill(~neighbor_mask, -1e9)
        if evaluate:
            action_index = int(masked_logits.argmax().item())
        else:
            probs = torch.softmax(masked_logits, dim=-1)
            action_index = int(torch.distributions.Categorical(probs=probs).sample().item())
        if action_index >= len(neighbors):
            action_index = int(torch.where(neighbor_mask)[0][0].item())
        return neighbors[action_index]

    def learn_dqn_neighbor(self) -> float | None:
        if len(self.memory) < self.BATCH_SIZE:
            return None
        if self.show:
            print("learning..")
            self.show = False

        (
            state_batch,
            action_batch,
            next_state_batch,
            reward_batch,
            done_batch,
            action_mask,
            next_action_mask,
            sample_indices,
            sample_weights,
        ) = self.memory.sample(self.BATCH_SIZE)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device).long()
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        action_mask = action_mask.to(self.device).bool()
        next_action_mask = next_action_mask.to(self.device).bool()
        sample_weights = sample_weights.to(self.device)

        with torch.no_grad():
            next_probs, next_log_probs = self._masked_policy(next_state_batch, next_action_mask)
            next_q1 = self.networks.target_q1(next_state_batch)
            next_q2 = self.networks.target_q2(next_state_batch)
            next_min_q = torch.min(next_q1, next_q2)
            next_soft_value = (next_probs * (next_min_q - self.alpha.detach() * next_log_probs)).sum(dim=1, keepdim=True)
            q_target = reward_batch + (1.0 - done_batch) * self.gamma * next_soft_value

        current_q1 = self.networks.q1(state_batch).gather(1, action_batch)
        current_q2 = self.networks.q2(state_batch).gather(1, action_batch)
        td_error_q1 = current_q1 - q_target
        td_error_q2 = current_q2 - q_target
        q1_loss = (
            F.smooth_l1_loss(current_q1, q_target, reduction="none", beta=1.0)
            * sample_weights
        ).mean()
        q2_loss = (
            F.smooth_l1_loss(current_q2, q_target, reduction="none", beta=1.0)
            * sample_weights
        ).mean()
        critic_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.networks.q1.parameters()) + list(self.networks.q2.parameters()),
            max_norm=10.0,
        )
        self.q_optimizer.step()
        priority_errors = 0.5 * (td_error_q1.detach().abs() + td_error_q2.detach().abs())
        self.memory.update_priorities(sample_indices, priority_errors)

        probs, log_probs = self._masked_policy(state_batch, action_mask)
        with torch.no_grad():
            min_q = torch.min(self.networks.q1(state_batch), self.networks.q2(state_batch))
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        alpha_loss_value = torch.tensor(0.0, device=self.device)
        if self.auto_alpha:
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=1)
            valid_counts = action_mask.float().sum(dim=1).clamp_min(1.0)
            target_entropy = self.target_entropy_ratio * torch.log(valid_counts)
            alpha_loss = (self.log_alpha * (entropy - target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = alpha_loss.detach()

        self.networks.sync_targets(hard=False, tau=self.tau)
        self.learn_step_counter += 1
        total_loss = critic_loss.detach() + actor_loss.detach() + alpha_loss_value
        return float(total_loss.item())

    def get_training_state(self) -> dict[str, Any]:
        return {
            "actor": self.networks.actor.state_dict(),
            "q1": self.networks.q1.state_dict(),
            "q2": self.networks.q2.state_dict(),
            "target_q1": self.networks.target_q1.state_dict(),
            "target_q2": self.networks.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "memory": self.memory.state_dict(),
            "learn_step_counter": self.learn_step_counter,
            "epsilon": self.EPSILON,
            "training_episode": self.training_episode,
        }

    def load_training_state(self, state: dict[str, Any]) -> None:
        self.networks.actor.load_state_dict(state["actor"])
        self.networks.q1.load_state_dict(state["q1"])
        self.networks.q2.load_state_dict(state["q2"])
        self.networks.target_q1.load_state_dict(state["target_q1"])
        self.networks.target_q2.load_state_dict(state["target_q2"])
        if state.get("actor_optimizer") is not None:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self._move_optimizer_state_to_device(self.actor_optimizer)
        if state.get("q_optimizer") is not None:
            self.q_optimizer.load_state_dict(state["q_optimizer"])
            self._move_optimizer_state_to_device(self.q_optimizer)
        if state.get("alpha_optimizer") is not None:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            self._move_optimizer_state_to_device(self.alpha_optimizer)
        if state.get("log_alpha") is not None:
            self.log_alpha.data.copy_(torch.as_tensor(state["log_alpha"], device=self.device, dtype=torch.float32))
        if state.get("memory") is not None:
            self.memory.load_state_dict(state["memory"])
        self.learn_step_counter = int(state.get("learn_step_counter", 0))
        self.training_episode = int(state.get("training_episode", self.training_episode))
        if state.get("epsilon") is not None:
            self.EPSILON = float(state["epsilon"])

    def save_tmp_file(self, ep: int, filename: str = "tmp_file.pt") -> None:
        torch.save(
            {
                "episode": ep,
                "shape": self.shape,
                "agent_type": "sacd_refactored",
                "agent_state": self.get_training_state(),
            },
            filename,
        )

    def load_full_model(
        self,
        gnn: bool = False,
        filename: str = "sacd_refactored.pt",
        strict: bool = True,
        load_optimizer: bool = True,
        buffer_name: str | None = None,
    ) -> int:
        try:
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location=self.device)
        if "agent_state" in checkpoint:
            self.load_training_state(checkpoint["agent_state"])
        else:
            self.load_training_state(checkpoint)
        return int(checkpoint.get("episode", 0))

    def _masked_policy(self, states: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_masks = masks.clone()
        empty_rows = ~safe_masks.any(dim=1)
        if empty_rows.any():
            safe_masks[empty_rows, 0] = True
        logits = self.networks.actor(states)
        masked_logits = logits.masked_fill(~safe_masks, -1e9)
        probs = torch.softmax(masked_logits, dim=1)
        log_probs = torch.log_softmax(masked_logits, dim=1)
        probs = probs * safe_masks.float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
        log_probs = log_probs.masked_fill(~safe_masks, 0.0)
        return probs, log_probs

    def _move_optimizer_state_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for optimizer_state in optimizer.state.values():
            for key, value in list(optimizer_state.items()):
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)
