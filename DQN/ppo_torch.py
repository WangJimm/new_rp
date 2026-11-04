from __future__ import annotations

"""Pure PyTorch PPO baseline for masked route-planning actions.

This implementation deliberately has no Ray/RLlib dependency. It uses the same
state, local outgoing-neighbor action order, rewards, and per-EV trajectory
boundaries as the RLlib baseline.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from .ppo_common import PPORTrajectoryMemory, compute_gae


@dataclass(frozen=True)
class TorchPPOShapeConfig:
    num_nodes: int
    max_neighbors: int
    time_normalizer_sec: float = 3600.0

    @property
    def current_time_dim(self) -> int:
        return 1

    @property
    def ppo_input_dim(self) -> int:
        return self.num_nodes * 2 + self.current_time_dim


class MaskedActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_masks = action_masks.bool().clone()
        empty_rows = ~safe_masks.any(dim=-1)
        if empty_rows.any():
            safe_masks[empty_rows, 0] = True
        features = self.feature(states)
        logits = self.policy_head(features).masked_fill(~safe_masks, -1e9)
        values = self.value_head(features).squeeze(-1)
        return logits, values


class TorchPPOAgentRefactored:
    """Masked discrete PPO implemented with only PyTorch and NumPy."""

    def __init__(
        self,
        shape: TorchPPOShapeConfig,
        *,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        train_batch_size: int = 4096,
        sgd_minibatch_size: int = 512,
        num_sgd_iter: int = 10,
        clip_param: float = 0.2,
        entropy_coeff: float = 0.01,
        vf_loss_coeff: float = 1.0,
        vf_clip_param: float = 10.0,
        initial_kl_coeff: float = 0.2,
        kl_target: float = 0.01,
        grad_clip: float = 10.0,
        memory_capacity: int = 20000,
        hidden_dim: int = 128,
        device: torch.device | None = None,
    ) -> None:
        self.shape = shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a_size = int(shape.max_neighbors)
        self.num_nodes = int(shape.num_nodes)
        self.use_gnn = False
        self.EPSILON = 0.0
        self.max_epsilon = 0.0
        self.min_epsilon = 0.0

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.train_batch_size = int(train_batch_size)
        self.sgd_minibatch_size = min(int(sgd_minibatch_size), self.train_batch_size)
        self.num_sgd_iter = int(num_sgd_iter)
        self.clip_param = float(clip_param)
        self.entropy_coeff = float(entropy_coeff)
        self.vf_loss_coeff = float(vf_loss_coeff)
        self.vf_clip_param = float(vf_clip_param)
        self.kl_coeff = float(initial_kl_coeff)
        self.kl_target = float(kl_target)
        self.grad_clip = float(grad_clip)

        self.network = MaskedActorCritic(
            shape.ppo_input_dim,
            shape.max_neighbors,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.memory = PPORTrajectoryMemory(shape.max_neighbors, capacity=memory_capacity)

        # Compatibility with the shared train/eval/checkpoint orchestration.
        self.eval_net = self.network
        self.target_net = self.network
        self.update_attempt_counter = 0
        self.learn_step_counter = 0
        self.gradient_step_counter = 0
        self.last_update_metrics: dict[str, float] = {}

    def concat_state_time(self, state: torch.Tensor, current_time_sec: float) -> torch.Tensor:
        normalized_time = float(current_time_sec) / max(float(self.shape.time_normalizer_sec), 1.0)
        time_tensor = torch.tensor([normalized_time], device=self.device, dtype=torch.float32)
        return torch.cat([state.to(self.device), time_tensor], dim=-1)

    def concat_state_tls(self, state: torch.Tensor, tls: list[list[float]]) -> torch.Tensor:
        return self.concat_state_time(state, 0.0)

    def _pad_neighbors(self, neighbors: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not neighbors:
            return (
                torch.zeros(self.a_size, dtype=torch.long, device=self.device),
                torch.zeros(self.a_size, dtype=torch.bool, device=self.device),
            )
        padded = neighbors + [neighbors[0]] * max(0, self.a_size - len(neighbors))
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
        local_mask = self._mask_from_neighbors(neighbors)
        state_batch = state.to(self.device, dtype=torch.float32).reshape(1, -1)
        mask_batch = torch.as_tensor(local_mask, device=self.device, dtype=torch.bool).reshape(1, -1)
        self.network.eval() if evaluate else self.network.train()
        logits, _ = self.network(state_batch, mask_batch)
        if evaluate:
            action_index = int(torch.argmax(logits, dim=-1).item())
        else:
            action_index = int(Categorical(logits=logits).sample().item())
        if action_index >= len(neighbors):
            action_index = 0
        return neighbors[action_index]

    def learn_dqn_neighbor(self) -> float | None:
        if self.memory.complete_size < self.train_batch_size:
            return None
        trajectories = self.memory.peek_complete()
        if not trajectories:
            return None
        self.update_attempt_counter += 1

        rollout = self._prepare_rollout_batch(trajectories)
        advantages = rollout["advantages"]
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        rollout["advantages"] = advantages

        batch_size = int(rollout["states"].shape[0])
        metrics: dict[str, list[float]] = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "clip_fraction": [],
            "grad_norm": [],
        }
        self.network.train()
        epochs_completed = 0
        for _ in range(self.num_sgd_iter):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.sgd_minibatch_size):
                indices = permutation[start : start + self.sgd_minibatch_size]
                states = rollout["states"][indices]
                masks = rollout["action_masks"][indices]
                actions = rollout["actions"][indices]
                old_logp = rollout["old_logp"][indices]
                old_logits = rollout["old_logits"][indices]
                minibatch_advantages = rollout["advantages"][indices]
                returns = rollout["returns"][indices]

                logits, values = self.network(states, masks)
                distribution = Categorical(logits=logits)
                old_distribution = Categorical(logits=old_logits)
                new_logp = distribution.log_prob(actions)
                log_ratio = new_logp - old_logp
                ratio = log_ratio.exp()

                surrogate1 = ratio * minibatch_advantages
                surrogate2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                ) * minibatch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_error_squared = (values - returns).pow(2)
                value_loss = torch.clamp(
                    value_error_squared,
                    min=0.0,
                    max=self.vf_clip_param,
                ).mean()

                entropy = distribution.entropy().mean()
                mean_kl = kl_divergence(old_distribution, distribution).mean()
                total_loss = (
                    policy_loss
                    + self.vf_loss_coeff * value_loss
                    - self.entropy_coeff * entropy
                    + self.kl_coeff * mean_kl
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.grad_clip,
                )
                self.optimizer.step()
                self.gradient_step_counter += 1

                metrics["total_loss"].append(float(total_loss.detach().cpu().item()))
                metrics["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                metrics["value_loss"].append(float(value_loss.detach().cpu().item()))
                metrics["entropy"].append(float(entropy.detach().cpu().item()))
                metrics["kl"].append(float(mean_kl.detach().cpu().item()))
                metrics["clip_fraction"].append(
                    float(((ratio - 1.0).abs() > self.clip_param).float().mean().detach().cpu().item())
                )
                metrics["grad_norm"].append(float(torch.as_tensor(grad_norm).detach().cpu().item()))
            epochs_completed += 1

        if not metrics["total_loss"]:
            raise RuntimeError("PyTorch PPO completed no minibatch gradient updates")

        mean_kl_value = float(np.mean(metrics["kl"]))
        if mean_kl_value > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif mean_kl_value < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        self.kl_coeff = float(np.clip(self.kl_coeff, 1e-5, 1000.0))

        trajectory_ids = {
            int(trajectory[0]["trajectory_id"])
            for trajectory in trajectories
            if trajectory
        }
        self.memory.remove_trajectories(trajectory_ids)
        self.learn_step_counter += 1
        self.last_update_metrics = {
            key: float(np.mean(values))
            for key, values in metrics.items()
        }
        self.last_update_metrics.update(
            {
                "batch_size": float(batch_size),
                "epochs_completed": float(epochs_completed),
                "kl_coeff": float(self.kl_coeff),
            }
        )
        return self.last_update_metrics["total_loss"]

    def _prepare_rollout_batch(
        self,
        trajectories: list[list[dict[str, Any]]],
    ) -> dict[str, torch.Tensor]:
        batches: dict[str, list[np.ndarray]] = {
            "states": [],
            "action_masks": [],
            "actions": [],
            "old_logits": [],
            "old_logp": [],
            "old_values": [],
            "advantages": [],
            "returns": [],
        }
        self.network.eval()
        for trajectory in trajectories:
            states_np = np.stack([item["state"] for item in trajectory]).astype(np.float32)
            masks_np = np.stack([item["action_mask"] for item in trajectory]).astype(np.bool_)
            actions_np = np.asarray([item["action"] for item in trajectory], dtype=np.int64)
            rewards_np = np.asarray([item["reward"] for item in trajectory], dtype=np.float32)
            terminateds_np = np.asarray([item["terminated"] for item in trajectory], dtype=np.bool_)

            states = torch.as_tensor(states_np, device=self.device, dtype=torch.float32)
            masks = torch.as_tensor(masks_np, device=self.device, dtype=torch.bool)
            actions = torch.as_tensor(actions_np, device=self.device, dtype=torch.long)
            with torch.no_grad():
                old_logits, old_values = self.network(states, masks)
                old_logp = Categorical(logits=old_logits).log_prob(actions)

                final_item = trajectory[-1]
                bootstrap_value = 0.0
                if bool(final_item["truncated"]) and not bool(final_item["terminated"]):
                    next_state = torch.as_tensor(
                        np.asarray(final_item["next_state"], dtype=np.float32),
                        device=self.device,
                    ).reshape(1, -1)
                    next_mask = torch.as_tensor(
                        np.asarray(final_item["next_action_mask"], dtype=np.bool_),
                        device=self.device,
                    ).reshape(1, -1)
                    _, next_value = self.network(next_state, next_mask)
                    bootstrap_value = float(next_value.item())

            advantages_np, returns_np = compute_gae(
                rewards_np,
                old_values.detach().cpu().numpy(),
                terminateds_np,
                bootstrap_value=bootstrap_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            batches["states"].append(states_np)
            batches["action_masks"].append(masks_np)
            batches["actions"].append(actions_np)
            batches["old_logits"].append(old_logits.detach().cpu().numpy().astype(np.float32))
            batches["old_logp"].append(old_logp.detach().cpu().numpy().astype(np.float32))
            batches["old_values"].append(old_values.detach().cpu().numpy().astype(np.float32))
            batches["advantages"].append(advantages_np)
            batches["returns"].append(returns_np)

        return {
            "states": torch.as_tensor(np.concatenate(batches["states"]), device=self.device, dtype=torch.float32),
            "action_masks": torch.as_tensor(np.concatenate(batches["action_masks"]), device=self.device, dtype=torch.bool),
            "actions": torch.as_tensor(np.concatenate(batches["actions"]), device=self.device, dtype=torch.long),
            "old_logits": torch.as_tensor(np.concatenate(batches["old_logits"]), device=self.device, dtype=torch.float32),
            "old_logp": torch.as_tensor(np.concatenate(batches["old_logp"]), device=self.device, dtype=torch.float32),
            "old_values": torch.as_tensor(np.concatenate(batches["old_values"]), device=self.device, dtype=torch.float32),
            "advantages": torch.as_tensor(np.concatenate(batches["advantages"]), device=self.device, dtype=torch.float32),
            "returns": torch.as_tensor(np.concatenate(batches["returns"]), device=self.device, dtype=torch.float32),
        }

    def _mask_from_neighbors(self, neighbors: list[int]) -> np.ndarray:
        mask = np.zeros(self.a_size, dtype=np.float32)
        mask[: min(len(neighbors), self.a_size)] = 1.0
        return mask

    def get_training_state(self) -> dict[str, Any]:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory": self.memory.state_dict(),
            "update_attempt_counter": self.update_attempt_counter,
            "learn_step_counter": self.learn_step_counter,
            "gradient_step_counter": self.gradient_step_counter,
            "kl_coeff": self.kl_coeff,
            "last_update_metrics": self.last_update_metrics,
        }

    def load_training_state(self, state: dict[str, Any]) -> None:
        self.network.load_state_dict(state["network"])
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
            self._move_optimizer_state_to_device()
        if state.get("memory") is not None:
            self.memory.load_state_dict(state["memory"])
        self.update_attempt_counter = int(state.get("update_attempt_counter", 0))
        self.learn_step_counter = int(state.get("learn_step_counter", 0))
        self.gradient_step_counter = int(state.get("gradient_step_counter", 0))
        self.kl_coeff = float(state.get("kl_coeff", self.kl_coeff))
        self.last_update_metrics = {
            str(key): float(value)
            for key, value in dict(state.get("last_update_metrics", {})).items()
        }

    def _move_optimizer_state_to_device(self) -> None:
        for optimizer_state in self.optimizer.state.values():
            for key, value in list(optimizer_state.items()):
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)

    def save_tmp_file(self, ep: int, filename: str = "tmp_file.pt") -> None:
        torch.save(
            {
                "episode": ep,
                "shape": self.shape,
                "agent_type": "ppo_torch_refactored",
                "agent_state": self.get_training_state(),
            },
            filename,
        )

    def load_full_model(
        self,
        gnn: bool = False,
        filename: str = "ppo_torch_refactored.pt",
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
