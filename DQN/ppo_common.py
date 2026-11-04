from __future__ import annotations

"""Shared on-policy rollout utilities for PPO implementations."""

from typing import Any, Iterable

import numpy as np


class PPORTrajectoryMemory:
    """On-policy rollout buffer that preserves independent EV trajectories."""

    def __init__(self, action_dim: int, capacity: int = 20000) -> None:
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.items: list[dict[str, Any]] = []
        self.trajectory_ids: dict[str, int] = {}
        self.next_trajectory_id = 0

    def __len__(self) -> int:
        return len(self.items)

    @property
    def complete_size(self) -> int:
        complete_ids = {
            int(item["trajectory_id"])
            for item in self.items
            if bool(item["terminated"]) or bool(item["truncated"])
        }
        return sum(int(item["trajectory_id"]) in complete_ids for item in self.items)

    def clear(self) -> None:
        self.items.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "items": self.items,
            "trajectory_ids": self.trajectory_ids,
            "next_trajectory_id": self.next_trajectory_id,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.action_dim = int(state.get("action_dim", self.action_dim))
        self.capacity = int(state.get("capacity", self.capacity))
        items = list(state.get("items", []))
        if items and any("trajectory_id" not in item for item in items):
            raise ValueError(
                "PPO rollout state predates per-EV trajectory metadata; start a fresh PPO run."
            )
        self.items = items
        self.trajectory_ids = {
            str(key): int(value)
            for key, value in dict(state.get("trajectory_ids", {})).items()
        }
        default_next = 1 + max(
            (int(item["trajectory_id"]) for item in self.items),
            default=-1,
        )
        self.next_trajectory_id = int(state.get("next_trajectory_id", default_next))

    def push(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "PPO transitions require trajectory metadata; use push_trajectory via the episode runner."
        )

    def push_trajectory(
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
        *,
        trajectory_key: str,
        timestep: int,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        trajectory_id = self._trajectory_id(trajectory_key)
        action_index = int(np.asarray(action).reshape(-1)[0])
        current_mask = self._mask_from_bag(bag, fallback_action=action_index)
        item: dict[str, Any] = {
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action_index, dtype=np.int64),
            "next_state": np.asarray(next_state, dtype=np.float32),
            "reward": np.asarray(float(np.asarray(reward).reshape(-1)[0]), dtype=np.float32),
            "done": np.asarray(bool(terminated or truncated), dtype=np.bool_),
            "terminated": np.asarray(bool(terminated), dtype=np.bool_),
            "truncated": np.asarray(bool(truncated), dtype=np.bool_),
            "action_mask": current_mask.astype(np.float32),
            "next_action_mask": np.asarray(next_action_mask, dtype=np.float32),
            "trajectory_id": trajectory_id,
            "timestep": int(timestep),
        }
        self.items.append(item)
        self._trim_to_capacity()

    def drain_complete(self) -> list[list[dict[str, Any]]]:
        trajectories = self.peek_complete()
        self.remove_trajectories(
            int(trajectory[0]["trajectory_id"])
            for trajectory in trajectories
            if trajectory
        )
        return trajectories

    def peek_complete(self) -> list[list[dict[str, Any]]]:
        """Return validated complete trajectories without removing them."""
        complete_ids = {
            int(item["trajectory_id"])
            for item in self.items
            if bool(item["terminated"]) or bool(item["truncated"])
        }
        if not complete_ids:
            return []

        completed = [item for item in self.items if int(item["trajectory_id"]) in complete_ids]
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in completed:
            grouped.setdefault(int(item["trajectory_id"]), []).append(item)

        trajectories: list[list[dict[str, Any]]] = []
        for trajectory_id, trajectory in grouped.items():
            trajectory.sort(key=lambda item: int(item["timestep"]))
            timesteps = [int(item["timestep"]) for item in trajectory]
            if timesteps != list(range(len(trajectory))):
                raise ValueError(
                    f"PPO trajectory {trajectory_id} has non-contiguous timesteps: {timesteps}"
                )
            terminal_flags = [
                bool(item["terminated"]) or bool(item["truncated"])
                for item in trajectory
            ]
            if any(terminal_flags[:-1]) or not terminal_flags[-1]:
                raise ValueError(
                    f"PPO trajectory {trajectory_id} must terminate exactly at its final transition"
                )
            trajectories.append(trajectory)
        return trajectories

    def remove_trajectories(self, trajectory_ids: Iterable[int]) -> None:
        remove_ids = {int(trajectory_id) for trajectory_id in trajectory_ids}
        if not remove_ids:
            return
        self.items = [
            item
            for item in self.items
            if int(item["trajectory_id"]) not in remove_ids
        ]

    def _trajectory_id(self, trajectory_key: str) -> int:
        key = str(trajectory_key)
        if key not in self.trajectory_ids:
            self.trajectory_ids[key] = self.next_trajectory_id
            self.next_trajectory_id += 1
        return self.trajectory_ids[key]

    def _trim_to_capacity(self) -> None:
        while len(self.items) > self.capacity:
            oldest_id = int(self.items[0]["trajectory_id"])
            self.items = [
                item for item in self.items
                if int(item["trajectory_id"]) != oldest_id
            ]

    def _mask_from_bag(self, bag: Any, fallback_action: int) -> np.ndarray:
        mask = np.zeros(self.action_dim, dtype=np.float32)
        neighbors = None
        if isinstance(bag, tuple) and len(bag) >= 3:
            neighbors = bag[2]
        if neighbors is not None and len(neighbors) > 0:
            mask[: min(len(neighbors), self.action_dim)] = 1.0
        if not mask.any() and 0 <= fallback_action < self.action_dim:
            mask[fallback_action] = 1.0
        if not mask.any():
            mask[0] = 1.0
        return mask


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminateds: np.ndarray,
    *,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE for exactly one ordered trajectory.

    `terminateds=True` cuts bootstrapping. A time-limit truncation should pass
    `terminateds=False` and provide the value of the final next observation.
    """
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    terminateds = np.asarray(terminateds, dtype=np.bool_).reshape(-1)
    if not (len(rewards) == len(values) == len(terminateds)):
        raise ValueError("rewards, values, and terminateds must have equal lengths")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = float(bootstrap_value)
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if bool(terminateds[index]) else 1.0
        delta = float(rewards[index]) + gamma * next_value * nonterminal - float(values[index])
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = float(values[index])
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)
