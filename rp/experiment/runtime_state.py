from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import math
import os
import torch


class EMAFeature:
    def __init__(self, half_life_sec: float | None = None, tau_sec: float | None = None):
        assert (half_life_sec is None) ^ (tau_sec is None)
        self.half_life = half_life_sec
        self.tau = tau_sec
        self.val: dict[str, float] = {}
        self.last_t: dict[str, float] = {}

    def _alpha(self, dt: float) -> float:
        if self.half_life is not None:
            return 1 - math.pow(0.5, dt / self.half_life)
        return 1 - math.exp(-dt / self.tau)

    def update(self, key: str, x: float, now_t: float) -> float:
        if key not in self.val:
            self.val[key] = x
            self.last_t[key] = now_t
            return x
        dt = max(0.0, now_t - self.last_t[key])
        alpha = self._alpha(dt)
        self.val[key] = alpha * x + (1 - alpha) * self.val[key]
        self.last_t[key] = now_t
        return self.val[key]


class EpisodeLogger:
    def __init__(self, log_path: Path, flush_every: int = 1):
        self.log_path = log_path
        self.flush_every = flush_every
        os.makedirs(log_path.parent, exist_ok=True)
        self._f = open(log_path, "w", encoding="utf-8")
        self._line_count = 0

    def start_episode(self, episode: int) -> None:
        self._f.write(f"\n===== Episode {episode} =====\n")
        self._f.flush()
        self._line_count = 0

    def step(self, to_node: str, node_idx: int, neighbors: list[int], action_idx: int) -> None:
        self._f.write(
            f"cur node {to_node} ({node_idx}), neighbor {neighbors}, action {action_idx}\n"
        )
        self._line_count += 1
        if self._line_count % self.flush_every == 0:
            self._f.flush()

    def decision_candidates(
        self,
        *,
        ev_id: str,
        current_edge: str,
        current_node: str,
        chosen_edge: str,
        rows: list[dict[str, Any]],
    ) -> None:
        self._f.write(
            f"decision ev={ev_id} cur_edge={current_edge} cur_node={current_node} chosen_edge={chosen_edge}\n"
        )
        for row in rows:
            self._f.write(
                "  "
                f"cand_edge={row['candidate_edge']} "
                f"next_node={row['next_node']} "
                f"tt={row['travel_time']:.3f} "
                f"speed={row['mean_speed']:.3f} "
                f"occ={row['occupancy']:.3f} "
                f"tls={row['tls_id']} "
                f"t2g={row['time_to_green']:.3f} "
                f"rg={row['remaining_green']:.3f} "
                f"halt={row['halting_number']:.3f} "
                f"veh={row['vehicle_number']:.3f}\n"
            )
        self._line_count += 1 + len(rows)
        if self._line_count % self.flush_every == 0:
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
        finally:
            self._f.close()


@dataclass
class EVRuntimeState:
    n_nodes: int
    done: bool = False
    out: bool = False
    decisions: int = 0
    total_cost: float = 0.0
    last_decision_node: str | None = None
    prev_decision_time: float | None = None
    prev_edge: str | None = None
    data_t: Any = None
    recent_nodes: deque[int] = field(default_factory=lambda: deque(maxlen=4))
    route_visit_counts: dict[int, int] = field(default_factory=dict)
    route_recent_nodes: deque[int] = field(default_factory=lambda: deque(maxlen=16))
    internal_edges: list[str] | None = None
    exit_lane_id: str | None = None
    last_committed_exit_edge: str | None = None
    current_exit_edge: str | None = None
    action_mask: Any = None
    last_action_nid: int | None = None
    last_state: Any = None
    last_est_time: float | None = None
    neighbors: list[int] | None = None
    idx: int | None = None
    neighbor_edges: list[str] | None = None
    traffic_light_state: Any = None
    routes: list[str] = field(default_factory=list)
    goal: deque[str] = field(default_factory=deque)
    time_spend: float = 0.0
    transition: dict[str, Any] = field(default_factory=dict)
    bag: Any = None
    trajectory_id: str | None = None
    trajectory_t: int = 0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        self.ever_visit = torch.zeros(self.n_nodes, 1)
        self.visit_age = torch.zeros(self.n_nodes, 1)


__all__ = ["EMAFeature", "EpisodeLogger", "EVRuntimeState"]
