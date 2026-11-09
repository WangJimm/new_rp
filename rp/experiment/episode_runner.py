from __future__ import annotations

import uuid
import random
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import torch

from .runtime_state import EVRuntimeState, EpisodeLogger
from rp.graph import (
    build_data_runtime,
    dynamic_node_feature_update,
    freeze_pyg_data,
)
from rp.routing import dijkstra_next_hop, dijkstra_path
from rp.sumo import (
    action_mask_dqn,
    apply_next_junction,
    candidate_next_junctions,
    find_tls_and_link_index_for_movement,
    lane2edge,
    remaining_routes_len,
    time_to_green_and_remaining_green,
)


class AgentLearningError(RuntimeError):
    """Fatal learner/update error that must not be skipped as a SUMO episode failure."""


TransitionPayload: TypeAlias = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
    Any,
    Any,
    Any,
]


@dataclass(frozen=True)
class EpisodeConfig:
    max_decisions_per_ev: int = 25
    max_simulation_steps: int = 3000
    decision_distance_threshold: float = 100.0
    goal_arrive_distance: float = 2.0
    success_bonus: float = 3.0
    uturn_penalty: float = 1.0
    dead_end_penalty: float = 20.0
    progress_reward_enabled: bool = False
    progress_reward_alpha: float = 0.5
    progress_reward_beta: float = 0.5
    traffic_penalty_enabled: bool = False
    traffic_penalty_delta: float = 0.2
    traffic_penalty_threshold: int = 4
    revisit_penalty_enabled: bool = True
    revisit_penalty: float = 1.0
    potential_shaping_enabled: bool = False
    potential_shaping_gamma: float = 0.99
    potential_distance_scale: float = 1000.0
    simulation_step_sec: float = 0.5
    memory_learning_threshold: int = 500
    memory_push_learning_threshold: int = 500
    enable_bottleneck: bool = False
    bottleneck_start_time_sec: float = 120.0
    bottleneck_end_time_sec: float = 900.0
    bottleneck_speed_factor: float = 0.5
    bottleneck_candidate_edges: tuple[str, ...] = ()


@dataclass(frozen=True)
class BottleneckEvent:
    event_id: str
    edge_ids: tuple[str, ...]
    lane_ids: tuple[str, ...]
    start_time_sec: float
    end_time_sec: float
    speed_factor: float
    description: str


class BottleneckController:
    def __init__(self, traci, config: EpisodeConfig) -> None:
        self.traci = traci
        self.config = config
        self.base_lane_speeds: dict[str, float] = {}
        self.events: list[BottleneckEvent] = self._build_events()
        self.applied_event_ids: set[str] = set()
        self.reverted_event_ids: set[str] = set()
        self.active_event_ids: set[str] = set()
        self.active_steps: int = 0

    def _build_events(self) -> list[BottleneckEvent]:
        if not self.config.enable_bottleneck or not self.config.bottleneck_candidate_edges:
            return []
        lane_ids: list[str] = []
        edge_ids: list[str] = []
        for edge_id in self.config.bottleneck_candidate_edges:
            try:
                nlanes = int(self.traci.edge.getLaneNumber(edge_id))
            except Exception:
                continue
            edge_ids.append(edge_id)
            for idx in range(nlanes):
                lane_id = f"{edge_id}_{idx}"
                lane_ids.append(lane_id)
                try:
                    self.base_lane_speeds[lane_id] = float(self.traci.lane.getMaxSpeed(lane_id))
                except Exception:
                    self.base_lane_speeds[lane_id] = 0.0
        if not lane_ids:
            return []
        return [
            BottleneckEvent(
                event_id="fixed_bottleneck",
                edge_ids=tuple(edge_ids),
                lane_ids=tuple(lane_ids),
                start_time_sec=self.config.bottleneck_start_time_sec,
                end_time_sec=self.config.bottleneck_end_time_sec,
                speed_factor=self.config.bottleneck_speed_factor,
                description="Lane speed reduction bottleneck",
            )
        ]

    def update(self, now: float) -> None:
        if not self.events:
            return
        any_active = False
        for event in self.events:
            if event.start_time_sec <= now <= event.end_time_sec:
                self._apply_event(event)
                any_active = True
            elif event.event_id in self.applied_event_ids and event.event_id not in self.reverted_event_ids:
                self._revert_event(event)
        if any_active:
            self.active_steps += 1

    def _apply_event(self, event: BottleneckEvent) -> None:
        if event.event_id in self.applied_event_ids:
            self.active_event_ids.add(event.event_id)
            return
        for lane_id in event.lane_ids:
            base_speed = self.base_lane_speeds.get(lane_id, 0.0)
            if base_speed <= 0.0:
                continue
            try:
                self.traci.lane.setMaxSpeed(lane_id, base_speed * event.speed_factor)
            except Exception:
                continue
        self.applied_event_ids.add(event.event_id)
        self.active_event_ids.add(event.event_id)

    def _revert_event(self, event: BottleneckEvent) -> None:
        for lane_id in event.lane_ids:
            base_speed = self.base_lane_speeds.get(lane_id, 0.0)
            if base_speed <= 0.0:
                continue
            try:
                self.traci.lane.setMaxSpeed(lane_id, base_speed)
            except Exception:
                continue
        self.reverted_event_ids.add(event.event_id)
        self.active_event_ids.discard(event.event_id)

    def restore_all(self) -> None:
        for event in self.events:
            self._revert_event(event)

    def summarize(self) -> dict[str, Any]:
        return {
            "bottleneck_enabled": bool(self.events),
            "bottleneck_triggered": bool(self.applied_event_ids),
            "bottleneck_active_steps": self.active_steps,
            "bottleneck_events": [
                {
                    "event_id": event.event_id,
                    "edge_ids": list(event.edge_ids),
                    "lane_ids": list(event.lane_ids),
                    "start_time_sec": event.start_time_sec,
                    "end_time_sec": event.end_time_sec,
                    "speed_factor": event.speed_factor,
                    "description": event.description,
                }
                for event in self.events
            ],
        }


@dataclass
class AgentDecisionSnapshot:
    current_edge: str
    current_node: str
    neighbors: list[int]
    next_junctions: list[str]
    outgoing_edges: list[Any]
    state_tensor: torch.Tensor
    action_mask: torch.Tensor
    graph_data: Any
    bag: tuple[int, int, list[int], torch.Tensor]


@dataclass(frozen=True)
class EpisodeRunInput:
    ev_ids: list[str]
    goal_nodes: dict[str, str]
    evaluate: bool = False
    train: bool = True
    agent_path: list[list[str]] | None = None
    debug: bool = False
    dijkstra_guidance_probability: float = 0.0
    rollout_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class EpisodeRuntime:
    states: dict[str, EVRuntimeState]
    total_steps: int = 0
    simulation_steps: int = 0
    decisions: dict[str, int] = field(default_factory=dict)
    teleport_start_count: int = 0
    teleport_end_count: int = 0
    teleport_started_ids: set[str] = field(default_factory=set)
    teleport_ended_ids: set[str] = field(default_factory=set)
    ev_teleport_started_ids: set[str] = field(default_factory=set)
    ev_teleport_ended_ids: set[str] = field(default_factory=set)
    seen_vehicle_ids: set[str] = field(default_factory=set)
    exploration_decision_count: int = 0
    dijkstra_guidance_available_count: int = 0
    dijkstra_guided_decision_count: int = 0

    @classmethod
    def create(cls, ev_ids: list[str], n_nodes: int) -> "EpisodeRuntime":
        states = {ev_id: EVRuntimeState(n_nodes=n_nodes) for ev_id in ev_ids}
        decisions = {ev_id: 0 for ev_id in ev_ids}
        return cls(states=states, decisions=decisions)


def resolve_ev_ids(explicit_ev_ids: list[str] | None, scenario_ev_ids: list[str]) -> list[str]:
    return explicit_ev_ids if explicit_ev_ids is not None else scenario_ev_ids


def dist_to_lane_end(traci, ev_id: str) -> float:
    lane_id = traci.vehicle.getLaneID(ev_id)
    pos = traci.vehicle.getLanePosition(ev_id)
    lane_len = traci.lane.getLength(lane_id)
    return max(0.0, lane_len - pos)


@dataclass(frozen=True)
class MovementTopologyCache:
    """Static controlled links captured after each scenario's traci.load."""

    first_tls: dict[tuple[str, str], tuple[str, int]]
    outgoing_lanes: dict[tuple[str, str], tuple[str, ...]]

    @classmethod
    def capture(cls, traci) -> MovementTopologyCache:
        first_tls: dict[tuple[str, str], tuple[str, int]] = {}
        lanes: dict[tuple[str, str], list[str]] = {}
        for tls_id in traci.trafficlight.getIDList():
            for link_idx, group in enumerate(traci.trafficlight.getControlledLinks(tls_id)):
                for in_lane, out_lane, _via_lane in group:
                    movement = (lane2edge(in_lane), lane2edge(out_lane))
                    first_tls.setdefault(movement, (tls_id, link_idx))
                    lanes.setdefault(movement, []).append(out_lane)
        return cls(first_tls, {key: tuple(dict.fromkeys(value)) for key, value in lanes.items()})


def build_tls_state(
    traci, cur_edge: str, next_junctions: list[str], outgoing_edges: list[Any],
    movement_topology: MovementTopologyCache | None = None,
) -> list[list[float]]:
    edge_by_node = {edge.getToNode().getID(): edge for edge in outgoing_edges}
    ordered_edges = [edge_by_node[node_id] for node_id in next_junctions]
    tls_state: list[list[float]] = []
    for edge in ordered_edges:
        if movement_topology is None:
            tls, link_idx = find_tls_and_link_index_for_movement(
                cur_edge, edge.getID(), traci_api=traci,
            )
        else:
            tls, link_idx = movement_topology.first_tls.get((cur_edge, edge.getID()), (None, None))
        time_to_green = 0.0
        remaining_green = 0.0
        if tls:
            time_to_green, remaining_green = time_to_green_and_remaining_green(
                tls,
                link_idx,
                traci_api=traci,
            )
        tls_state.append([time_to_green / 60.0, remaining_green / 60.0])
    return tls_state


def resolve_movement_out_lanes(
    traci, cur_edge: str, out_edge_id: str,
    movement_topology: MovementTopologyCache | None = None,
) -> list[str]:
    if movement_topology is not None:
        return list(movement_topology.outgoing_lanes.get((cur_edge, out_edge_id), ()))
    lane_ids: list[str] = []
    for tls in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls)
        for link_group in links:
            for in_lane, out_lane, via_lane in link_group:
                if (
                    lane2edge(in_lane) == cur_edge
                    and lane2edge(out_lane) == out_edge_id
                ):
                    lane_ids.append(out_lane)
    return list(dict.fromkeys(lane_ids))


def candidate_lane_metrics(
    traci, cur_edge: str, edge: Any,
    movement_topology: MovementTopologyCache | None = None,
) -> dict[str, float | list[str]]:
    out_edge_id = edge.getID()
    lane_ids = resolve_movement_out_lanes(traci, cur_edge, out_edge_id, movement_topology)
    if not lane_ids:
        lane_ids = [lane.getID() for lane in edge.getLanes()]

    halting_values: list[float] = []
    vehicle_values: list[float] = []
    for lane_id in lane_ids:
        try:
            halting_values.append(float(traci.lane.getLastStepHaltingNumber(lane_id)))
        except Exception:
            halting_values.append(0.0)
        try:
            vehicle_values.append(float(traci.lane.getLastStepVehicleNumber(lane_id)))
        except Exception:
            vehicle_values.append(0.0)

    halting_total = sum(halting_values)
    vehicle_total = sum(vehicle_values)
    halting_norm = min(halting_total / 20.0, 1.0)
    vehicle_norm = min(vehicle_total / 20.0, 1.0)
    return {
        "lane_ids": lane_ids,
        "halting_total": halting_total,
        "vehicle_total": vehicle_total,
        "halting_norm": halting_norm,
        "vehicle_norm": vehicle_norm,
    }


def collect_candidate_debug_rows(traci, cur_edge: str, next_junctions: list[str], outgoing_edges: list[Any]) -> list[dict[str, Any]]:
    edge_by_node = {edge.getToNode().getID(): edge for edge in outgoing_edges}
    ordered_edges = [edge_by_node[node_id] for node_id in next_junctions]
    rows: list[dict[str, Any]] = []
    for edge in ordered_edges:
        candidate_edge = edge.getID()
        next_node = edge.getToNode().getID()
        tls_id, link_idx = find_tls_and_link_index_for_movement(
            cur_edge,
            candidate_edge,
            traci_api=traci,
        )
        time_to_green = 0.0
        remaining_green = 0.0
        if tls_id is not None and link_idx is not None:
            time_to_green, remaining_green = time_to_green_and_remaining_green(
                tls_id,
                link_idx,
                traci_api=traci,
            )
        try:
            travel_time = float(traci.edge.getTraveltime(candidate_edge))
        except Exception:
            travel_time = float("nan")
        try:
            mean_speed = float(traci.edge.getLastStepMeanSpeed(candidate_edge))
        except Exception:
            mean_speed = float("nan")
        try:
            occupancy = float(traci.edge.getLastStepOccupancy(candidate_edge))
        except Exception:
            occupancy = float("nan")
        lane_metrics = candidate_lane_metrics(traci, cur_edge, edge)
        rows.append(
            {
                "candidate_edge": candidate_edge,
                "next_node": next_node,
                "travel_time": travel_time,
                "mean_speed": mean_speed,
                "occupancy": occupancy,
                "tls_id": tls_id,
                "time_to_green": time_to_green,
                "remaining_green": remaining_green,
                "lane_ids": lane_metrics["lane_ids"],
                "halting_number": lane_metrics["halting_total"],
                "vehicle_number": lane_metrics["vehicle_total"],
            }
        )
    return rows


def build_candidate_action_features(
    *,
    traci,
    static_pack: dict[str, Any],
    graph_data: Any,
    state: EVRuntimeState,
    cur_edge: str,
    next_junctions: list[str],
    outgoing_edges: list[Any],
    neighbors: list[int],
    movement_topology: MovementTopologyCache | None = None,
) -> torch.Tensor:
    edge_by_node = {edge.getToNode().getID(): edge for edge in outgoing_edges}
    ordered_edges = [edge_by_node[node_id] for node_id in next_junctions]
    tls_rows = build_tls_state(traci, cur_edge, next_junctions, outgoing_edges, movement_topology)
    edge2idx = static_pack.get("eid2eidx") or static_pack.get("edge2idx") or {}
    previous_node = state.recent_nodes[-2] if len(state.recent_nodes) >= 2 else None
    rows: list[torch.Tensor] = []

    for idx, edge in enumerate(ordered_edges):
        edge_id = edge.getID()
        edge_idx = edge2idx.get(edge_id)
        if edge_idx is None:
            edge_attr = torch.zeros(graph_data.edge_attr.size(1), dtype=torch.float32)
        else:
            edge_attr = graph_data.edge_attr[edge_idx].detach().cpu().float()

        neighbor_idx = neighbors[idx]
        is_uturn = 1.0 if previous_node is not None and neighbor_idx == previous_node else 0.0
        visited_flag = float(state.ever_visit[neighbor_idx].item())
        visit_age = float(state.visit_age[neighbor_idx].item())
        tls_tensor = torch.tensor(tls_rows[idx], dtype=torch.float32)
        lane_metrics = candidate_lane_metrics(traci, cur_edge, edge, movement_topology)
        extra = torch.tensor(
            [
                is_uturn,
                visited_flag,
                visit_age,
                float(lane_metrics["halting_norm"]),
                float(lane_metrics["vehicle_norm"]),
            ],
            dtype=torch.float32,
        )
        rows.append(torch.cat([edge_attr, tls_tensor, extra], dim=0))

    if not rows:
        return torch.zeros((0, graph_data.edge_attr.size(1) + 7), dtype=torch.float32)
    return torch.stack(rows, dim=0)


def build_agent_snapshot(
    *,
    traci,
    net,
    static_pack: dict[str, Any],
    travel_time_ema,
    mean_speed_ema,
    occupancy_ema,
    agent,
    state: EVRuntimeState,
    ev_id: str,
    goal_node_id: str,
    cur_edge_override: str | None = None,
    debug: bool = False,
    movement_topology: MovementTopologyCache | None = None,
) -> AgentDecisionSnapshot:
    nid2idx = static_pack["nid2idx"]
    cur_edge = cur_edge_override or traci.vehicle.getRoadID(ev_id)
    current_node = net.getEdge(cur_edge).getToNode().getID()
    next_junctions, outgoing_edges = candidate_next_junctions(
        net,
        cur_edge,
        traci_api=traci,
    )
    neighbors = [nid2idx[node_id] for node_id in next_junctions]

    source = nid2idx[current_node]
    target = nid2idx[goal_node_id]
    state_tensor = torch.cat(
        (
            torch.nn.functional.one_hot(torch.tensor(source), agent.num_nodes).float(),
            torch.nn.functional.one_hot(torch.tensor(target), agent.num_nodes).float(),
        )
    )
    action_mask = action_mask_dqn(
        net,
        cur_edge,
        nid2idx,
        n=agent.num_nodes,
        traci_api=traci,
        torch_api=torch,
    )
    if hasattr(agent, "concat_state_time"):
        state_tensor = agent.concat_state_time(state_tensor, traci.simulation.getTime())
    else:
        state_tensor = agent.concat_state_tls(
            state_tensor,
            build_tls_state(traci, cur_edge, next_junctions, outgoing_edges, movement_topology),
        )

    graph_data = build_data_runtime(
        static_pack,
        travel_time_ema,
        mean_speed_ema,
        occupancy_ema,
        debug,
        edge_feature_mode=getattr(agent, "edge_feature_mode", None) or "all",
        traci_api=traci,
    )
    state.recent_nodes.append(source)
    graph_data, state.ever_visit, state.visit_age = dynamic_node_feature_update(
        graph_data,
        target,
        state.recent_nodes,
        state.ever_visit,
        state.visit_age,
    )
    candidate_action_features = build_candidate_action_features(
        traci=traci,
        static_pack=static_pack,
        graph_data=graph_data,
        state=state,
        cur_edge=cur_edge,
        next_junctions=next_junctions,
        outgoing_edges=outgoing_edges,
        neighbors=neighbors,
        movement_topology=movement_topology,
    )
    graph_data.action_features = candidate_action_features
    if hasattr(agent, "concat_state_action_features"):
        state_tensor = agent.concat_state_action_features(state_tensor, candidate_action_features)

    return AgentDecisionSnapshot(
        current_edge=cur_edge,
        current_node=current_node,
        neighbors=neighbors,
        next_junctions=next_junctions,
        outgoing_edges=outgoing_edges,
        state_tensor=state_tensor,
        action_mask=action_mask,
        graph_data=graph_data,
        bag=(source, target, neighbors, candidate_action_features),
    )


def make_terminal_transition(
    *,
    agent,
    state: EVRuntimeState,
    reward: float,
    goal_node_id: str,
    done: float,
    current_time_sec: float = 0.0,
) -> TransitionPayload:
    state_np = state.last_state.detach().cpu().numpy().astype(np.float32)
    goal_idx = state.bag[1] if state.bag is not None else 0
    if agent.use_gnn:
        next_state = torch.zeros(agent.a_size * 2)
    else:
        base_next_state = torch.cat(
            (
                torch.nn.functional.one_hot(torch.tensor(goal_idx), agent.num_nodes).float(),
                torch.nn.functional.one_hot(torch.tensor(goal_idx), agent.num_nodes).float(),
            )
        )
        if hasattr(agent, "concat_state_time"):
            next_state = agent.concat_state_time(base_next_state, current_time_sec)
        else:
            next_state = torch.cat((base_next_state, torch.zeros(agent.a_size * 2)))
        if hasattr(agent, "concat_state_action_features"):
            next_state = agent.concat_state_action_features(next_state, None)
    _, next_action_mask = agent._pad_neighbors(state.neighbors or [])
    return (
        state_np,
        np.array([state.idx], dtype=np.int64),
        next_state.detach().cpu().numpy().astype(np.float32),
        np.array([reward], dtype=np.float32),
        np.array([done], dtype=np.float32),
        next_action_mask.detach().cpu().numpy().astype(np.bool_),
        freeze_pyg_data(state.data_t) if state.data_t is not None else None,
        freeze_pyg_data(state.data_t) if state.data_t is not None else None,
        state.bag,
        state.bag,
    )


def make_step_transition(
    *,
    state: EVRuntimeState,
    next_state_tensor: torch.Tensor,
    next_neighbors: list[int],
    reward: float,
    done: float,
    next_data_t: Any,
    next_bag: Any,
) -> TransitionPayload:
    state_np = state.last_state.detach().cpu().numpy().astype(np.float32)
    _, next_action_mask = state.transition["agent"]._pad_neighbors(next_neighbors)
    return (
        state_np,
        np.array([state.idx], dtype=np.int64),
        next_state_tensor.detach().cpu().numpy().astype(np.float32),
        np.array([reward], dtype=np.float32),
        np.array([done], dtype=np.float32),
        next_action_mask.detach().cpu().numpy().astype(np.bool_),
        freeze_pyg_data(state.data_t) if state.data_t is not None else None,
        freeze_pyg_data(next_data_t) if next_data_t is not None else None,
        state.bag,
        next_bag,
    )


def push_agent_transition(
    *,
    agent,
    state: EVRuntimeState,
    ev_id: str,
    payload: TransitionPayload,
    terminated: bool,
    truncated: bool = False,
) -> None:
    """Store a transition while preserving per-EV trajectory metadata for PPO."""
    if state.trajectory_id is None:
        raise RuntimeError(f"trajectory_id is not initialized for {ev_id}")
    push_trajectory = getattr(agent.memory, "push_trajectory", None)
    if callable(push_trajectory):
        push_trajectory(
            *payload,
            trajectory_key=state.trajectory_id,
            timestep=state.trajectory_t,
            terminated=terminated,
            truncated=truncated,
        )
    else:
        agent.memory.push(*payload)
    state.trajectory_t += 1


def handle_arrival(
    *,
    agent,
    train: bool,
    evaluate: bool,
    now: float,
    ev_id: str,
    goal_node_id: str,
    state: EVRuntimeState,
    success_bonus: float,
    memory_push_learning_threshold: int,
    terminal_shaping_reward: float = 0.0,
) -> int:
    state.done = True
    if state.prev_decision_time is None:
        step_time = 0.0
    else:
        step_time = (now - state.prev_decision_time) / 60.0
    reward = -step_time + float(success_bonus) + float(terminal_shaping_reward)
    state.time_spend += step_time
    state.total_cost += reward

    if evaluate or state.last_state is None or state.idx is None:
        return 0

    payload = make_terminal_transition(
        agent=agent,
        state=state,
        reward=reward,
        goal_node_id=goal_node_id,
        done=1.0,
        current_time_sec=now,
    )
    push_agent_transition(
        agent=agent,
        state=state,
        ev_id=ev_id,
        payload=payload,
        terminated=True,
    )
    if train and len(agent.memory) >= memory_push_learning_threshold:
        return 1
    return 0


def should_make_decision(remaining_distance: float, threshold: float) -> bool:
    return remaining_distance < threshold


def learn_if_needed(
    agent,
    total_steps: int,
    train: bool,
    evaluate: bool,
    memory_learning_threshold: int,
) -> dict[str, Any]:
    loss_values: list[float] = []
    empty_stats = {
        "update_attempts": 0,
        "learn_calls": 0,
        "gradient_steps": 0,
        "update_metrics": {},
        "loss_values": loss_values,
        "last_loss": None,
        "mean_loss": None,
    }
    if not train or evaluate:
        return empty_stats
    if len(agent.memory) < memory_learning_threshold:
        return empty_stats

    update_attempts = 0
    successful_updates = 0
    gradient_steps = 0
    update_metrics: dict[str, float] = {}
    learn_updates = min(total_steps, 64)
    for _ in range(learn_updates):
        attempt_before = getattr(agent, "update_attempt_counter", None)
        learn_before = getattr(agent, "learn_step_counter", None)
        gradient_before = getattr(agent, "gradient_step_counter", None)
        try:
            loss = agent.learn_dqn_neighbor()
        except Exception as exc:
            complete_size = getattr(agent.memory, "complete_size", None)
            raise AgentLearningError(
                f"{type(agent).__name__} update failed; "
                f"memory_size={len(agent.memory)}, complete_size={complete_size}, "
                f"successful_updates={successful_updates}"
            ) from exc

        attempt_after = getattr(agent, "update_attempt_counter", None)
        learn_after = getattr(agent, "learn_step_counter", None)
        gradient_after = getattr(agent, "gradient_step_counter", None)

        if isinstance(attempt_before, int) and isinstance(attempt_after, int):
            update_attempts += max(0, attempt_after - attempt_before)
        if isinstance(learn_before, int) and isinstance(learn_after, int):
            successful_updates += max(0, learn_after - learn_before)
        elif loss is not None:
            successful_updates += 1
        if isinstance(gradient_before, int) and isinstance(gradient_after, int):
            gradient_steps += max(0, gradient_after - gradient_before)
        elif isinstance(learn_before, int) and isinstance(learn_after, int):
            gradient_steps += max(0, learn_after - learn_before)

        if loss is not None:
            loss_values.append(float(loss))
        if isinstance(learn_before, int) and isinstance(learn_after, int) and learn_after > learn_before:
            update_metrics = {
                str(key): float(value)
                for key, value in dict(getattr(agent, "last_update_metrics", {})).items()
            }

    if update_attempts == 0:
        update_attempts = successful_updates
    return {
        "update_attempts": update_attempts,
        "learn_calls": successful_updates,
        "gradient_steps": gradient_steps,
        "update_metrics": update_metrics,
        "loss_values": loss_values,
        "last_loss": loss_values[-1] if loss_values else None,
        "mean_loss": float(np.mean(loss_values)) if loss_values else None,
    }


def summarize_episode(states: dict[str, EVRuntimeState], agent_path: list[list[str]] | None, evaluate: bool) -> list[float]:
    total_costs: list[float] = []
    for state in states.values():
        if evaluate and agent_path is not None:
            agent_path.append(state.routes)
        total_costs.append(state.total_cost)
    return total_costs


def summarize_episode_time_costs(states: dict[str, EVRuntimeState]) -> list[float]:
    return [float(state.time_spend) for state in states.values()]


class DQNGNNEpisodeRunner:
    def __init__(
        self,
        *,
        traci,
        net,
        static_pack: dict[str, Any],
        agent,
        travel_time_ema,
        mean_speed_ema,
        occupancy_ema,
        config: EpisodeConfig | None = None,
        logger: EpisodeLogger | None = None,
    ) -> None:
        self.traci = traci
        self.net = net
        self.static_pack = static_pack
        self.agent = agent
        self.travel_time_ema = travel_time_ema
        self.mean_speed_ema = mean_speed_ema
        self.occupancy_ema = occupancy_ema
        self.config = config or EpisodeConfig()
        self.logger = logger
        self.nid2idx = static_pack["nid2idx"]
        self._reward_penalty_dead_end = float(self.config.dead_end_penalty)
        self.last_run_stats: dict[str, Any] = {}
        self.movement_topology: MovementTopologyCache | None = None

    def run(self, run_input: EpisodeRunInput) -> list[float]:
        # Rebuild for every loaded scenario, even when a runner is reused.
        self.movement_topology = MovementTopologyCache.capture(self.traci)
        runtime = EpisodeRuntime.create(run_input.ev_ids, n_nodes=len(self.nid2idx))
        for ev_id, state in runtime.states.items():
            state.trajectory_id = f"{run_input.rollout_id}:{ev_id}"
        bottleneck_controller = BottleneckController(self.traci, self.config)

        while True:
            if runtime.simulation_steps >= self.config.max_simulation_steps:
                break
            if not self._advance_simulation():
                break
            runtime.simulation_steps += 1

            now = self.traci.simulation.getTime()
            self._record_teleports(runtime, run_input, now)

            bottleneck_controller.update(now)
            arrived = set(self.traci.simulation.getArrivedIDList())
            runtime.total_steps += self._handle_arrivals(arrived, runtime, run_input, now)

            present = set(self.traci.vehicle.getIDList())
            runtime.seen_vehicle_ids.update(present.intersection(runtime.states))
            if self._all_finished(runtime):
                break

            self._mark_missing_seen_evs_out(present, arrived, runtime, run_input, now)
            self._step_active_evs(present, arrived, runtime, run_input, now)

        final_now = float(self.traci.simulation.getTime())
        final_reason = (
            "simulation_timeout"
            if runtime.simulation_steps >= self.config.max_simulation_steps
            else "simulation_ended"
        )
        for ev_id, state in runtime.states.items():
            if not state.done and not state.out:
                runtime.total_steps += self._terminate_failure(
                    ev_id,
                    state,
                    run_input,
                    final_now,
                    reason=final_reason,
                )

        bottleneck_controller.restore_all()
        learn_stats = learn_if_needed(
            self.agent,
            runtime.total_steps,
            run_input.train,
            run_input.evaluate,
            self.config.memory_learning_threshold,
        )
        self.last_run_stats = self._summarize_runtime(runtime, learn_stats, bottleneck_controller.summarize())
        return summarize_episode(runtime.states, run_input.agent_path, run_input.evaluate)

    def _advance_simulation(self) -> bool:
        remaining = self.traci.simulation.getMinExpectedNumber()
        if remaining <= 0:
            return False
        self.traci.simulationStep()
        self._compress_route_on_depart()
        return True

    def _compress_route_on_depart(self) -> None:
        for vehicle_id in self.traci.simulation.getDepartedIDList():
            if not vehicle_id.startswith("EV_"):
                continue
            current_edge = self.traci.vehicle.getRoadID(vehicle_id)
            if not current_edge or current_edge.startswith(":"):
                continue
            self.traci.vehicle.setRoute(vehicle_id, [current_edge])

    def _record_teleports(
        self,
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        now: float,
    ) -> None:
        try:
            started_ids = set(self.traci.simulation.getStartingTeleportIDList())
        except Exception:
            started_ids = set()
        try:
            ended_ids = set(self.traci.simulation.getEndingTeleportIDList())
        except Exception:
            ended_ids = set()

        runtime.teleport_start_count += len(started_ids)
        runtime.teleport_end_count += len(ended_ids)
        runtime.teleport_started_ids.update(started_ids)
        runtime.teleport_ended_ids.update(ended_ids)
        runtime.ev_teleport_started_ids.update({veh_id for veh_id in started_ids if veh_id.startswith("EV_")})
        runtime.ev_teleport_ended_ids.update({veh_id for veh_id in ended_ids if veh_id.startswith("EV_")})
        for ev_id in runtime.ev_teleport_started_ids:
            state = runtime.states.get(ev_id)
            if state is not None and not state.done and not state.out:
                runtime.total_steps += self._terminate_failure(
                    ev_id,
                    state,
                    run_input,
                    now,
                    reason="teleport",
                )

    def _summarize_runtime(
        self,
        runtime: EpisodeRuntime,
        learn_stats: dict[str, Any],
        bottleneck_stats: dict[str, Any],
    ) -> dict[str, Any]:
        ev_costs_minutes = summarize_episode_time_costs(runtime.states)
        success_flags = [bool(state.done) for state in runtime.states.values()]
        horizon_cost_minutes = (
            float(self.config.max_simulation_steps)
            * float(self.config.simulation_step_sec)
            / 60.0
        )
        penalized_ev_costs_minutes = [
            cost if succeeded else max(cost, horizon_cost_minutes)
            for cost, succeeded in zip(ev_costs_minutes, success_flags)
        ]
        return {
            "simulation_steps": runtime.simulation_steps,
            "transition_steps": runtime.total_steps,
            "episode_cost_minutes": float(sum(ev_costs_minutes)),
            "ev_costs_minutes": ev_costs_minutes,
            "penalized_episode_cost_minutes": float(sum(penalized_ev_costs_minutes)),
            "penalized_ev_costs_minutes": penalized_ev_costs_minutes,
            "success_count": int(sum(success_flags)),
            "failure_count": int(len(success_flags) - sum(success_flags)),
            "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
            "failure_reasons": {
                ev_id: state.failure_reason
                for ev_id, state in runtime.states.items()
                if not state.done
            },
            "exploration_decision_count": runtime.exploration_decision_count,
            "dijkstra_guidance_available_count": runtime.dijkstra_guidance_available_count,
            "dijkstra_guided_decision_count": runtime.dijkstra_guided_decision_count,
            "truncated_by_max_steps": runtime.simulation_steps >= self.config.max_simulation_steps,
            "teleport_start_count": runtime.teleport_start_count,
            "teleport_end_count": runtime.teleport_end_count,
            "teleport_started_ids": sorted(runtime.teleport_started_ids),
            "teleport_ended_ids": sorted(runtime.teleport_ended_ids),
            "ev_teleport_started_ids": sorted(runtime.ev_teleport_started_ids),
            "ev_teleport_ended_ids": sorted(runtime.ev_teleport_ended_ids),
            "update_attempts": int(learn_stats.get("update_attempts", 0)),
            "learn_calls": int(learn_stats.get("learn_calls", 0)),
            "gradient_steps": int(learn_stats.get("gradient_steps", 0)),
            "update_metrics": dict(learn_stats.get("update_metrics", {})),
            "last_loss": learn_stats.get("last_loss"),
            "mean_loss": learn_stats.get("mean_loss"),
            **bottleneck_stats,
        }

    def _handle_arrivals(
        self,
        arrived: set[str],
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        now: float,
    ) -> int:
        total_steps = 0
        for ev_id in arrived:
            state = runtime.states.get(ev_id)
            if state is None or state.done or state.out:
                continue
            goal_node_id = run_input.goal_nodes.get(ev_id)
            if goal_node_id is None:
                continue
            total_steps += handle_arrival(
                agent=self.agent,
                train=run_input.train,
                evaluate=run_input.evaluate,
                now=now,
                ev_id=ev_id,
                goal_node_id=goal_node_id,
                state=state,
                success_bonus=self.config.success_bonus,
                memory_push_learning_threshold=self.config.memory_push_learning_threshold,
                terminal_shaping_reward=self._terminal_potential_reward(state.prev_edge, goal_node_id),
            )
        return total_steps

    def _all_finished(self, runtime: EpisodeRuntime) -> bool:
        return all(state.done or state.out for state in runtime.states.values())

    def _mark_missing_seen_evs_out(
        self,
        present: set[str],
        arrived: set[str],
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        now: float,
    ) -> None:
        missing_ids = runtime.seen_vehicle_ids - present - arrived
        for ev_id in missing_ids:
            state = runtime.states.get(ev_id)
            if state is not None and not state.done and not state.out:
                runtime.total_steps += self._terminate_failure(
                    ev_id,
                    state,
                    run_input,
                    now,
                    reason="vehicle_missing",
                )

    @staticmethod
    def _is_fatal_traci_error(exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        fatal_markers = (
            "socket",
            "tcpip",
            "peer",
            "connection closed",
            "server closed",
        )
        return any(marker in text for marker in fatal_markers)

    def _terminate_failure(
        self,
        ev_id: str,
        state: EVRuntimeState,
        run_input: EpisodeRunInput,
        now: float,
        *,
        reason: str,
    ) -> int:
        if state.done or state.out:
            return 0
        step_time = 0.0 if state.prev_decision_time is None else max(0.0, now - state.prev_decision_time) / 60.0
        terminal_shaping = self._terminal_potential_reward(
            state.prev_edge,
            run_input.goal_nodes.get(ev_id, ""),
        )
        reward = -step_time - self._reward_penalty_dead_end + terminal_shaping
        state.time_spend += step_time
        state.total_cost += reward
        state.failure_reason = reason

        pushed = 0
        if not run_input.evaluate and state.last_state is not None and state.idx is not None:
            payload = make_terminal_transition(
                agent=self.agent,
                state=state,
                reward=reward,
                goal_node_id=run_input.goal_nodes.get(ev_id, ""),
                done=1.0,
                current_time_sec=now,
            )
            push_agent_transition(
                agent=self.agent,
                state=state,
                ev_id=ev_id,
                payload=payload,
                terminated=True,
            )
            if run_input.train and len(self.agent.memory) >= self.config.memory_push_learning_threshold:
                pushed = 1
        state.out = True
        return pushed

    def _mark_ev_route_failed(
        self,
        ev_id: str,
        state: EVRuntimeState,
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        now: float,
        exc: BaseException,
    ) -> None:
        runtime.total_steps += self._terminate_failure(
            ev_id,
            state,
            run_input,
            now,
            reason="route_application_failure",
        )
        print(f"[warn] dropping EV after route application failure: ev={ev_id}: {exc}")

    def _step_active_evs(
        self,
        present: set[str],
        arrived: set[str],
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        now: float,
    ) -> None:
        for ev_id, state in runtime.states.items():
            if ev_id in arrived:
                continue
            if state.done or state.out or ev_id not in present:
                continue
            if runtime.decisions[ev_id] >= self.config.max_decisions_per_ev:
                if self._ready_for_decision(ev_id):
                    runtime.total_steps += self._terminate_failure(
                        ev_id,
                        state,
                        run_input,
                        now,
                        reason="max_decisions",
                    )
                continue
            if not self._ready_for_decision(ev_id):
                continue
            self._step_single_ev(ev_id, state, runtime, run_input, now)

    def _ready_for_decision(self, ev_id: str) -> bool:
        try:
            remaining = remaining_routes_len(
                self.net,
                ev_id,
                traci_api=self.traci,
            )
        except Exception:
            return False
        return should_make_decision(remaining, self.config.decision_distance_threshold)

    def _step_single_ev(
        self,
        ev_id: str,
        state: EVRuntimeState,
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        now: float,
    ) -> None:
        goal_node_id = run_input.goal_nodes[ev_id]
        if state.current_exit_edge is None:
            self._handle_first_decision(ev_id, goal_node_id, state, runtime, run_input)
            return
        if state.current_exit_edge != state.last_committed_exit_edge:
            self._handle_followup_decision(ev_id, goal_node_id, state, runtime, run_input, now)

    def _handle_first_decision(
        self,
        ev_id: str,
        goal_node_id: str,
        state: EVRuntimeState,
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
    ) -> None:
        snapshot = build_agent_snapshot(
            traci=self.traci,
            net=self.net,
            static_pack=self.static_pack,
            travel_time_ema=self.travel_time_ema,
            mean_speed_ema=self.mean_speed_ema,
            occupancy_ema=self.occupancy_ema,
            agent=self.agent,
            state=state,
            ev_id=ev_id,
            goal_node_id=goal_node_id,
            cur_edge_override=None,
            debug=run_input.debug,
            movement_topology=self.movement_topology,
        )
        if not snapshot.neighbors:
            runtime.total_steps += self._terminate_failure(
                ev_id,
                state,
                run_input,
                self.traci.simulation.getTime(),
                reason="no_initial_neighbor",
            )
            return

        state.neighbors = snapshot.neighbors
        state.action_mask = snapshot.action_mask
        state.data_t = snapshot.graph_data
        state.bag = snapshot.bag
        state.transition["agent"] = self.agent
        state.last_state = snapshot.state_tensor
        state.last_action_nid = self._choose_action(
            snapshot,
            goal_node_id,
            run_input,
            runtime,
        )
        state.idx = snapshot.neighbors.index(state.last_action_nid)
        next_junction = snapshot.next_junctions[state.idx]
        try:
            chosen_edge = apply_next_junction(
                ev_id,
                next_junction,
                snapshot.outgoing_edges,
                traci_api=self.traci,
            )
        except RuntimeError as exc:
            if self._is_fatal_traci_error(exc):
                raise
            self._mark_ev_route_failed(
                ev_id,
                state,
                runtime,
                run_input,
                self.traci.simulation.getTime(),
                exc,
            )
            return
        state.routes.append(chosen_edge)
        state.prev_decision_time = self.traci.simulation.getTime()
        state.prev_edge = snapshot.current_edge
        state.last_committed_exit_edge = state.current_exit_edge
        state.current_exit_edge = chosen_edge
        runtime.decisions[ev_id] += 1

        if self.logger:
            self.logger.step(snapshot.current_node, self.nid2idx[snapshot.current_node], snapshot.neighbors, state.last_action_nid)
            self.logger.decision_candidates(
                ev_id=ev_id,
                current_edge=snapshot.current_edge,
                current_node=snapshot.current_node,
                chosen_edge=chosen_edge,
                rows=collect_candidate_debug_rows(self.traci, snapshot.current_edge, snapshot.next_junctions, snapshot.outgoing_edges),
            )

    def _handle_followup_decision(
        self,
        ev_id: str,
        goal_node_id: str,
        state: EVRuntimeState,
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        now: float,
    ) -> None:
        current_node = self.net.getEdge(state.current_exit_edge).getToNode().getID()
        if current_node == goal_node_id:
            if dist_to_lane_end(self.traci, ev_id) > self.config.goal_arrive_distance:
                return
            return

        snapshot = build_agent_snapshot(
            traci=self.traci,
            net=self.net,
            static_pack=self.static_pack,
            travel_time_ema=self.travel_time_ema,
            mean_speed_ema=self.mean_speed_ema,
            occupancy_ema=self.occupancy_ema,
            agent=self.agent,
            state=state,
            ev_id=ev_id,
            goal_node_id=goal_node_id,
            cur_edge_override=state.current_exit_edge,
            debug=run_input.debug,
            movement_topology=self.movement_topology,
        )
        if not snapshot.neighbors:
            self._handle_dead_end(ev_id, state, runtime, run_input, snapshot, goal_node_id, now)
            return

        reward = self._compute_followup_reward(state, snapshot.current_node, snapshot.current_edge, goal_node_id, now)

        self._push_followup_transition(
            ev_id=ev_id,
            state=state,
            runtime=runtime,
            run_input=run_input,
            snapshot=snapshot,
            reward=reward,
            done=0.0,
        )

        state.neighbors = snapshot.neighbors
        state.action_mask = snapshot.action_mask
        state.bag = snapshot.bag
        state.data_t = snapshot.graph_data
        state.last_state = snapshot.state_tensor
        state.transition["agent"] = self.agent
        state.last_action_nid = self._choose_action(
            snapshot,
            goal_node_id,
            run_input,
            runtime,
        )
        state.idx = snapshot.neighbors.index(state.last_action_nid)
        next_junction = snapshot.next_junctions[state.idx]
        try:
            chosen_edge = apply_next_junction(
                ev_id,
                next_junction,
                snapshot.outgoing_edges,
                traci_api=self.traci,
            )
        except RuntimeError as exc:
            if self._is_fatal_traci_error(exc):
                raise
            self._mark_ev_route_failed(ev_id, state, runtime, run_input, now, exc)
            return
        state.routes.append(chosen_edge)
        state.data_t = snapshot.graph_data
        state.last_state = snapshot.state_tensor
        state.last_committed_exit_edge = state.current_exit_edge
        state.current_exit_edge = chosen_edge
        state.prev_edge = snapshot.current_edge
        state.prev_decision_time = now
        runtime.decisions[ev_id] += 1

        if self.logger:
            self.logger.step(snapshot.current_node, self.nid2idx[snapshot.current_node], snapshot.neighbors, state.last_action_nid)
            self.logger.decision_candidates(
                ev_id=ev_id,
                current_edge=snapshot.current_edge,
                current_node=snapshot.current_node,
                chosen_edge=chosen_edge,
                rows=collect_candidate_debug_rows(self.traci, snapshot.current_edge, snapshot.next_junctions, snapshot.outgoing_edges),
            )

    def _choose_action(
        self,
        snapshot: AgentDecisionSnapshot,
        goal_node_id: str,
        run_input: EpisodeRunInput,
        runtime: EpisodeRuntime,
    ) -> int:
        guidance_probability = min(1.0, max(0.0, float(run_input.dijkstra_guidance_probability)))
        if guidance_probability > 0.0 and not run_input.evaluate:
            guided_action = self._dijkstra_guided_action(
                snapshot.current_edge,
                goal_node_id,
                snapshot.neighbors,
            )
            epsilon = min(1.0, max(0.0, float(getattr(self.agent, "EPSILON", 0.0))))
            if random.random() <= epsilon:
                runtime.exploration_decision_count += 1
                if guided_action is not None:
                    runtime.dijkstra_guidance_available_count += 1
                    if random.random() <= guidance_probability:
                        runtime.dijkstra_guided_decision_count += 1
                        return guided_action
                return random.choice(snapshot.neighbors)
            evaluate = True
        else:
            evaluate = run_input.evaluate

        return self.agent.choose_action_gnn(
            snapshot.state_tensor,
            snapshot.action_mask,
            snapshot.neighbors,
            snapshot.bag[0],
            snapshot.bag[1],
            snapshot.graph_data,
            evaluate=evaluate,
        )

    def _dijkstra_guided_action(
        self,
        current_edge: str,
        goal_node_id: str,
        neighbors: list[int],
    ) -> int | None:
        try:
            next_node, _ = dijkstra_next_hop(
                self.net,
                self.static_pack["adj"],
                current_edge,
                goal_node_id,
                lambda edge_id: float(self.static_pack["edge_len"].get(edge_id, 1.0)),
            )
        except Exception:
            return None
        guided_action = self.nid2idx.get(next_node)
        return guided_action if guided_action in neighbors else None

    def _compute_followup_reward(
        self,
        state: EVRuntimeState,
        current_node: str,
        current_edge: str,
        goal_node_id: str,
        now: float,
    ) -> float:
        reward = 0.0
        current_node_idx = self.nid2idx[current_node]
        # build_agent_snapshot() appends the current node before this reward is
        # calculated. A -> B -> A therefore ends in [..., A, B, A], so the
        # node two moves back is at -3 (not -2).
        is_uturn = len(state.recent_nodes) >= 3 and current_node_idx == state.recent_nodes[-3]
        if is_uturn:
            reward -= self.config.uturn_penalty
        elif self.config.revisit_penalty_enabled and current_node_idx in list(state.recent_nodes)[:-1]:
            reward -= self.config.revisit_penalty
        step_time = 0.0 if state.prev_decision_time is None else (now - state.prev_decision_time) / 60.0
        state.time_spend += step_time
        reward -= step_time
        reward += self._progress_reward(state.prev_edge, current_edge, goal_node_id)
        reward += self._potential_shaping_reward(state.prev_edge, current_edge, goal_node_id)
        reward += self._traffic_density_penalty(current_edge)
        state.total_cost += reward
        return reward

    def _progress_reward(self, previous_edge: str | None, current_edge: str | None, goal_node_id: str) -> float:
        if not self.config.progress_reward_enabled or not previous_edge or not current_edge:
            return 0.0
        previous_distance = self._static_shortest_distance(previous_edge, goal_node_id)
        current_distance = self._static_shortest_distance(current_edge, goal_node_id)
        if previous_distance is None or current_distance is None:
            return 0.0
        if current_distance <= previous_distance:
            return float(self.config.progress_reward_alpha)
        return -float(self.config.progress_reward_beta)

    def _potential_shaping_reward(
        self,
        previous_edge: str | None,
        current_edge: str | None,
        goal_node_id: str,
    ) -> float:
        if not self.config.potential_shaping_enabled or not previous_edge or not current_edge:
            return 0.0
        previous_distance = self._static_shortest_distance(previous_edge, goal_node_id)
        current_distance = self._static_shortest_distance(current_edge, goal_node_id)
        if previous_distance is None or current_distance is None:
            return 0.0
        scale = max(float(self.config.potential_distance_scale), 1e-8)
        previous_phi = -previous_distance / scale
        current_phi = -current_distance / scale
        return float(self.config.potential_shaping_gamma) * current_phi - previous_phi

    def _terminal_potential_reward(
        self,
        previous_edge: str | None,
        goal_node_id: str,
    ) -> float:
        if not self.config.potential_shaping_enabled or not previous_edge or not goal_node_id:
            return 0.0
        previous_distance = self._static_shortest_distance(previous_edge, goal_node_id)
        if previous_distance is None:
            return 0.0
        scale = max(float(self.config.potential_distance_scale), 1e-8)
        # Terminal states have Phi=0, so F = gamma*0 - Phi(s).
        return previous_distance / scale

    def _static_shortest_distance(self, current_edge: str, goal_node_id: str) -> float | None:
        try:
            _, _, total = dijkstra_path(
                self.net,
                self.static_pack["adj"],
                current_edge,
                goal_node_id,
                lambda edge_id: float(self.static_pack["edge_len"].get(edge_id, 1.0)),
            )
        except Exception:
            return None
        if total is None:
            return None
        return float(total)

    def _traffic_density_penalty(self, current_edge: str | None) -> float:
        if not self.config.traffic_penalty_enabled or not current_edge:
            return 0.0
        try:
            vehicle_count = int(self.traci.edge.getLastStepVehicleNumber(current_edge))
        except Exception:
            return 0.0
        if vehicle_count < int(self.config.traffic_penalty_threshold):
            return 0.0
        return -float(self.config.traffic_penalty_delta) * float(vehicle_count)

    def _push_followup_transition(
        self,
        *,
        ev_id: str,
        state: EVRuntimeState,
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        snapshot: AgentDecisionSnapshot,
        reward: float,
        done: float,
    ) -> None:
        if run_input.evaluate or state.last_state is None or state.idx is None:
            return
        payload = make_step_transition(
            state=state,
            next_state_tensor=snapshot.state_tensor,
            next_neighbors=snapshot.neighbors,
            reward=reward,
            done=done,
            next_data_t=snapshot.graph_data,
            next_bag=snapshot.bag,
        )
        push_agent_transition(
            agent=self.agent,
            state=state,
            ev_id=ev_id,
            payload=payload,
            terminated=bool(done),
        )
        if run_input.train and len(self.agent.memory) >= self.config.memory_push_learning_threshold:
            runtime.total_steps += 1

    def _handle_dead_end(
        self,
        ev_id: str,
        state: EVRuntimeState,
        runtime: EpisodeRuntime,
        run_input: EpisodeRunInput,
        snapshot: AgentDecisionSnapshot,
        goal_node_id: str,
        now: float,
    ) -> None:
        runtime.total_steps += self._terminate_failure(
            ev_id,
            state,
            run_input,
            now,
            reason="dead_end",
        )


__all__ = [
    "AgentLearningError",
    "TransitionPayload",
    "EpisodeConfig",
    "EpisodeRunInput",
    "DQNGNNEpisodeRunner",
    "resolve_ev_ids",
]
