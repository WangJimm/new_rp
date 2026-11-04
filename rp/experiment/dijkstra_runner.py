from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from .config import RuntimeConfig
from rp.routing import dijkstra_path
from .scenario_loader import (
    ScenarioSpec,
    build_testing_scenario,
    build_training_scenario,
    build_validation_scenario,
)
from rp.sumo import apply_full_route, build_load_args, remaining_routes_len


@dataclass(frozen=True)
class TrafficJamConfig:
    enabled: bool = False
    incident_edges: tuple[str, ...] = ("5707608#2", "5707608#0")
    speed_factor: float = 0.3
    start_time: float = 120.0
    end_time: float = 900.0


@dataclass(frozen=True)
class DijkstraRunConfig:
    decision_distance_threshold: float = 100.0
    goal_arrive_distance: float = 1.0
    step_length: float = 0.5
    max_simulation_steps: int = 6000
    traffic_jam: TrafficJamConfig = field(default_factory=TrafficJamConfig)


@dataclass
class DijkstraEVState:
    done: bool = False
    out: bool = False
    decided: bool = False
    start_time: float | None = None
    time_spend: float = 0.0
    routes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DijkstraSimulationResult:
    total_time_spend: float
    ev_times: dict[str, float]
    ev_routes: dict[str, list[str]]
    scenario: ScenarioSpec
    teleport_start_count: int
    teleport_end_count: int
    teleport_started_ids: list[str]
    teleport_ended_ids: list[str]
    ev_teleport_started_ids: list[str]
    ev_teleport_ended_ids: list[str]


class TrafficJamController:
    def __init__(self, traci, config: TrafficJamConfig):
        self.traci = traci
        self.config = config
        self.applied = False
        self.lane_ids: list[str] = []
        self.base_lane_speed: dict[str, float] = {}

        if not config.enabled:
            return

        for edge_id in config.incident_edges:
            lane_count = self.traci.edge.getLaneNumber(edge_id)
            for lane_idx in range(lane_count):
                lane_id = f"{edge_id}_{lane_idx}"
                self.lane_ids.append(lane_id)
                self.base_lane_speed[lane_id] = self.traci.lane.getMaxSpeed(lane_id)

    def maybe_apply(self, now: float) -> None:
        if not self.config.enabled:
            return
        if self.applied and now > self.config.end_time:
            self.restore()
            self.applied = False
            return
        if self.applied or now < self.config.start_time or now > self.config.end_time:
            return
        for lane_id in self.lane_ids:
            self.traci.lane.setMaxSpeed(lane_id, self.base_lane_speed[lane_id] * self.config.speed_factor)
        self.applied = True

    def restore(self) -> None:
        if not self.config.enabled:
            return
        for lane_id in self.lane_ids:
            self.traci.lane.setMaxSpeed(lane_id, self.base_lane_speed[lane_id])


def dijkstra_run_config_from_runtime(runtime_config: RuntimeConfig, traffic_jam: bool = False) -> DijkstraRunConfig:
    enabled = bool(traffic_jam or runtime_config.enable_bottleneck)
    incident_edges = tuple(runtime_config.bottleneck_candidate_edges) or TrafficJamConfig().incident_edges
    return DijkstraRunConfig(
        decision_distance_threshold=runtime_config.decision_distance_threshold,
        goal_arrive_distance=runtime_config.goal_arrive_distance,
        step_length=runtime_config.step_length,
        max_simulation_steps=runtime_config.max_simulation_steps,
        traffic_jam=TrafficJamConfig(
            enabled=enabled,
            incident_edges=incident_edges,
            speed_factor=runtime_config.bottleneck_speed_factor,
            start_time=runtime_config.bottleneck_start_time_sec,
            end_time=runtime_config.bottleneck_end_time_sec,
        ),
    )


class DijkstraBaselineRunner:
    def __init__(
        self,
        *,
        traci,
        net,
        static_pack: dict[str, Any],
        runtime_config: RuntimeConfig,
        run_config: DijkstraRunConfig | None = None,
        cost_fn: Callable[[str], float] | None = None,
    ) -> None:
        self.traci = traci
        self.net = net
        self.static_pack = static_pack
        self.runtime_config = runtime_config
        self.run_config = run_config or DijkstraRunConfig()
        self.cost_fn = cost_fn or self._traci_travel_time
        self.adj = static_pack["adj"]

    def _traci_travel_time(self, edge_id: str) -> float:
        return self.traci.edge.getTraveltime(edge_id)

    def run(self, scenario: ScenarioSpec, testing_paths: list[list[str]] | None = None) -> DijkstraSimulationResult:
        self.traci.load(self._build_load_args(scenario))
        states = {ev_id: DijkstraEVState() for ev_id in scenario.ev_ids}
        traffic_jam = TrafficJamController(self.traci, self.run_config.traffic_jam)
        teleport_started_ids: set[str] = set()
        teleport_ended_ids: set[str] = set()
        teleport_start_count = 0
        teleport_end_count = 0

        try:
            simulation_steps = 0
            while True:
                if simulation_steps >= self.run_config.max_simulation_steps:
                    unfinished = [
                        ev_id
                        for ev_id, state in states.items()
                        if not state.done and not state.out
                    ]
                    raise RuntimeError(
                        "Dijkstra simulation exceeded max_simulation_steps: "
                        f"limit={self.run_config.max_simulation_steps}, "
                        f"unfinished={unfinished}"
                    )
                self.traci.simulationStep()
                simulation_steps += 1
                self._compress_route_on_depart()
                step_started_ids, step_ended_ids = self._record_teleports()
                teleport_started_ids.update(step_started_ids)
                teleport_ended_ids.update(step_ended_ids)
                teleport_start_count += len(step_started_ids)
                teleport_end_count += len(step_ended_ids)

                if self.traci.simulation.getMinExpectedNumber() <= 0:
                    break

                now = self.traci.simulation.getTime()
                traffic_jam.maybe_apply(now)
                present = set(self.traci.vehicle.getIDList())

                if all(state.done for state in states.values()):
                    break

                for ev_id, state in states.items():
                    if state.done or state.out or state.decided or ev_id not in present:
                        continue
                    self._maybe_assign_dijkstra_route(ev_id, state, scenario.goal_nodes[ev_id], now)

                self._handle_arrivals(states, now)
        finally:
            traffic_jam.restore()

        ev_times = {ev_id: state.time_spend for ev_id, state in states.items()}
        ev_routes = {ev_id: state.routes for ev_id, state in states.items()}
        if testing_paths is not None:
            testing_paths.extend(ev_routes.values())

        return DijkstraSimulationResult(
            total_time_spend=sum(ev_times.values()),
            ev_times=ev_times,
            ev_routes=ev_routes,
            scenario=scenario,
            teleport_start_count=teleport_start_count,
            teleport_end_count=teleport_end_count,
            teleport_started_ids=sorted(teleport_started_ids),
            teleport_ended_ids=sorted(teleport_ended_ids),
            ev_teleport_started_ids=sorted(veh_id for veh_id in teleport_started_ids if veh_id.startswith("EV_")),
            ev_teleport_ended_ids=sorted(veh_id for veh_id in teleport_ended_ids if veh_id.startswith("EV_")),
        )

    def _build_load_args(self, scenario: ScenarioSpec) -> list[str]:
        return build_load_args(self.runtime_config, scenario)

    def _compress_route_on_depart(self) -> None:
        for vehicle_id in self.traci.simulation.getDepartedIDList():
            if not vehicle_id.startswith("EV_"):
                continue
            current_edge = self.traci.vehicle.getRoadID(vehicle_id)
            if not current_edge or current_edge.startswith(":"):
                continue
            self.traci.vehicle.setRoute(vehicle_id, [current_edge])

    def _record_teleports(self) -> tuple[set[str], set[str]]:
        try:
            started_ids = set(self.traci.simulation.getStartingTeleportIDList())
        except Exception:
            started_ids = set()
        try:
            ended_ids = set(self.traci.simulation.getEndingTeleportIDList())
        except Exception:
            ended_ids = set()
        return started_ids, ended_ids

    def _maybe_assign_dijkstra_route(
        self,
        ev_id: str,
        state: DijkstraEVState,
        goal_node_id: str,
        now: float,
    ) -> None:
        remaining = remaining_routes_len(
            self.net,
            ev_id,
            traci_api=self.traci,
        )
        if remaining >= self.run_config.decision_distance_threshold:
            return

        current_edge = self.traci.vehicle.getRoadID(ev_id)
        current_node = self.net.getEdge(current_edge).getToNode().getID()
        if current_node == goal_node_id:
            if self._dist_to_lane_end(ev_id) > self.run_config.goal_arrive_distance:
                return
            state.done = True
            state.time_spend = 0.0 if state.start_time is None else (now - state.start_time) / 60.0
            return

        path_nodes, path_edges, _ = dijkstra_path(
            net=self.net,
            adj=self.adj,
            cur_edge=current_edge,
            goal_node=goal_node_id,
            cost_fn=self.cost_fn,
        )
        if path_edges is None:
            state.out = True
            return

        state.routes = path_edges
        state.start_time = now
        apply_full_route(
            ev_id,
            current_edge,
            path_edges,
            suffix_keep=None,
            traci_api=self.traci,
        )
        state.decided = True

    def _handle_arrivals(self, states: dict[str, DijkstraEVState], now: float) -> None:
        for ev_id in self.traci.simulation.getArrivedIDList():
            state = states.get(ev_id)
            if state is None or state.done or state.out:
                continue
            state.done = True
            if state.start_time is None:
                state.time_spend = 0.0
            else:
                state.time_spend += (now - state.start_time) / 60.0

    def _dist_to_lane_end(self, ev_id: str) -> float:
        lane_id = self.traci.vehicle.getLaneID(ev_id)
        position = self.traci.vehicle.getLanePosition(ev_id)
        lane_length = self.traci.lane.getLength(lane_id)
        return max(0.0, lane_length - position)


def run_dijkstra_baseline_for_testing_key(
    *,
    traci,
    net,
    static_pack: dict[str, Any],
    runtime_config: RuntimeConfig,
    scenario_index: int,
    probability_index: int,
    seed: int,
    testing_paths: list[list[str]] | None = None,
    traffic_jam: bool = False,
) -> DijkstraSimulationResult:
    scenario = build_testing_scenario(
        runtime_config,
        scenario_index=scenario_index,
        probability_index=probability_index,
        seed=seed,
    )
    runner = DijkstraBaselineRunner(
        traci=traci,
        net=net,
        static_pack=static_pack,
        runtime_config=runtime_config,
        run_config=dijkstra_run_config_from_runtime(runtime_config, traffic_jam=traffic_jam),
    )
    return runner.run(scenario, testing_paths=testing_paths)


def run_dijkstra_baseline_for_training_key(
    *,
    traci,
    net,
    static_pack: dict[str, Any],
    runtime_config: RuntimeConfig,
    scenario_index: int,
    probability_index: int,
    seed: int,
    training_paths: list[list[str]] | None = None,
    traffic_jam: bool = False,
) -> DijkstraSimulationResult:
    scenario = build_training_scenario(
        runtime_config,
        scenario_index=scenario_index,
        probability_index=probability_index,
        seed=seed,
    )
    runner = DijkstraBaselineRunner(
        traci=traci,
        net=net,
        static_pack=static_pack,
        runtime_config=runtime_config,
        run_config=dijkstra_run_config_from_runtime(runtime_config, traffic_jam=traffic_jam),
    )
    return runner.run(scenario, testing_paths=training_paths)


def run_dijkstra_baseline_for_validation_key(
    *,
    traci,
    net,
    static_pack: dict[str, Any],
    runtime_config: RuntimeConfig,
    scenario_index: int,
    probability_index: int,
    seed: int,
    validation_paths: list[list[str]] | None = None,
    traffic_jam: bool = False,
) -> DijkstraSimulationResult:
    scenario = build_validation_scenario(
        runtime_config,
        scenario_index=scenario_index,
        probability_index=probability_index,
        seed=seed,
    )
    runner = DijkstraBaselineRunner(
        traci=traci,
        net=net,
        static_pack=static_pack,
        runtime_config=runtime_config,
        run_config=dijkstra_run_config_from_runtime(runtime_config, traffic_jam=traffic_jam),
    )
    return runner.run(scenario, testing_paths=validation_paths)


__all__ = [
    "TrafficJamConfig",
    "DijkstraRunConfig",
    "DijkstraEVState",
    "DijkstraSimulationResult",
    "TrafficJamController",
    "dijkstra_run_config_from_runtime",
    "DijkstraBaselineRunner",
    "run_dijkstra_baseline_for_testing_key",
    "run_dijkstra_baseline_for_training_key",
    "run_dijkstra_baseline_for_validation_key",
]
