"""Deprecated procedural Dijkstra simulation retained for compatibility."""

from __future__ import annotations

import warnings


def dijkstra_sim(
    net,
    static_pack,
    ev_ids,
    filesource,
    rd1=None,
    rd2=None,
    rd3=None,
    rd4=None,
    rd5=None,
    test=False,
    set_goal=None,
    testing_paths=None,
    traffic_jam=False,
    using_env=None,
    ev_route=None,
    *,
    traci_api,
    torch_api,
    json_api,
    count_evs,
    remaining_distance,
    pathfinder,
    cost_fn,
    apply_route,
    max_simulation_steps=6000,
):
    """Run the deprecated procedural Dijkstra compatibility experiment."""
    warnings.warn(
        "legacy.dijkstra_sim is deprecated; use DijkstraBaselineRunner",
        DeprecationWarning,
        stacklevel=2,
    )
    ignored_arguments = {
        "ev_ids": ev_ids,
        "rd1": rd1,
        "rd2": rd2,
        "test": test,
        "set_goal": set_goal,
    }
    supplied_ignored = {
        name: value
        for name, value in ignored_arguments.items()
        if value not in (None, False, [])
    }
    if supplied_ignored:
        warnings.warn(
            "legacy.dijkstra_sim compatibility arguments have no effect: "
            f"{sorted(supplied_ignored)}",
            DeprecationWarning,
            stacklevel=2,
        )
    if max_simulation_steps <= 0:
        raise ValueError("max_simulation_steps must be positive")

    class EVState:
        def __init__(self):
            self.done = False
            self.out = False
            self.routes = []
            self.time_spend = 0.0

    if traffic_jam:
        incident_edges = ["5707608#2", "5707608#0"]
        speed_factor = 0.3
        start_time_threshold, _end_time_threshold = 120, 600
        lane_ids = []
        base_lane_speed = {}
        for edge_id in incident_edges:
            lane_count = traci_api.edge.getLaneNumber(edge_id)
            for lane_index in range(lane_count):
                lane_id = f"{edge_id}_{lane_index}"
                lane_ids.append(lane_id)
                base_lane_speed[lane_id] = traci_api.lane.getMaxSpeed(lane_id)
        applied = False

    config_path = str(filesource / "osm.sumocfg")
    adjacency = static_pack["adj"]
    decision_distance_threshold = 100.0
    goal_arrival_distance = 1.0

    def distance_to_lane_end(vehicle_id):
        lane_id = traci_api.vehicle.getLaneID(vehicle_id)
        position = traci_api.vehicle.getLanePosition(vehicle_id)
        lane_length = traci_api.lane.getLength(lane_id)
        return max(0.0, lane_length - position)

    general_vehicle_flows = [
        "free_0900",
        "norm_1500",
        "busy_2100",
        "heavy_2400",
    ]
    general_route = str(
        filesource
        / f"testing_data/output_{general_vehicle_flows[rd5]}_{rd3}.rou.xml"
    )
    ev_routes = str(
        f"./{using_env}/testing_data/"
        f"ev_{general_vehicle_flows[rd5]}_p0{rd4}_{rd3}.rou.xml"
    )
    if ev_route:
        ev_routes = ev_route

    vehicles = []
    vehicle_count = count_evs(ev_routes)
    for index in range(vehicle_count):
        vehicles.append(f"EV_{index}")

    goal_path = (
        f"./{using_env}/testing_data_goal/"
        f"goal_ev_{general_vehicle_flows[rd5]}_p0{rd4}_{rd3}.json"
    )
    with open(goal_path, "r", encoding="utf-8") as source:
        test_goal_nodes = json_api.load(source)

    if len(test_goal_nodes) != len(vehicles):
        raise ValueError(
            "legacy dijkstra goal/EV count mismatch: "
            f"goals={len(test_goal_nodes)}, evs={len(vehicles)}, "
            f"goal_file={goal_path}, ev_route={ev_routes}"
        )

    goal_nodes = {}
    for index, vehicle_id in enumerate(vehicles):
        goal_nodes[vehicle_id] = test_goal_nodes[index]

    load_arguments = [
        "-c",
        config_path,
        "--step-length",
        "0.5",
        "--route-files",
        f"{general_route}, {ev_routes}",
        "--no-warnings",
    ]
    traci_api.load(load_arguments)

    states = {
        vehicle_id: EVState()
        for vehicle_id in vehicles
    }
    route_start_times = {vehicle_id: None for vehicle_id in vehicles}
    decided = {vehicle_id: False for vehicle_id in vehicles}
    if testing_paths is None:
        testing_paths = []

    def is_ev(vehicle_id: str) -> bool:
        return vehicle_id.startswith("EV_")

    def compress_route_on_depart():
        for vehicle_id in traci_api.simulation.getDepartedIDList():
            if not is_ev(vehicle_id):
                continue
            current_edge = traci_api.vehicle.getRoadID(vehicle_id)
            if not current_edge or current_edge.startswith(":"):
                continue
            traci_api.vehicle.setRoute(vehicle_id, [current_edge])

    def run_simulation_loop():
        nonlocal applied
        for _step in range(max_simulation_steps):
            traci_api.simulationStep()
            compress_route_on_depart()
            remaining_vehicles = traci_api.simulation.getMinExpectedNumber()
            if remaining_vehicles <= 0:
                return
            now = traci_api.simulation.getTime()

            if traffic_jam and not applied and now >= start_time_threshold:
                for lane_id in lane_ids:
                    traci_api.lane.setMaxSpeed(
                        lane_id,
                        base_lane_speed[lane_id] * speed_factor,
                    )
                applied = True

            present = set(traci_api.vehicle.getIDList())
            if all(state.done for state in states.values()):
                return
            for vehicle_id, state in states.items():
                if state.done or vehicle_id not in present or decided[vehicle_id]:
                    continue

                remaining = remaining_distance(net, vehicle_id)
                if remaining >= decision_distance_threshold:
                    continue
                current_edge = traci_api.vehicle.getRoadID(vehicle_id)

                target_node = net.getEdge(current_edge).getToNode().getID()
                if target_node == goal_nodes[vehicle_id]:
                    if distance_to_lane_end(vehicle_id) > goal_arrival_distance:
                        continue
                    state.done = True
                    start_time = route_start_times[vehicle_id]
                    state.time_spend = (
                        0.0 if start_time is None else (now - start_time) / 60.0
                    )
                    continue

                _path_nodes, path_edges, _total = pathfinder(
                    net=net,
                    adj=adjacency,
                    cur_edge=current_edge,
                    goal_node=goal_nodes[vehicle_id],
                    cost_fn=cost_fn,
                )
                if path_edges is None:
                    state.out = True
                    continue
                state.routes = path_edges
                route_start_times[vehicle_id] = now
                apply_route(
                    vehicle_id,
                    current_edge,
                    path_edges,
                    suffix_keep=None,
                )
                decided[vehicle_id] = True

            arrived = set(traci_api.simulation.getArrivedIDList())
            for vehicle_id in arrived:
                if vehicle_id not in states:
                    continue
                state = states[vehicle_id]
                if state.done or state.out:
                    continue
                state.done = True
                start_time = route_start_times[vehicle_id]
                if start_time is not None:
                    state.time_spend += (now - start_time) / 60.0
            if all(state.done or state.out for state in states.values()):
                return

        unfinished = [
            vehicle_id
            for vehicle_id, state in states.items()
            if not state.done and not state.out
        ]
        raise RuntimeError(
            "legacy dijkstra simulation exceeded max_simulation_steps: "
            f"limit={max_simulation_steps}, unfinished={unfinished}"
        )

    try:
        run_simulation_loop()
    finally:
        if traffic_jam:
            for lane_id in lane_ids:
                traci_api.lane.setMaxSpeed(lane_id, base_lane_speed[lane_id])

    total_time_spend = 0
    for vehicle_id, state in states.items():
        total_time_spend += state.time_spend
        testing_paths.append(state.routes)

    return total_time_spend


__all__ = ["dijkstra_sim"]
