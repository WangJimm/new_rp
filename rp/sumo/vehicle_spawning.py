"""TraCI vehicle creation helpers with injected runtime dependencies."""

from __future__ import annotations

from collections.abc import Mapping


def spawn_for_rl(
    vehicle_id: str,
    from_edge: str,
    vtype: str = "ev",
    depart_lane: str = "best",
    depart_pos: str = "base",
    depart_speed: str = "max",
    *,
    traci_api,
):
    """Insert one vehicle on a temporary one-edge route."""
    route_id = f"tmp_{vehicle_id}"
    traci_api.route.add(route_id, [from_edge])
    traci_api.vehicle.add(
        vehicle_id,
        route_id,
        typeID=vtype,
        departLane=depart_lane,
        departPos=depart_pos,
        departSpeed=depart_speed,
    )


def spawn_evs_testing(
    base_depart=0,
    depart_gap=5.0,
    net=None,
    adj=None,
    src_edges=None,
    goal_nodes=None,
    *,
    traci_api,
):
    """Insert EV1-style testing vehicles from keyed source/goal mappings."""
    if not isinstance(src_edges, Mapping) or not isinstance(goal_nodes, Mapping):
        raise TypeError("src_edges and goal_nodes must be EV-keyed mappings")
    source_ids = list(src_edges)
    source_keys = set(source_ids)
    goal_keys = set(goal_nodes)
    if source_keys != goal_keys:
        missing_goals = sorted(source_keys - goal_keys)
        unexpected_goals = sorted(goal_keys - source_keys)
        raise ValueError(
            "src_edges and goal_nodes must contain identical vehicle IDs: "
            f"missing_goals={missing_goals}, unexpected_goals={unexpected_goals}"
        )

    for index, vehicle_id in enumerate(source_ids):
        source_edge = src_edges[vehicle_id]
        goal_node = goal_nodes[vehicle_id]
        print(f"{vehicle_id}, s_edge {source_edge}, cs {goal_node}")
        route_id = f"route_{vehicle_id}"
        traci_api.route.add(route_id, [source_edge])
        traci_api.vehicle.add(
            vehID=vehicle_id,
            routeID=route_id,
            depart=str(base_depart + index * depart_gap),
            departLane="best",
            departSpeed="max",
        )
        traci_api.vehicle.setSpeedMode(vehicle_id, 32)


def spawn_evs(
    start_edges,
    goal_candidates,
    vehicle_count,
    base_depart=0,
    depart_gap=5.0,
    net=None,
    adj=None,
    filename="None",
    using_env=" ",
    *,
    traci_api,
    random_api,
    time_api,
    pathfinder,
    cost_fn,
    save_routes,
    max_source_attempts=100,
):
    """Plan bounded source/goal pairs, then insert reproducibly named EVs."""
    if vehicle_count < 0:
        raise ValueError("vehicle_count must be non-negative")
    if vehicle_count and not start_edges:
        raise ValueError("start_edges must not be empty when spawning vehicles")
    if vehicle_count and not goal_candidates:
        raise ValueError("goal_candidates must not be empty when spawning vehicles")
    if max_source_attempts <= 0:
        raise ValueError("max_source_attempts must be positive")

    goal_nodes = {}
    ev_routes = []
    selected_goals = random_api.choices(goal_candidates, k=vehicle_count)
    waypoints = {}
    planned_vehicles = []

    for index in range(vehicle_count):
        vehicle_id = f"EV_{index + 1}"
        goal_node = selected_goals[index]
        attempted_sources = []
        for _attempt in range(max_source_attempts):
            source_edge = random_api.choice(start_edges)
            attempted_sources.append(source_edge)
            result = pathfinder(
                net=net,
                adj=adj,
                cur_edge=source_edge,
                goal_node=goal_node,
                cost_fn=cost_fn,
            )
            if not result[0] or len(result[1]) <= 2:
                continue

            path_nodes = result[0]
            waypoint_index = 4
            waypoint_step = 4
            selected_waypoints = []
            while waypoint_index <= len(path_nodes):
                selected_waypoints.append(path_nodes[waypoint_index - 1])
                waypoint_index += waypoint_step
            if (
                len(selected_waypoints) == 0
                or path_nodes[-1] != selected_waypoints[-1]
            ):
                selected_waypoints.append(path_nodes[-1])
            waypoints[vehicle_id] = list(selected_waypoints)
            break
        else:
            raise RuntimeError(
                "spawn_evs could not find a route longer than two edges: "
                f"vehicle_id={vehicle_id}, goal_node={goal_node}, "
                f"attempts={max_source_attempts}, "
                f"sampled_sources={attempted_sources}"
            )

        ev_routes.append([source_edge])
        goal_nodes[vehicle_id] = goal_node
        planned_vehicles.append((vehicle_id, source_edge, index))

    for vehicle_id, source_edge, index in planned_vehicles:
        route_id = f"route_{vehicle_id}"
        traci_api.route.add(route_id, [source_edge])
        traci_api.vehicle.add(
            vehID=vehicle_id,
            routeID=route_id,
            depart=str(base_depart + index * depart_gap),
            departLane="best",
            departSpeed="max",
        )
        traci_api.vehicle.setSpeedMode(vehicle_id, 32)

    save_routes(ev_routes, filename, using_env=using_env)
    return goal_nodes, waypoints


__all__ = ["spawn_for_rl", "spawn_evs_testing", "spawn_evs"]
