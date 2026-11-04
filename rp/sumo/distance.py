"""Vehicle-position distance calculations with explicit TraCI injection."""

from __future__ import annotations

from rp.routing import internal_chains_for_route, trace_to_target_edge


def distance_to_next_external_on(net, vehicle_id, target_edge_id, *, traci_api):
    """Measure current-lane remainder, internal chain, and target lane."""
    current_lane_id = traci_api.vehicle.getLaneID(vehicle_id)
    lane = net.getLane(current_lane_id)
    lane_length = lane.getLength()
    lane_position = traci_api.vehicle.getLanePosition(vehicle_id)
    current_remainder = max(lane_length - lane_position, 0.0)

    links = lane.getOutgoing() or []
    best = None
    for link in links:
        traced = trace_to_target_edge(net, link, target_edge_id)
        if traced is None:
            continue
        internal_edges, exit_lane = traced
        internal_length = sum(
            net.getEdge(edge_id).getLength() for edge_id in internal_edges
        )
        exit_length = exit_lane.getLength()
        total = current_remainder + internal_length + exit_length
        if best is None or total < best[0]:
            best = (
                total,
                internal_length,
                exit_length,
                internal_edges,
                exit_lane.getID(),
            )
    return best


def remaining_routes_len(net, vehicle_id, *, traci_api):
    """Return the legacy remaining distance along the vehicle route."""
    current_edge = traci_api.vehicle.getRoadID(vehicle_id)
    route = traci_api.vehicle.getRoute(vehicle_id)
    route_index = traci_api.vehicle.getRouteIndex(vehicle_id)
    suffix = list(route[route_index:])

    hops = internal_chains_for_route(net, suffix)
    if len(suffix) == 1:
        if suffix[-1] != current_edge:
            raise RuntimeError(
                "remaining_routes_len route mismatch: "
                f"veh={vehicle_id}, cur_edge={current_edge}, route={route}, "
                f"route_index={route_index}, suffix={suffix}"
            )
        lane_id = traci_api.vehicle.getLaneID(vehicle_id)
        lane_length = traci_api.lane.getLength(lane_id)
        lane_position = traci_api.vehicle.getLanePosition(vehicle_id)
        return max(lane_length - lane_position, 0.0)

    if not current_edge.startswith(":"):
        remaining = (
            net.getEdge(current_edge).getLength()
            - traci_api.vehicle.getLanePosition(vehicle_id)
        )
        for hop in hops:
            internal_length = hop["internal_len"]
            external_length = net.getEdge(hop["to_edge"]).getLength()
            remaining += internal_length + external_length
        return remaining

    if current_edge.startswith(":"):
        current_hop = hops[0]
        internal_edges = current_hop["internal_edges"]
        exit_lane_id = current_hop["exit_lane_id"]
        internal_index = internal_edges.index(current_edge)
        lane_id = traci_api.vehicle.getLaneID(vehicle_id)
        lane_position = traci_api.vehicle.getLanePosition(vehicle_id)

        current_remainder = net.getLane(lane_id).getLength() - lane_position
        rest_internal = sum(
            net.getEdge(edge_id).getLength()
            for edge_id in internal_edges[internal_index + 1 :]
        )
        exit_remainder = net.getLane(exit_lane_id).getLength()
        remaining = current_remainder + rest_internal + exit_remainder

        for hop in hops[1:]:
            internal_length = hop["internal_len"]
            external_length = net.getEdge(hop["to_edge"]).getLength()
            remaining += internal_length + external_length
        return remaining

    raise RuntimeError(
        "remaining_routes_len reached unreachable branch: "
        f"veh={vehicle_id}, cur_edge={current_edge}, route={route}, "
        f"route_index={route_index}, suffix={suffix}"
    )


def remaining_to_exit_external_end(
    net,
    vehicle_id,
    internal_edges,
    exit_lane_id,
    *,
    traci_api,
):
    """Measure from the current position to the end of an exit external lane."""
    exit_lane = net.getLane(exit_lane_id)
    exit_edge_id = exit_lane.getEdge().getID()

    road = traci_api.vehicle.getRoadID(vehicle_id)
    lane_id = traci_api.vehicle.getLaneID(vehicle_id)
    lane_position = traci_api.vehicle.getLanePosition(vehicle_id)

    if not road.startswith(":"):
        if road != exit_edge_id:
            gateway_remainder = net.getLane(lane_id).getLength() - lane_position
            internal_remainder = sum(
                net.getEdge(edge_id).getLength() for edge_id in internal_edges
            )
            exit_remainder = net.getLane(exit_lane_id).getLength()
            return (
                max(gateway_remainder, 0.0)
                + internal_remainder
                + exit_remainder
            )

        return net.getLane(lane_id).getLength() - lane_position

    try:
        internal_index = internal_edges.index(road)
    except ValueError:
        print("[Error] wrong internal")
        print(f"goal edge {exit_lane_id}")
        current_edge = traci_api.vehicle.getRoadID(vehicle_id)
        route = traci_api.vehicle.getRoute(vehicle_id)
        route_index = traci_api.vehicle.getRouteIndex(vehicle_id)
        print(f"cur_edge {current_edge}")
        print(f"route {route}")
        print(f"idx {route_index}")
        suffix = list(route[route_index:])
        print(f"suffix route {suffix}")
        hops = internal_chains_for_route(net, suffix)
        print(f"hops {hops}")
        raise RuntimeError(
            "remaining_to_exit_external_end wrong internal edge: "
            f"veh={vehicle_id}, goal_edge={exit_lane_id}, road={road}, "
            f"cur_edge={current_edge}, route={route}, route_index={route_index}, "
            f"suffix={suffix}, hops={hops}, internal_edges={internal_edges}"
        )

    current_remainder = net.getLane(lane_id).getLength() - lane_position
    rest_internal = sum(
        net.getEdge(edge_id).getLength()
        for edge_id in internal_edges[internal_index + 1 :]
    )
    exit_remainder = net.getLane(exit_lane_id).getLength()
    return max(current_remainder, 0.0) + rest_internal + exit_remainder


__all__ = [
    "distance_to_next_external_on",
    "remaining_routes_len",
    "remaining_to_exit_external_end",
]
