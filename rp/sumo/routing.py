"""TraCI-facing helpers for querying and applying vehicle routes."""

from __future__ import annotations

from rp.routing import is_external_edge
from .traffic_signals import lane2edge


def reachable_edges_from(cur_edge, *, traci_api):
    """Return edge IDs reachable from every lane of ``cur_edge``."""
    lane_count = traci_api.edge.getLaneNumber(cur_edge)
    current_lanes = [f"{cur_edge}_{index}" for index in range(lane_count)]
    reachable = set()
    for lane_id in current_lanes:
        for link in traci_api.lane.getLinks(lane_id, extended=True):
            to_edge = lane2edge(link[0])
            reachable.add(to_edge)
    return reachable


def candidate_next_junctions(net, cur_edge, *, traci_api):
    """Return sorted reachable junction IDs and their outgoing edges."""
    to_node = net.getEdge(cur_edge).getToNode()
    outgoing_edges = [
        edge for edge in to_node.getOutgoing() if is_external_edge(edge)
    ]
    reachable = reachable_edges_from(cur_edge, traci_api=traci_api)
    outgoing_edges = [
        edge for edge in outgoing_edges if edge.getID() in reachable
    ]
    junctions = sorted({edge.getToNode().getID() for edge in outgoing_edges})
    return junctions, outgoing_edges


def apply_next_junction(ev_id, chosen_jn, outs, *, traci_api):
    """Choose the fastest matching edge and append it to the route suffix."""
    best_edge, best_travel_time = None, 1e9
    for edge in outs:
        if edge.getToNode().getID() != chosen_jn:
            continue
        edge_id = edge.getID()
        travel_time = traci_api.edge.getTraveltime(edge_id)
        if travel_time < best_travel_time:
            best_travel_time, best_edge = travel_time, edge_id

    current_edge = traci_api.vehicle.getRoadID(ev_id)
    route = traci_api.vehicle.getRoute(ev_id)
    route_index = traci_api.vehicle.getRouteIndex(ev_id)
    if best_edge is None:
        candidate_edges = [
            (edge.getID(), edge.getToNode().getID()) for edge in outs
        ]
        raise RuntimeError(
            "apply_next_junction could not resolve outgoing edge: "
            f"ev_id={ev_id}, chosen_jn={chosen_jn}, cur_edge={current_edge}, "
            f"route={route}, route_index={route_index}, candidates={candidate_edges}"
        )
    if route_index < 0 or route_index >= len(route):
        raise RuntimeError(
            "apply_next_junction got invalid route index: "
            f"ev_id={ev_id}, cur_edge={current_edge}, route={route}, "
            f"route_index={route_index}, chosen_jn={chosen_jn}, "
            f"best_edge={best_edge}"
        )

    start_index = route_index
    suffix = list(route[start_index:])
    if not suffix:
        raise RuntimeError(
            "apply_next_junction got empty route suffix: "
            f"ev_id={ev_id}, cur_edge={current_edge}, route={route}, "
            f"route_index={route_index}, start_idx={start_index}, "
            f"chosen_jn={chosen_jn}, best_edge={best_edge}"
        )
    suffix.append(best_edge)
    try:
        traci_api.vehicle.setRoute(ev_id, suffix)
    except Exception as exc:
        raise RuntimeError(
            "apply_next_junction failed to set SUMO route: "
            f"ev_id={ev_id}, cur_edge={current_edge}, original_route={route}, "
            f"route_index={route_index}, start_idx={start_index}, "
            f"chosen_jn={chosen_jn}, best_edge={best_edge}, "
            f"new_route={suffix}"
        ) from exc
    return best_edge


def apply_full_route(
    ev_id,
    cur_edge,
    path_edges,
    suffix_keep=None,
    *,
    traci_api,
):
    """Apply the current edge followed by all or part of a planned path."""
    route_ids = [cur_edge] + (
        path_edges if suffix_keep is None else path_edges[:suffix_keep]
    )
    traci_api.vehicle.setRoute(ev_id, route_ids)


__all__ = [
    "reachable_edges_from",
    "candidate_next_junctions",
    "apply_next_junction",
    "apply_full_route",
]
