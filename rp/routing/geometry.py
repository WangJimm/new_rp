"""SUMO network geometry helpers that do not depend on TraCI state."""

from __future__ import annotations

from .topology import is_internal_edge


def _get_lane_from_id_or_edge(net, maybe_lane_or_edge_id):
    """Resolve a lane ID, or fall back to the first lane of an edge ID."""
    if not maybe_lane_or_edge_id:
        return None
    value = str(maybe_lane_or_edge_id)
    try:
        return net.getLane(value)
    except Exception:
        try:
            edge = net.getEdge(value)
            lanes = edge.getLanes()
            return lanes[0] if lanes else None
        except Exception:
            return None


def trace_to_target_edge(net, first_link, target_edge_id):
    """Trace a lane link through internal edges to a target external edge."""
    internal_edges = []
    current_link = first_link
    guard = 0
    while True:
        guard += 1
        if guard > 32:
            return None

        exit_lane = current_link.getToLane()
        via_lane_id = current_link.getViaLaneID()

        if not via_lane_id:
            exit_edge = exit_lane.getEdge()
            if is_internal_edge(exit_edge):
                return None
            if exit_edge.getID() == target_edge_id:
                return internal_edges, exit_lane
            return None

        via_lane = _get_lane_from_id_or_edge(net, via_lane_id)
        if via_lane is None:
            return None
        internal_edge = via_lane.getEdge()
        if not is_internal_edge(internal_edge):
            return None
        internal_edges.append(internal_edge.getID())

        via_links = via_lane.getOutgoing()
        if not via_links:
            return None
        current_link = via_links[0]


def _safe_get_lanes(net, lane_or_edge_id) -> list:
    """Resolve a lane or edge ID into a consistently shaped lane list."""
    if lane_or_edge_id is None:
        return []

    value = str(lane_or_edge_id)
    try:
        lane = net.getLane(value)
    except Exception:
        lane = None
    if lane is not None:
        return [lane]

    try:
        edge = net.getEdge(value)
    except Exception:
        return []
    return list(edge.getLanes() or [])


# Compatibility alias for code that imported the former private helper.
_safe_get_lane = _safe_get_lanes


def distance_from_exit_lane_to_target_edge(
    net,
    start_exit_lane_id,
    target_edge_id,
):
    """Measure an internal chain plus target lane, excluding the start lane."""
    start_lanes = _safe_get_lanes(net, start_exit_lane_id)
    if not start_lanes:
        return None

    best = None
    for source_lane in start_lanes:
        links = source_lane.getOutgoing() or []

        for link in links:
            traced = trace_to_target_edge(net, link, target_edge_id)
            if traced is None:
                continue
            internal_edges, exit_lane = traced
            internal_length = sum(
                net.getEdge(edge_id).getLength() for edge_id in internal_edges
            )
            exit_length = exit_lane.getLength()
            total = internal_length + exit_length
            if best is None or total < best[0]:
                best = (
                    total,
                    internal_length,
                    exit_length,
                    internal_edges,
                    exit_lane.getID(),
                )
    return best


def internal_chain_between_edges(net, from_edge_id: str, to_edge_id: str):
    """Return the shortest internal chain between two external edges."""
    from_edge = net.getEdge(from_edge_id)
    best = None
    for lane in from_edge.getLanes():
        for link in lane.getOutgoing() or []:
            traced = trace_to_target_edge(net, link, to_edge_id)
            if traced is None:
                continue
            internal_edges, exit_lane = traced
            internal_length = sum(
                net.getEdge(edge_id).getLength() for edge_id in internal_edges
            )
            if best is None or internal_length < best[0]:
                best = (internal_length, internal_edges, exit_lane.getID())
    if best is None:
        return None
    internal_length, internal_edges, exit_lane_id = best
    return {
        "from_edge": from_edge_id,
        "to_edge": to_edge_id,
        "internal_edges": internal_edges,
        "internal_len": internal_length,
        "exit_lane_id": exit_lane_id,
    }


def internal_chains_for_route(net, route_edges):
    """Resolve every consecutive external-edge pair into an internal chain."""
    hops = []
    for index in range(len(route_edges) - 1):
        from_edge = route_edges[index]
        to_edge = route_edges[index + 1]
        info = internal_chain_between_edges(net, from_edge, to_edge)
        if info is None:
            return None
        hops.append(info)
    return hops


__all__ = [
    "trace_to_target_edge",
    "distance_from_exit_lane_to_target_edge",
    "internal_chain_between_edges",
    "internal_chains_for_route",
]
