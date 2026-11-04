"""Pure helpers for classifying and traversing SUMO network topology."""

from __future__ import annotations

from collections import deque


MIN_DECISION_LEN = 2.0
MAX_SKIP_HOPS = 8
VCLASS_FILTER = None


def is_external_node(node):
    return node.getType() != "internal" and not node.getID().startswith(":")


def is_external_edge(edge):
    function = edge.getFunction()
    return function != "internal" and not edge.getID().startswith(":")


def is_decisionable_edge(edge):
    try:
        return not edge.isInternal() and edge.getLength() >= MIN_DECISION_LEN
    except Exception:
        return False


def edge_allows(edge, vehicle_class):
    if vehicle_class is None:
        return True
    try:
        lane = edge.getLanes()[0]
        return lane.allows(vehicle_class)
    except Exception:
        return True


def expand_to_decision_edges(
    net,
    start_edge,
    max_hops=MAX_SKIP_HOPS,
    vclass=VCLASS_FILTER,
):
    """Traverse short/internal edges to the first decisionable edge layer."""
    start_id = start_edge.getID()
    results = []
    queue = deque([(start_edge, [])])
    visited = {start_id}
    hops = 0

    while queue and hops <= max_hops:
        hops += 1
        for _ in range(len(queue)):
            current_edge, prefix = queue.popleft()

            if is_decisionable_edge(current_edge) and len(prefix) > 0:
                target = current_edge.getToNode().getID()
                results.append((target, current_edge.getID(), prefix.copy()))
                continue

            next_node = current_edge.getToNode()
            for next_edge in next_node.getOutgoing():
                if vclass is not None and not edge_allows(next_edge, vclass):
                    continue
                next_edge_id = next_edge.getID()
                if next_edge_id in visited:
                    continue
                visited.add(next_edge_id)
                queue.append((next_edge, prefix + [current_edge.getID()]))

    if not results:
        target = start_edge.getToNode().getID()
        results.append((target, start_id, []))

    unique = {}
    for target, first_real, prefix in results:
        unique[first_real] = (target, first_real, prefix)
    return list(unique.values())


def build_macro_candidates(net, cur_edge_id, outs_edges):
    """Build de-duplicated decision macros for outgoing edges."""
    macros = []
    for outgoing_edge in outs_edges:
        if is_decisionable_edge(outgoing_edge):
            macros.append(
                {
                    "node": outgoing_edge.getToNode().getID(),
                    "first_edge": outgoing_edge.getID(),
                    "prefix": [],
                }
            )
        else:
            expansions = expand_to_decision_edges(net, outgoing_edge)
            for target_node, first_real, prefix in expansions:
                macros.append(
                    {
                        "node": target_node,
                        "first_edge": first_real,
                        "prefix": prefix,
                    }
                )

    seen = set()
    unique_macros = []
    for macro in macros:
        key = macro["first_edge"]
        if key in seen:
            continue
        seen.add(key)
        unique_macros.append(macro)
    return unique_macros


def _is_internal_edge(edge) -> bool:
    try:
        return edge.getFunction() == "internal"
    except Exception:
        return str(edge.getID()).startswith(":")


def is_all_internal_to(net, cur_edge_id: str, target_node_id: str, max_hops: int = 30):
    """Return whether an internal-only path reaches the target and its length."""
    source_node = net.getEdge(cur_edge_id).getToNode()
    if source_node.getID() == target_node_id:
        return True, 0.0

    queue = deque()
    visited = set()

    for edge in source_node.getOutgoing():
        if _is_internal_edge(edge):
            queue.append((edge, edge.getLength()))
            visited.add(edge.getID())

    hops = 0
    while queue and hops <= max_hops:
        hops += 1
        for _ in range(len(queue)):
            edge, accumulated_length = queue.popleft()
            next_node = edge.getToNode()
            if next_node.getID() == target_node_id:
                return True, accumulated_length

            for next_edge in next_node.getOutgoing():
                if not _is_internal_edge(next_edge):
                    continue
                if next_edge.getID() in visited:
                    continue
                visited.add(next_edge.getID())
                queue.append(
                    (next_edge, accumulated_length + next_edge.getLength())
                )

    return False, 0.0


def is_internal_edge(edge):
    try:
        return edge.getFunction() == "internal"
    except Exception:
        return edge.getID().startswith(":")


__all__ = [
    "MIN_DECISION_LEN",
    "MAX_SKIP_HOPS",
    "VCLASS_FILTER",
    "is_external_node",
    "is_external_edge",
    "is_decisionable_edge",
    "edge_allows",
    "expand_to_decision_edges",
    "build_macro_candidates",
    "is_all_internal_to",
    "is_internal_edge",
]
