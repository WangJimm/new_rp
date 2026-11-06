"""Shortest paths respecting SUMO edge connections and vehicle permissions."""
from __future__ import annotations

import heapq


def cost_by_length(edge_lengths, edge_id):
    return edge_lengths[edge_id]


def dijkstra_next_hop(net, adj, cur_edge, goal_node, cost_fn, *, vclass="passenger"):
    """Return the first legal edge and its exit junction on the cheapest path."""
    nodes, edges, _ = dijkstra_path(
        net, adj, cur_edge, goal_node, cost_fn, vclass=vclass,
    )
    if edges is None:
        return None, None
    if not edges:
        return goal_node, None
    return nodes[1], edges[0]


def dijkstra_path(net, adj, cur_edge, goal_node, cost_fn, *, vclass="passenger"):
    """Return legal nodes, edges and cost after ``cur_edge`` (which is excluded).

    Search states are incoming edges: two arrivals at the same junction can
    permit different turns. Every transition, including the first one, must
    have a SUMO connection allowing ``vclass``. ``adj`` still limits the graph
    to the caller's edges; ``cost_fn`` retains its travel-time/length semantics.
    Use ``vclass=None`` to ignore permissions while still checking connections.
    An unreachable goal returns ``(None, None, None)``; a current edge already
    exiting at the goal returns ``([goal], [], 0.0)``.
    """
    start = net.getEdge(cur_edge)
    src = start.getToNode().getID()
    if vclass not in (None, "ignoring") and not start.allows(vclass):
        return None, None, None
    if src == goal_node:
        return [src], [], 0.0

    dist = {cur_edge: 0.0}
    previous = {}
    weights = {}
    queue = [(0.0, cur_edge)]
    while queue:
        distance, edge_id = heapq.heappop(queue)
        if distance != dist[edge_id]:
            continue
        edge = net.getEdge(edge_id)
        node = edge.getToNode().getID()
        if node == goal_node:
            path = []
            cursor = edge_id
            while cursor != cur_edge:
                path.append(cursor)
                cursor = previous[cursor]
            path.reverse()
            nodes = [src] + [net.getEdge(e).getToNode().getID() for e in path]
            return nodes, path, distance

        allowed = {e.getID() for e in edge.getAllowedOutgoing(vclass)}
        for _, next_edge in adj.get(node, []):
            if next_edge not in allowed:
                continue
            if next_edge not in weights:
                weights[next_edge] = cost_fn(next_edge)
            next_distance = distance + weights[next_edge]
            if next_edge not in dist or next_distance < dist[next_edge]:
                dist[next_edge] = next_distance
                previous[next_edge] = edge_id
                heapq.heappush(queue, (next_distance, next_edge))
    return None, None, None


__all__ = ["cost_by_length", "dijkstra_next_hop", "dijkstra_path"]
