"""Pure shortest-path algorithms used by the routing experiments."""

from __future__ import annotations

import heapq


def cost_by_length(edge_lengths, edge_id):
    return edge_lengths[edge_id]


def dijkstra_next_hop(net, adj, cur_edge, goal_node, cost_fn):
    """Return the next junction and first edge on the cheapest path."""
    src = net.getEdge(cur_edge).getToNode().getID()
    if src == goal_node:
        return goal_node, None

    dist = {src: 0.0}
    prev_node = {}
    prev_edge = {}
    pq = [(0.0, src)]
    while pq:
        distance, node = heapq.heappop(pq)

        if distance != dist[node]:
            continue
        if node == goal_node:
            break
        for next_node, edge_id in adj.get(node, []):
            weight = cost_fn(edge_id)
            next_distance = distance + weight

            if next_node not in dist or next_distance < dist[next_node]:
                dist[next_node] = next_distance
                prev_node[next_node] = node
                prev_edge[next_node] = edge_id
                heapq.heappush(pq, (next_distance, next_node))

    if goal_node not in prev_node:
        return None, None

    next_node = goal_node
    while prev_node[next_node] != src:
        next_node = prev_node[next_node]
    first_edge = prev_edge[next_node]
    return next_node, first_edge


def dijkstra_path(net, adj, cur_edge, goal_node, cost_fn):
    """
    Return the cheapest nodes, edges, and total cost from ``cur_edge``'s exit.

    Unreachable goals return ``(None, None, None)``. If the current edge exits
    directly at the goal, the result is ``([goal], [], 0.0)``.
    """
    src = net.getEdge(cur_edge).getToNode().getID()
    if src == goal_node:
        return [src], [], 0.0

    dist = {src: 0.0}
    prev_node = {}
    prev_edge = {}
    pq = [(0.0, src)]

    while pq:
        distance, node = heapq.heappop(pq)
        if distance != dist[node]:
            continue
        if node == goal_node:
            break

        for next_node, edge_id in adj.get(node, []):
            weight = cost_fn(edge_id)
            next_distance = distance + weight
            if next_node not in dist or next_distance < dist[next_node]:
                dist[next_node] = next_distance
                prev_node[next_node] = node
                prev_edge[next_node] = edge_id
                heapq.heappush(pq, (next_distance, next_node))

    if goal_node not in prev_node:
        return None, None, None

    path_nodes = [goal_node]
    path_edges = []
    node = goal_node
    while node != src:
        previous = prev_node[node]
        edge_id = prev_edge[node]
        path_nodes.append(previous)
        path_edges.append(edge_id)
        node = previous

    path_nodes.reverse()
    path_edges.reverse()
    total_cost = dist[goal_node]
    return path_nodes, path_edges, total_cost


__all__ = ["cost_by_length", "dijkstra_next_hop", "dijkstra_path"]
