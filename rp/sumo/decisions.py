"""Decision-time helpers that depend on a live TraCI connection."""

from __future__ import annotations

from typing import Any

from .routing import candidate_next_junctions


def reached_goal_junction(
    ev: str,
    goal_junc: str,
    net: Any,
    eps: float = 1.0,
    position_tolerance: float = 1e-3,
    *,
    traci_api: Any,
) -> bool:
    """Return whether an EV is at its goal with bounded position tolerance."""
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if position_tolerance < 0:
        raise ValueError("position_tolerance must be non-negative")
    cur_edge_id = traci_api.vehicle.getRoadID(ev)
    if not cur_edge_id or cur_edge_id.startswith("!"):
        return False

    edge = net.getEdge(cur_edge_id)
    if edge.getFunction() == "internal":
        return edge.getFromNode().getID() == goal_junc

    if edge.getToNode().getID() != goal_junc:
        return False

    lane_id = traci_api.vehicle.getLaneID(ev)
    lane_length = traci_api.lane.getLength(lane_id)
    lane_position = traci_api.vehicle.getLanePosition(ev)
    raw_remain = lane_length - lane_position
    if raw_remain < -position_tolerance:
        return False
    remain = max(raw_remain, 0.0)
    return remain <= eps


def action_mask_dqn(
    net: Any,
    cur_edge: str,
    nid2idx: dict[str, int],
    n: int | None = None,
    *,
    traci_api: Any,
    torch_api: Any,
):
    """Build a topology-sized CPU action mask for reachable junctions."""
    junctions, _ = candidate_next_junctions(
        net,
        cur_edge,
        traci_api=traci_api,
    )
    if n is None:
        n = len(nid2idx)
    if n < 0:
        raise ValueError(f"action mask size must be non-negative: n={n}")
    missing = [node_id for node_id in junctions if node_id not in nid2idx]
    if missing:
        raise ValueError(
            "Reachable junctions are missing from nid2idx: "
            f"cur_edge={cur_edge}, missing={missing}, "
            f"reachable={junctions}, node_count={len(nid2idx)}"
        )
    out_mask = [nid2idx[node_id] for node_id in junctions]
    invalid = [index for index in out_mask if index < 0 or index >= n]
    if invalid:
        raise ValueError(
            "Reachable junction indices fall outside the action mask: "
            f"cur_edge={cur_edge}, invalid_indices={invalid}, n={n}"
        )
    action_mask = torch_api.zeros(n, device="cpu")
    for index in out_mask:
        action_mask[index] = 1.0
    return action_mask


__all__ = ["reached_goal_junction", "action_mask_dqn"]
