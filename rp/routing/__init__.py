"""Stable public routing API."""

from .algorithms import cost_by_length, dijkstra_next_hop, dijkstra_path
from .geometry import (
    distance_from_exit_lane_to_target_edge,
    internal_chain_between_edges,
    internal_chains_for_route,
    trace_to_target_edge,
)
from .topology import (
    MAX_SKIP_HOPS,
    MIN_DECISION_LEN,
    VCLASS_FILTER,
    build_macro_candidates,
    edge_allows,
    expand_to_decision_edges,
    is_all_internal_to,
    is_decisionable_edge,
    is_external_edge,
    is_external_node,
    is_internal_edge,
)

__all__ = [
    "cost_by_length",
    "dijkstra_next_hop",
    "dijkstra_path",
    "distance_from_exit_lane_to_target_edge",
    "internal_chain_between_edges",
    "internal_chains_for_route",
    "trace_to_target_edge",
    "MAX_SKIP_HOPS",
    "MIN_DECISION_LEN",
    "VCLASS_FILTER",
    "build_macro_candidates",
    "edge_allows",
    "expand_to_decision_edges",
    "is_all_internal_to",
    "is_decisionable_edge",
    "is_external_edge",
    "is_external_node",
    "is_internal_edge",
]
