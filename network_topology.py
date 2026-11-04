"""Compatibility shim; new code should import from :mod:`rp.routing`."""

from rp.routing.topology import (
    MAX_SKIP_HOPS,
    MIN_DECISION_LEN,
    VCLASS_FILTER,
    _is_internal_edge,
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
