"""Compatibility shim; new code should import from :mod:`rp.routing`."""

from rp.routing.geometry import (
    _get_lane_from_id_or_edge,
    _safe_get_lane,
    distance_from_exit_lane_to_target_edge,
    internal_chain_between_edges,
    internal_chains_for_route,
    trace_to_target_edge,
)

__all__ = [
    "_get_lane_from_id_or_edge",
    "trace_to_target_edge",
    "_safe_get_lane",
    "distance_from_exit_lane_to_target_edge",
    "internal_chain_between_edges",
    "internal_chains_for_route",
]
