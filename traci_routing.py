"""Compatibility shim; new code should import from :mod:`rp.sumo`."""

from rp.sumo.routing import (
    apply_full_route,
    apply_next_junction,
    candidate_next_junctions,
    reachable_edges_from,
)

__all__ = [
    "reachable_edges_from",
    "candidate_next_junctions",
    "apply_next_junction",
    "apply_full_route",
]
