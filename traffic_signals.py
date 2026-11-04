"""Compatibility shim; new code should import from :mod:`rp.sumo`."""

from rp.sumo.traffic_signals import (
    GREEN_CHARS,
    _pick_active_logic,
    build_inedge_to_tls_and_indices,
    compute_signal_backbone_for_edges,
    find_tls_and_link_index_for_movement,
    green_ratio_for_link_indices,
    lane2edge,
    time_to_green_and_remaining_green,
)

__all__ = [
    "GREEN_CHARS",
    "_pick_active_logic",
    "green_ratio_for_link_indices",
    "lane2edge",
    "build_inedge_to_tls_and_indices",
    "compute_signal_backbone_for_edges",
    "find_tls_and_link_index_for_movement",
    "time_to_green_and_remaining_green",
]
