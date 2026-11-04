"""Stable public API for SUMO and TraCI integration."""

from .decisions import action_mask_dqn, reached_goal_junction
from .distance import (
    distance_to_next_external_on,
    remaining_routes_len,
    remaining_to_exit_external_end,
)
from .route_files import (
    batch_keep_first_edge,
    change_route,
    count_evs_in_rou,
    keep_first_edge_in_rou_file,
    save_routes_xml,
)
from .routing import (
    apply_full_route,
    apply_next_junction,
    candidate_next_junctions,
    reachable_edges_from,
)
from .runtime import build_load_args
from .traffic_signals import (
    GREEN_CHARS,
    build_inedge_to_tls_and_indices,
    compute_signal_backbone_for_edges,
    find_tls_and_link_index_for_movement,
    green_ratio_for_link_indices,
    lane2edge,
    time_to_green_and_remaining_green,
)
from .vehicle_spawning import spawn_evs, spawn_evs_testing, spawn_for_rl

__all__ = [
    "action_mask_dqn",
    "reached_goal_junction",
    "distance_to_next_external_on",
    "remaining_routes_len",
    "remaining_to_exit_external_end",
    "batch_keep_first_edge",
    "change_route",
    "count_evs_in_rou",
    "keep_first_edge_in_rou_file",
    "save_routes_xml",
    "apply_full_route",
    "apply_next_junction",
    "candidate_next_junctions",
    "reachable_edges_from",
    "build_load_args",
    "GREEN_CHARS",
    "build_inedge_to_tls_and_indices",
    "compute_signal_backbone_for_edges",
    "find_tls_and_link_index_for_movement",
    "green_ratio_for_link_indices",
    "lane2edge",
    "time_to_green_and_remaining_green",
    "spawn_evs",
    "spawn_evs_testing",
    "spawn_for_rl",
]
