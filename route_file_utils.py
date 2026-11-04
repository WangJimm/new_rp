"""Compatibility shim; new code should import from :mod:`rp.sumo`."""

from rp.sumo.route_files import (
    _split_edges,
    batch_keep_first_edge,
    change_route,
    count_evs_in_rou,
    keep_first_edge_in_rou_file,
    save_routes_xml,
)

__all__ = [
    "count_evs_in_rou",
    "change_route",
    "keep_first_edge_in_rou_file",
    "batch_keep_first_edge",
    "save_routes_xml",
]
