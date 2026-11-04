"""Compatibility shim; new code should import from :mod:`rp.sumo`."""

from rp.sumo.distance import (
    distance_to_next_external_on,
    remaining_routes_len,
    remaining_to_exit_external_end,
)

__all__ = [
    "distance_to_next_external_on",
    "remaining_routes_len",
    "remaining_to_exit_external_end",
]
