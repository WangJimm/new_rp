"""Compatibility shim; new code should import from :mod:`rp.sumo`."""

from rp.sumo.vehicle_spawning import spawn_evs, spawn_evs_testing, spawn_for_rl

__all__ = ["spawn_for_rl", "spawn_evs_testing", "spawn_evs"]
