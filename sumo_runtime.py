"""Compatibility shim; new code should import from :mod:`rp.sumo`."""

from rp.sumo.runtime import build_load_args

__all__ = ["build_load_args"]
