"""Compatibility shim; new code should import from :mod:`rp.graph`."""

from rp.graph.snapshots import freeze_pyg_data

__all__ = ["freeze_pyg_data"]
