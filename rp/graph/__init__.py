"""Stable, lazily loaded public API for graph construction and snapshots."""

from __future__ import annotations

from importlib import import_module


_FEATURE_EXPORTS = {
    "DECAY",
    "FLOOR",
    "VALID_EDGE_FEATURE_MODES",
    "dynamic_node_feature_update",
    "get_edge_index_v2",
    "prepare_static",
    "collect_dynamic_edge_traveltime",
    "build_data_runtime",
    "build_dynamic_features",
}
_SNAPSHOT_EXPORTS = {"freeze_pyg_data"}

__all__ = sorted(_FEATURE_EXPORTS | _SNAPSHOT_EXPORTS)


def __getattr__(name: str):
    if name in _FEATURE_EXPORTS:
        return getattr(import_module(".features", __name__), name)
    if name in _SNAPSHOT_EXPORTS:
        return getattr(import_module(".snapshots", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
