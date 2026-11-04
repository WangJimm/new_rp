"""Compatibility shim; new code should import from :mod:`rp.graph`."""

from rp.graph.features import (
    DECAY,
    FLOOR,
    VALID_EDGE_FEATURE_MODES,
    build_data_runtime,
    build_dynamic_features,
    collect_dynamic_edge_traveltime,
    dynamic_node_feature_update,
    get_edge_index_v2,
    prepare_static,
    tls_node_features,
)

__all__ = [
    "DECAY",
    "FLOOR",
    "VALID_EDGE_FEATURE_MODES",
    "dynamic_node_feature_update",
    "get_edge_index_v2",
    "prepare_static",
    "collect_dynamic_edge_traveltime",
    "build_data_runtime",
    "build_dynamic_features",
    "tls_node_features",
]
