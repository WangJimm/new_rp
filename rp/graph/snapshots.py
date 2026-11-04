"""Tensor/data snapshot helpers used by replay-memory transitions."""

from __future__ import annotations

import copy
from typing import Any


def freeze_pyg_data(data: Any, *, torch_api: Any = None):
    """Clone PyG-like data and detach every tensor onto the CPU."""
    if torch_api is None:
        import torch as torch_api

    frozen = data.clone()
    for key in frozen.keys():
        value = frozen[key]
        if torch_api.is_tensor(value):
            frozen[key] = value.detach().clone().cpu()
        else:
            frozen[key] = copy.deepcopy(value)
    return frozen


__all__ = ["freeze_pyg_data"]
