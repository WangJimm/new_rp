from __future__ import annotations

import argparse
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


TRAINING_STATE_PICKLE_MAGIC = b"RP_TRAINING_STATE_PICKLE_V1\n"
TENSOR_NDARRAY_MARKER = "__rp_tensor_ndarray_v1__"
SACD_WEIGHT_KEYS = ("actor", "q1", "q2", "target_q1", "target_q2", "log_alpha")


class _PicklePlaceholder:
    def __new__(cls, *args: Any, **kwargs: Any):
        return object.__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


class _StateUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return _PicklePlaceholder


def load_state_payload(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        prefix = f.read(len(TRAINING_STATE_PICKLE_MAGIC))
        if prefix == TRAINING_STATE_PICKLE_MAGIC:
            return restore_autosave_tensors(_StateUnpickler(f).load())

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This state is not the refactored pickle format, so torch is required to read it."
        ) from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def restore_autosave_tensors(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if isinstance(value, dict):
        if value.get(TENSOR_NDARRAY_MARKER) is True and "array" in value:
            import torch

            return torch.from_numpy(value["array"])
        return {key: restore_autosave_tensors(item, _seen) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        copied = value.copy()
        if copied.dtype == object:
            for index in np.ndindex(copied.shape):
                copied[index] = restore_autosave_tensors(copied[index], _seen)
        elif copied.dtype.fields:
            for field_name, (field_dtype, _) in copied.dtype.fields.items():
                if field_dtype == object:
                    for index in np.ndindex(copied.shape):
                        copied[field_name][index] = restore_autosave_tensors(copied[field_name][index], _seen)
        return copied
    if isinstance(value, list):
        return [restore_autosave_tensors(item, _seen) for item in value]
    if isinstance(value, tuple):
        items = [restore_autosave_tensors(item, _seen) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, deque):
        return deque((restore_autosave_tensors(item, _seen) for item in value), maxlen=value.maxlen)
    if hasattr(value, "__dict__"):
        object_id = id(value)
        if object_id in _seen:
            return value
        _seen.add(object_id)
        for attr_name, attr_value in list(vars(value).items()):
            try:
                setattr(value, attr_name, restore_autosave_tensors(attr_value, _seen))
            except Exception:
                pass
        _seen.remove(object_id)
    return value


def extract_agent_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("agent_state")
    if not isinstance(state, dict):
        state = payload
    missing = [key for key in SACD_WEIGHT_KEYS if key not in state]
    if missing:
        raise KeyError(f"Input does not look like a SACD state; missing keys: {missing}")
    return {key: state[key] for key in SACD_WEIGHT_KEYS}


def save_warmstart(path: Path, agent_state: dict[str, Any], source: Path, source_episode: Any) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to write the warm-start checkpoint.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": 0,
            "agent_type": "sacd_weights_warm_start",
            "agent_state": agent_state,
            "source_state": str(source),
            "source_episode": source_episode,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SACD network weights from a training_state_latest.pt file for PER warm-start training."
    )
    parser.add_argument("--input", required=True, type=Path, help="Old SACD training_state_latest.pt")
    parser.add_argument("--output", required=True, type=Path, help="Output SACD warm-start .pt checkpoint")
    args = parser.parse_args()

    payload = load_state_payload(args.input)
    agent_state = extract_agent_state(payload)
    save_warmstart(args.output, agent_state, args.input, payload.get("episode"))
    print(f"saved SACD weights warm-start checkpoint: {args.output}")


if __name__ == "__main__":
    main()
