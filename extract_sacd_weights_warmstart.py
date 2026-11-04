"""Compatibility entry point for :mod:`scripts.extract_sacd_weights_warmstart`."""

from scripts.extract_sacd_weights_warmstart import (
    SACD_WEIGHT_KEYS,
    TENSOR_NDARRAY_MARKER,
    TRAINING_STATE_PICKLE_MAGIC,
    extract_agent_state,
    load_state_payload,
    main,
    restore_autosave_tensors,
    save_warmstart,
)

__all__ = [
    "TRAINING_STATE_PICKLE_MAGIC",
    "TENSOR_NDARRAY_MARKER",
    "SACD_WEIGHT_KEYS",
    "load_state_payload",
    "restore_autosave_tensors",
    "extract_agent_state",
    "save_warmstart",
    "main",
]

if __name__ == "__main__":
    main()
