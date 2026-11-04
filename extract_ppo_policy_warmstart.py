"""Compatibility entry point for :mod:`scripts.extract_ppo_policy_warmstart`."""

from scripts.extract_ppo_policy_warmstart import (
    TRAINING_STATE_PICKLE_MAGIC,
    extract_policy_weights,
    load_state_payload,
    main,
    save_warmstart,
)

__all__ = [
    "TRAINING_STATE_PICKLE_MAGIC",
    "load_state_payload",
    "extract_policy_weights",
    "save_warmstart",
    "main",
]

if __name__ == "__main__":
    main()
