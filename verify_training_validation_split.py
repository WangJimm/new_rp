"""Compatibility entry point for the training/validation split verifier."""

from scripts.verify_training_validation_split import (
    main,
    parse_args,
    scenario_file_preview,
    validation_trigger_steps,
)

__all__ = [
    "parse_args",
    "scenario_file_preview",
    "validation_trigger_steps",
    "main",
]

if __name__ == "__main__":
    main()
