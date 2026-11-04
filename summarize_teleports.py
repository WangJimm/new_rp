"""Compatibility entry point for :mod:`scripts.summarize_teleports`."""

from scripts.summarize_teleports import (
    DEFAULT_SCENARIOS,
    build_scenario_key,
    main,
    parse_args,
    summarize_records,
    summarize_validation_result,
)

__all__ = [
    "DEFAULT_SCENARIOS",
    "parse_args",
    "build_scenario_key",
    "summarize_records",
    "summarize_validation_result",
    "main",
]

if __name__ == "__main__":
    main()
