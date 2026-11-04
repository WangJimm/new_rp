"""Compatibility shim; new code should import from :mod:`rp.experiment`."""

from rp.experiment.scenario_selection import (
    ScenarioKey,
    build_training_scenario_schedule,
    choose_training_scenario_key,
    iter_testing_scenario_keys,
    iter_training_scenario_keys,
    iter_validation_scenario_keys,
)

__all__ = [
    "ScenarioKey",
    "choose_training_scenario_key",
    "build_training_scenario_schedule",
    "iter_training_scenario_keys",
    "iter_testing_scenario_keys",
    "iter_validation_scenario_keys",
]
