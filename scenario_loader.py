"""Compatibility shim; new code should import from :mod:`rp.experiment`."""

from rp.experiment.scenario_loader import (
    ScenarioSpec,
    _build_ev_ids,
    _load_goal_nodes,
    build_testing_scenario,
    build_training_scenario,
    build_validation_scenario,
    count_evs_in_rou,
)

__all__ = [
    "ScenarioSpec",
    "build_training_scenario",
    "build_testing_scenario",
    "build_validation_scenario",
]
