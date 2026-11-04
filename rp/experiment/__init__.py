"""Stable, lazily loaded public API for experiment orchestration."""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "PROJECT_DIR": ".config",
    "ENV_CHOICES": ".config",
    "build_new_env_config": ".config",
    "build_selected_env_config": ".config",
    "validate_environment_inputs": ".config",
    "resolve_sumo_home": ".config",
    "resolve_sumo_config": ".config",
    "RPVariantSpec": ".config",
    "RP_VARIANT_SPECS": ".config",
    "RP_VARIANT_CHOICES": ".config",
    "resolve_rp_variant_spec": ".config",
    "SumoPaths": ".config",
    "TrainingSchedule": ".config",
    "EnvironmentConfig": ".config",
    "RuntimeConfig": ".config",
    "build_small_env_config": ".config",
    "build_env_config": ".config",
    "build_static_env_config": ".config",
    "EMAFeature": ".runtime_state",
    "EpisodeLogger": ".runtime_state",
    "EVRuntimeState": ".runtime_state",
    "ScenarioSpec": ".scenario_loader",
    "build_training_scenario": ".scenario_loader",
    "build_testing_scenario": ".scenario_loader",
    "build_validation_scenario": ".scenario_loader",
    "ScenarioKey": ".scenario_selection",
    "choose_training_scenario_key": ".scenario_selection",
    "build_training_scenario_schedule": ".scenario_selection",
    "iter_training_scenario_keys": ".scenario_selection",
    "iter_testing_scenario_keys": ".scenario_selection",
    "iter_validation_scenario_keys": ".scenario_selection",
    "ExecutionMode": ".training",
    "EpsilonScheduler": ".training",
    "dijkstra_guidance_probability": ".training",
    "TrainingProgress": ".training",
    "build_execution_mode": ".training",
    "resolve_training_scenario": ".training",
    "resolve_testing_scenarios": ".training",
    "resolve_validation_scenarios": ".training",
    "resolve_training_scenarios": ".training",
    "AgentLearningError": ".episode_runner",
    "TransitionPayload": ".episode_runner",
    "EpisodeConfig": ".episode_runner",
    "EpisodeRunInput": ".episode_runner",
    "DQNGNNEpisodeRunner": ".episode_runner",
    "resolve_ev_ids": ".episode_runner",
    "TrafficJamConfig": ".dijkstra_runner",
    "DijkstraRunConfig": ".dijkstra_runner",
    "DijkstraEVState": ".dijkstra_runner",
    "DijkstraSimulationResult": ".dijkstra_runner",
    "TrafficJamController": ".dijkstra_runner",
    "dijkstra_run_config_from_runtime": ".dijkstra_runner",
    "DijkstraBaselineRunner": ".dijkstra_runner",
    "run_dijkstra_baseline_for_testing_key": ".dijkstra_runner",
    "run_dijkstra_baseline_for_training_key": ".dijkstra_runner",
    "run_dijkstra_baseline_for_validation_key": ".dijkstra_runner",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)
