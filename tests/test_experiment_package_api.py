from __future__ import annotations

import unittest
from types import SimpleNamespace

import config
import scenario_loader
import scenario_selection
import training_orchestrator
from rp import experiment
from rp.experiment import scenario_loader as loader_implementation

try:
    import episode_runner

    EPISODE_RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    EPISODE_RUNTIME_AVAILABLE = False


class ExperimentPackageApiTests(unittest.TestCase):
    def test_public_api_exports_stable_experiment_operations(self):
        expected = {
            "ScenarioSpec",
            "ScenarioKey",
            "build_training_scenario",
            "build_testing_scenario",
            "choose_training_scenario_key",
            "build_training_scenario_schedule",
            "ExecutionMode",
            "EpsilonScheduler",
            "TrainingProgress",
            "resolve_training_scenarios",
            "AgentLearningError",
            "TransitionPayload",
            "EpisodeConfig",
            "EpisodeRunInput",
            "DQNGNNEpisodeRunner",
            "resolve_ev_ids",
            "RuntimeConfig",
            "TrainingSchedule",
            "build_static_env_config",
            "DijkstraBaselineRunner",
            "run_dijkstra_baseline_for_testing_key",
        }
        self.assertTrue(expected.issubset(set(experiment.__all__)))

    def test_private_scenario_helpers_are_not_public_package_api(self):
        self.assertNotIn("_build_ev_ids", experiment.__all__)
        self.assertNotIn("_load_goal_nodes", experiment.__all__)

    def test_root_modules_are_identity_preserving_compatibility_shims(self):
        self.assertIs(scenario_loader.ScenarioSpec, experiment.ScenarioSpec)
        self.assertIs(scenario_selection.ScenarioKey, experiment.ScenarioKey)
        self.assertIs(
            training_orchestrator.EpsilonScheduler,
            experiment.EpsilonScheduler,
        )
        self.assertIs(
            scenario_loader._build_ev_ids,
            loader_implementation._build_ev_ids,
        )
        self.assertIs(config.RuntimeConfig, experiment.RuntimeConfig)
        self.assertIs(
            config.build_static_env_config,
            experiment.build_static_env_config,
        )


if EPISODE_RUNTIME_AVAILABLE:

    class EpisodeRunnerCompatibilityTests(unittest.TestCase):
        def test_root_episode_runner_shares_public_implementation(self):
            self.assertIs(episode_runner.EpisodeConfig, experiment.EpisodeConfig)
            self.assertIs(
                episode_runner.TransitionPayload,
                experiment.TransitionPayload,
            )
            self.assertIs(
                episode_runner.DQNGNNEpisodeRunner,
                experiment.DQNGNNEpisodeRunner,
            )


try:
    import dijkstra_refactored

    DIJKSTRA_RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    DIJKSTRA_RUNTIME_AVAILABLE = False


if DIJKSTRA_RUNTIME_AVAILABLE:

    class DijkstraRunnerCompatibilityTests(unittest.TestCase):
        def test_root_dijkstra_runner_shares_public_implementation(self):
            self.assertIs(
                dijkstra_refactored.DijkstraBaselineRunner,
                experiment.DijkstraBaselineRunner,
            )
            self.assertIs(
                dijkstra_refactored.run_dijkstra_baseline_for_testing_key,
                experiment.run_dijkstra_baseline_for_testing_key,
            )

        def test_default_travel_time_cost_is_bound_to_each_runner(self):
            def make_traci(value):
                return SimpleNamespace(
                    edge=SimpleNamespace(
                        getTraveltime=lambda _edge_id: value,
                    )
                )

            first = experiment.DijkstraBaselineRunner(
                traci=make_traci(11.0),
                net=object(),
                static_pack={"adj": {}},
                runtime_config=object(),
            )
            second = experiment.DijkstraBaselineRunner(
                traci=make_traci(22.0),
                net=object(),
                static_pack={"adj": {}},
                runtime_config=object(),
            )

            self.assertEqual(first.cost_fn("edge"), 11.0)
            self.assertEqual(second.cost_fn("edge"), 22.0)


if __name__ == "__main__":
    unittest.main()
