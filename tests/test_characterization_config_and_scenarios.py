from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from config import (
    PROJECT_DIR,
    RP_VARIANT_CHOICES,
    TrainingSchedule,
    resolve_rp_variant_spec,
    resolve_sumo_config,
)
from scenario_selection import (
    ScenarioKey,
    build_training_scenario_schedule,
    iter_testing_scenario_keys,
    iter_training_scenario_keys,
    iter_validation_scenario_keys,
)


def make_config(
    *,
    training_flows: tuple[str, ...] = ("free", "normal", "busy"),
    testing_flows: tuple[str, ...] = ("free", "normal", "busy", "heavy"),
    validation_flows: tuple[str, ...] | None = None,
    training_seed_start: int = 100,
    training_seed_end: int = 150,
    testing_seed_start: int = 1,
    testing_seed_end: int = 10,
    validation_seed_start: int | None = None,
    validation_seed_end: int | None = None,
) -> SimpleNamespace:
    environment = SimpleNamespace(
        training_flows=training_flows,
        testing_flows=testing_flows,
        validation_flows=validation_flows or testing_flows,
        training_seed_start=training_seed_start,
        training_seed_end=training_seed_end,
        testing_seed_start=testing_seed_start,
        testing_seed_end=testing_seed_end,
        validation_seed_start=validation_seed_start,
        validation_seed_end=validation_seed_end,
    )
    return SimpleNamespace(environment=environment)


class ConfigCharacterizationTests(unittest.TestCase):
    """Capture configuration behavior before modules are reorganized."""

    def test_project_directory_stays_at_the_checkout_root(self) -> None:
        self.assertEqual(PROJECT_DIR, Path(__file__).resolve().parents[1])

    def test_training_schedule_truncates_fractional_boundaries(self) -> None:
        schedule = TrainingSchedule(
            num_episodes=11,
            warmup_ratio=0.15,
            training_ratio=0.8,
        )

        self.assertEqual(schedule.warmup_end, 1)
        self.assertEqual(schedule.training_end, 8)

    def test_explicit_epsilon_boundaries_do_not_scale_with_extended_training(self) -> None:
        schedule = TrainingSchedule(
            num_episodes=2500,
            warmup_ratio=0.15,
            training_ratio=0.8,
            warmup_episodes=225,
            epsilon_decay_end_episode=1200,
        )

        self.assertEqual(schedule.warmup_end, 225)
        self.assertEqual(schedule.training_end, 1200)

    def test_sumo_config_prefers_plain_name_then_xml_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            plain_path = run_dir / "osm.sumocfg"
            xml_path = run_dir / "osm.sumocfg.xml"

            self.assertEqual(resolve_sumo_config(run_dir), plain_path)

            xml_path.touch()
            self.assertEqual(resolve_sumo_config(run_dir), xml_path)

            plain_path.touch()
            self.assertEqual(resolve_sumo_config(run_dir), plain_path)

    def test_registered_variant_to_agent_mapping_is_stable(self) -> None:
        expected = {
            "dqn_baseline": ("dqn", None),
            "edqn_baseline": ("edqn", None),
            "ppo_baseline": ("ppo_rllib", None),
            "ppo_torch_baseline": ("ppo_torch", None),
            "sacd_baseline": ("sacd", None),
            "gine_no_edge_features": ("gine_action", "none"),
            "gine_fixed_action_head": ("gine", "static_occupancy"),
            "proposed": ("gine_action", "static_occupancy"),
        }

        self.assertEqual(set(RP_VARIANT_CHOICES), set(expected))
        for name, (agent_type, edge_feature_mode) in expected.items():
            with self.subTest(name=name):
                spec = resolve_rp_variant_spec(name)
                self.assertEqual(spec.name, name)
                self.assertEqual(spec.agent_type, agent_type)
                self.assertEqual(spec.edge_feature_mode, edge_feature_mode)

    def test_unknown_variant_error_includes_the_invalid_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown_variant"):
            resolve_rp_variant_spec("unknown_variant")


class ScenarioSelectionCharacterizationTests(unittest.TestCase):
    """Freeze seeded scheduling and exhaustive scenario iteration order."""

    def test_seed_42_training_schedule_matches_current_sequence(self) -> None:
        config = make_config()

        actual = build_training_scenario_schedule(config, num_episodes=8, seed=42)

        self.assertEqual(
            actual,
            [
                ScenarioKey(2, 1, 101),
                ScenarioKey(2, 2, 115),
                ScenarioKey(0, 1, 147),
                ScenarioKey(0, 1, 137),
                ScenarioKey(1, 1, 101),
                ScenarioKey(0, 1, 114),
                ScenarioKey(2, 1, 135),
                ScenarioKey(0, 2, 114),
            ],
        )

    def test_training_keys_are_seed_then_probability_then_flow(self) -> None:
        config = make_config(training_seed_start=7, training_seed_end=8)

        actual = iter_training_scenario_keys(config)

        self.assertEqual(len(actual), 12)
        self.assertEqual(
            actual[:6],
            [
                ScenarioKey(0, 1, 7),
                ScenarioKey(1, 1, 7),
                ScenarioKey(2, 1, 7),
                ScenarioKey(0, 2, 7),
                ScenarioKey(1, 2, 7),
                ScenarioKey(2, 2, 7),
            ],
        )
        self.assertEqual(actual[-1], ScenarioKey(2, 2, 8))

    def test_testing_keys_cover_both_probabilities_for_each_seed(self) -> None:
        config = make_config(
            testing_flows=("normal", "busy"),
            testing_seed_start=3,
            testing_seed_end=4,
        )

        actual = iter_testing_scenario_keys(config)

        self.assertEqual(
            actual,
            [
                ScenarioKey(0, 1, 3),
                ScenarioKey(1, 1, 3),
                ScenarioKey(0, 2, 3),
                ScenarioKey(1, 2, 3),
                ScenarioKey(0, 1, 4),
                ScenarioKey(1, 1, 4),
                ScenarioKey(0, 2, 4),
                ScenarioKey(1, 2, 4),
            ],
        )

    def test_validation_uses_testing_seed_range_when_not_configured(self) -> None:
        config = make_config(
            validation_flows=("normal",),
            testing_seed_start=9,
            testing_seed_end=10,
        )

        actual = iter_validation_scenario_keys(config)

        self.assertEqual(
            actual,
            [
                ScenarioKey(0, 1, 9),
                ScenarioKey(0, 2, 9),
                ScenarioKey(0, 1, 10),
                ScenarioKey(0, 2, 10),
            ],
        )

    def test_validation_honors_its_own_seed_range(self) -> None:
        config = make_config(
            validation_flows=("normal",),
            testing_seed_start=1,
            testing_seed_end=2,
            validation_seed_start=20,
            validation_seed_end=20,
        )

        self.assertEqual(
            iter_validation_scenario_keys(config),
            [ScenarioKey(0, 1, 20), ScenarioKey(0, 2, 20)],
        )


if __name__ == "__main__":
    unittest.main()
