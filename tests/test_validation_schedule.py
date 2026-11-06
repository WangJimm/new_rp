"""Check one-based validation scheduling independently of exploration and SUMO."""
from __future__ import annotations

import argparse
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rp.experiment.config import RuntimeConfig, TrainingSchedule
from scripts import run_experiment as run


class ValidationScheduleTests(unittest.TestCase):
    def config(self, **kwargs):
        return RuntimeConfig(repo_root=Path('.'), paths=None, environment=None,
            schedule=TrainingSchedule(num_episodes=3000, warmup_episodes=225, epsilon_decay_end_episode=1200), **kwargs)

    def episodes(self, config):
        return [i + 1 for i in range(config.schedule.num_episodes)
                if run.training_validation_due(config, i)]

    def test_requested_schedule_has_exactly_six_evaluations(self):
        config = self.config(validation_start_episode=2000, validation_interval=200)
        self.assertEqual(self.episodes(config), [2000, 2200, 2400, 2600, 2800, 3000])

    def test_unspecified_schedule_keeps_original_13_evaluations(self):
        self.assertEqual(self.episodes(self.config()), list(range(1200, 3001, 150)))

    def test_legacy_rounding_and_short_runs_are_preserved(self):
        config = replace(self.config(), schedule=TrainingSchedule(
            num_episodes=3000, epsilon_decay_end_episode=1251))
        self.assertEqual(self.episodes(config)[0], 1350)
        config = replace(config, schedule=TrainingSchedule(num_episodes=20,
            warmup_ratio=0, training_ratio=0))
        self.assertEqual(self.episodes(config), list(range(1, 21)))

    def test_explicit_start_is_the_anchor_even_when_not_a_multiple(self):
        config = self.config(validation_start_episode=2050, validation_interval=200)
        self.assertEqual(self.episodes(config), [2050, 2250, 2450, 2650, 2850])

    def test_validation_does_not_change_or_wait_for_epsilon_boundaries(self):
        config = replace(self.config(), schedule=TrainingSchedule(num_episodes=3000))
        updated = run.apply_validation_schedule_args(config, argparse.Namespace(
            validation_start_episode=2000, validation_interval=200))
        self.assertEqual(updated.schedule, config.schedule)
        self.assertEqual(updated.schedule.warmup_end, 450)
        self.assertEqual(updated.schedule.training_end, 2400)
        self.assertTrue(run.training_validation_due(updated, 1999))
        config = replace(updated, schedule=TrainingSchedule(num_episodes=3000,
            warmup_episodes=225, epsilon_decay_end_episode=1200))
        self.assertEqual(config.schedule.warmup_end, 225)
        self.assertEqual(config.schedule.training_end, 1200)
        self.assertEqual(self.episodes(config), [2000, 2200, 2400, 2600, 2800, 3000])

    def test_zero_based_runner_boundaries_are_correct(self):
        config = self.config(validation_start_episode=2000, validation_interval=200)
        for internal in [1199, 1499, 1998, 2000, 2198, 2200]:
            self.assertFalse(run.training_validation_due(config, internal))
        self.assertTrue(run.training_validation_due(config, 1999))
        self.assertTrue(run.training_validation_due(config, 2199))
        self.assertTrue(run.training_validation_due(config, 2999))

    def test_partial_override_preserves_other_setting(self):
        config = self.config(validation_start_episode=2000, validation_interval=200)
        updated = run.apply_validation_schedule_args(config, argparse.Namespace(validation_interval=250))
        self.assertEqual(run.resolve_training_validation_schedule(updated), (2000, 250))
        legacy = replace(self.config(), schedule=TrainingSchedule(num_episodes=3000,
            warmup_episodes=225, epsilon_decay_end_episode=1200))
        updated = run.apply_validation_schedule_args(legacy, argparse.Namespace(validation_interval=200))
        self.assertEqual(run.resolve_training_validation_schedule(updated), (1200, 200))

    def test_invalid_start_and_interval_are_rejected_before_running(self):
        for key, value in [('validation_start_episode', 0), ('validation_start_episode', -1),
                           ('validation_interval', 0), ('validation_interval', -200)]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    run.apply_validation_schedule_args(self.config(), argparse.Namespace(**{key: value}))

    def test_start_after_training_end_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exceeds --num-episodes'):
            run.apply_validation_schedule_args(self.config(), argparse.Namespace(
                validation_start_episode=3200, validation_interval=200))

    def test_cli_accepts_requested_settings(self):
        with patch('sys.argv', ['run_experiment', '--env', 'new_env', '--num-episodes', '3000',
                              '--validation-start-episode', '2000', '--validation-interval', '200']):
            args = run.parse_args()
        self.assertEqual(args.validation_start_episode, 2000)
        self.assertEqual(args.validation_interval, 200)

    def test_no_override_leaves_config_unchanged(self):
        config = self.config()
        self.assertIs(run.apply_validation_schedule_args(config, argparse.Namespace()), config)

    def test_runner_uses_custom_schedule_and_records_it_in_saved_results(self):
        bundle = SimpleNamespace(config=self.config(validation_start_episode=2000, validation_interval=200),
            agent=object(), agent_type='gine_action', rp_variant='proposed')
        result = {'success_count': 99, 'ev_count': 100, 'success_rate': .99,
                  'selection_time': 10., 'total_time': 8., 'total_dijkstra_time': 7., 'better_ratio': .1}
        with patch.object(run, 'run_with_agent_eval_mode', return_value=result) as evaluate, \
             patch.object(run, 'save_agent_checkpoint_atomic') as save, \
             patch.object(run, 'save_validation_result_atomic') as report, \
             patch.object(run, 'close_sumo_before_save', return_value=False), \
             patch.object(run, 'restart_sumo_after_save'):
            artifacts = run.create_training_artifacts()
            self.assertIsNone(run.maybe_run_validation_and_checkpoint(bundle, artifacts, 1199, Path('out')))
            evaluate.assert_not_called()
            returned = run.maybe_run_validation_and_checkpoint(bundle, artifacts, 1999, Path('out'))
        evaluate.assert_called_once()
        self.assertEqual(returned['validation_schedule'], {'start_episode': 2000, 'interval': 200})
        self.assertEqual(returned['episode_number'], 2000)
        self.assertIn('episode_002000.pt', returned['checkpoint'])
        self.assertEqual(report.call_count, 2)
        self.assertEqual(save.call_count, 2)


if __name__ == '__main__':
    unittest.main()
