from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rp.experiment import RuntimeConfig
from scripts import run_experiment


class EDQNFineTuneCheckpointingTests(unittest.TestCase):
    def make_bundle(self):
        schedule = SimpleNamespace(num_episodes=20, training_end=0)
        config = SimpleNamespace(schedule=schedule)
        return SimpleNamespace(
            config=config,
            agent=object(),
            agent_type="edqn",
            rp_variant="edqn_baseline",
        )

    @staticmethod
    def validation_result(*, success_count, ev_count, cost, better_ratio):
        return {
            "success_count": success_count,
            "ev_count": ev_count,
            "success_rate": success_count / ev_count,
            "selection_time": cost,
            "total_time": cost,
            "total_dijkstra_time": cost + 1.0,
            "better_ratio": better_ratio,
        }

    def run_checkpoint_selection(self, artifacts, result, output_dir):
        with (
            patch.object(run_experiment, "run_with_agent_eval_mode", return_value=result),
            patch.object(run_experiment, "save_agent_checkpoint_atomic") as save_checkpoint,
            patch.object(run_experiment, "close_sumo_before_save", return_value=False),
            patch.object(run_experiment, "restart_sumo_after_save"),
        ):
            returned = run_experiment.maybe_run_validation_and_checkpoint(
                self.make_bundle(),
                artifacts,
                episode=0,
                output_dir=output_dir,
            )
        return returned, save_checkpoint

    def test_checkpoint_is_rejected_below_full_arrival(self):
        artifacts = run_experiment.create_training_artifacts()
        result = self.validation_result(
            success_count=99,
            ev_count=100,
            cost=10.0,
            better_ratio=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            returned, save_checkpoint = self.run_checkpoint_selection(
                artifacts,
                result,
                Path(tmp),
            )

        save_checkpoint.assert_not_called()
        self.assertFalse(returned["arrival_gate_passed"])
        self.assertEqual(artifacts.best_validation_reward, float("inf"))
        self.assertEqual(artifacts.best_better_ratio, float("-inf"))

    def test_full_arrival_saves_both_independent_best_checkpoints(self):
        artifacts = run_experiment.create_training_artifacts()
        result = self.validation_result(
            success_count=100,
            ev_count=100,
            cost=12.0,
            better_ratio=0.8,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            returned, save_checkpoint = self.run_checkpoint_selection(
                artifacts,
                result,
                output_dir,
            )

        self.assertEqual(save_checkpoint.call_count, 2)
        saved_paths = {Path(call.args[2]).name for call in save_checkpoint.call_args_list}
        self.assertEqual(
            saved_paths,
            {
                "edqn_best_100pct_arrival_lowest_cost.pt",
                "edqn_best_100pct_arrival_better_ratio.pt",
            },
        )
        self.assertTrue(returned["arrival_gate_passed"])
        self.assertEqual(artifacts.best_validation_reward, 12.0)
        self.assertEqual(artifacts.best_better_ratio, 0.8)
        self.assertEqual(artifacts.best_better_ratio_cost, 12.0)

    def test_equal_better_ratio_uses_lower_cost_as_tiebreaker(self):
        artifacts = run_experiment.create_training_artifacts()
        artifacts.best_validation_reward = 10.0
        artifacts.best_better_ratio = 0.8
        artifacts.best_better_ratio_cost = 12.0
        result = self.validation_result(
            success_count=100,
            ev_count=100,
            cost=11.0,
            better_ratio=0.8,
        )

        with tempfile.TemporaryDirectory() as tmp:
            returned, save_checkpoint = self.run_checkpoint_selection(
                artifacts,
                result,
                Path(tmp),
            )

        save_checkpoint.assert_called_once()
        self.assertTrue(str(save_checkpoint.call_args.args[2]).endswith("best_100pct_arrival_better_ratio.pt"))
        self.assertFalse(returned["saved_lowest_cost_checkpoint"])
        self.assertTrue(returned["saved_better_ratio_checkpoint"])
        self.assertEqual(artifacts.best_better_ratio_cost, 11.0)

    def test_runtime_config_enables_revisit_penalty_by_default(self):
        fields = RuntimeConfig.__dataclass_fields__
        self.assertTrue(fields["revisit_penalty_enabled"].default)
        self.assertEqual(fields["revisit_penalty"].default, 1.0)


if __name__ == "__main__":
    unittest.main()
