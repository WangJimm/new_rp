"""Validation checkpoint/report persistence without training or a SUMO process."""
from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scripts import run_experiment as run


class TinyAgent:
    def save_tmp_file(self, episode, filename):
        torch.save({"episode": episode, "weight": torch.tensor([2., 3.])}, filename)


class ValidationHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)
        self.bundle = SimpleNamespace(
            config=SimpleNamespace(schedule=SimpleNamespace(num_episodes=3000, training_end=1200)),
            agent=TinyAgent(), agent_type="gine_action", rp_variant="proposed",
        )
        self.artifacts = run.create_training_artifacts()

    def result(self, successes=99, total=100, cost=10.):
        return {
            "success_count": successes, "ev_count": total,
            "success_rate": successes / total if total else 0.,
            "selection_time": cost, "total_time": 8., "total_penalized_time": cost,
            "total_dijkstra_time": 7., "better_ratio": .25, "total_count": 90,
            "cases": [{"seed": 40001, "failure_reasons": {"EV_2": "max_decisions"},
                       "ev_costs_minutes": [1., 2.], "penalized_ev_costs_minutes": [1., 50.]}],
            "agent_path": [["a", "b"]], "agent_ev_teleport_started_ids": [],
            "agent_cost_list": [8.], "dijkstra_cost_list": [7.],
        }

    def validate(self, result=None, episode=1199, close=False):
        result = self.result() if result is None else result
        with patch.object(run, "run_with_agent_eval_mode", return_value=result), \
             patch.object(run, "close_sumo_before_save", return_value=close) as stop, \
             patch.object(run, "restart_sumo_after_save") as restart:
            returned = run.maybe_run_validation_and_checkpoint(
                self.bundle, self.artifacts, episode, self.output,
                close_sumo_before_save_enabled=close,
            )
        return returned, stop, restart

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_sub_100pct_saves_loadable_snapshot_and_entire_report(self):
        result, _, _ = self.validate(episode=1499)
        path = self.output / "validation/episode_001500.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["episode"], 1499)
        self.assertTrue(torch.equal(checkpoint["weight"], torch.tensor([2., 3.])))
        report = self.read(path.with_suffix(".json"))
        self.assertEqual(report["episode_number"], 1500)
        self.assertEqual(report["checkpoint"], str(path))
        for key in ("cases", "agent_path", "agent_cost_list", "dijkstra_cost_list",
                    "agent_ev_teleport_started_ids"):
            self.assertEqual(report[key], result[key])
        self.assertFalse(report["arrival_gate_passed"])
        self.assertFalse((self.output / "gine_action_best_100pct_arrival_lowest_cost.pt").exists())
        self.assertTrue((self.output / "gine_action_best_arrival.pt").exists())
        self.assertEqual(self.artifacts.validation_records[-1]["validation_result_file"],
                         str(path.with_suffix(".json")))

    def test_higher_arrival_wins_even_with_higher_penalized_cost(self):
        self.validate(self.result(98, cost=8.))
        self.validate(self.result(99, cost=15.), episode=1349)
        saved = self.read(self.output / "gine_action_best_arrival.json")
        self.assertEqual(saved["success_count"], 99)
        self.assertEqual(saved["selection_time"], 15.)
        self.assertEqual(saved["episode_number"], 1350)
        self.assertEqual(saved["source_checkpoint"], str(self.output / "validation/episode_001350.pt"))
        self.assertEqual(self.artifacts.best_arrival_episode, 1349)

    def test_lower_arrival_does_not_replace_best_even_with_cheaper_cost(self):
        self.validate(self.result(99, cost=10.))
        result, _, _ = self.validate(self.result(98, cost=1.), episode=1349)
        self.assertFalse(result["saved_best_arrival_checkpoint"])
        self.assertEqual(self.read(self.output / "gine_action_best_arrival.json")["episode_number"], 1200)
        self.assertTrue((self.output / "validation/episode_001350.json").exists())

    def test_equal_arrival_uses_lower_cost_and_exact_ties_keep_previous(self):
        self.validate(self.result(cost=10.))
        improved, _, _ = self.validate(self.result(cost=9.), episode=1349)
        self.assertTrue(improved["saved_best_arrival_checkpoint"])
        tied, _, _ = self.validate(self.result(cost=9.), episode=1499)
        self.assertFalse(tied["saved_best_arrival_checkpoint"])
        self.assertEqual(self.read(self.output / "gine_action_best_arrival.json")["episode_number"], 1350)
        self.assertTrue((self.output / "validation/episode_001500.pt").exists())

    def test_arrival_ranking_uses_rate_rather_than_absolute_success_count(self):
        self.validate(self.result(99, total=100))
        result, _, _ = self.validate(self.result(100, total=200, cost=1.), episode=1349)
        self.assertFalse(result["saved_best_arrival_checkpoint"])

    def test_full_arrival_keeps_both_original_strict_models(self):
        result, _, _ = self.validate(self.result(100))
        self.assertTrue(result["arrival_gate_passed"])
        self.assertTrue(result["saved_lowest_cost_checkpoint"])
        self.assertTrue(result["saved_better_ratio_checkpoint"])
        self.assertEqual(len(list(self.output.rglob("*.pt"))), 4)
        self.assertTrue((self.output / "gine_action_best_100pct_arrival_lowest_cost.pt").exists())
        self.assertTrue((self.output / "gine_action_best_100pct_arrival_better_ratio.pt").exists())

    def test_no_eligible_best_does_not_suppress_snapshot_or_close_restart(self):
        self.validate(self.result(100, cost=5.))
        result, stop, restart = self.validate(self.result(99, cost=10.), episode=1349, close=True)
        self.assertFalse(result["saved_best_arrival_checkpoint"])
        self.assertFalse(result["saved_lowest_cost_checkpoint"])
        self.assertTrue(stop.call_args.args[0])
        self.assertTrue(restart.call_args.args[1])
        self.assertTrue((self.output / "validation/episode_001350.pt").exists())

    def test_zero_evs_and_nonfinite_cost_do_not_create_best_arrival(self):
        for result in (self.result(0, total=0), self.result(cost=float("nan"))):
            returned, _, _ = self.validate(result)
            self.assertFalse(returned["saved_best_arrival_checkpoint"])
        report = self.read(self.output / "validation/episode_001200.json")
        self.assertIsNone(report["selection_time"])
        self.assertFalse((self.output / "gine_action_best_arrival.pt").exists())

    def test_failed_evaluation_creates_no_successful_snapshot(self):
        with patch.object(run, "run_with_agent_eval_mode", side_effect=RuntimeError("SUMO failure")), \
             patch.object(run, "restart_sumo_session") as restart:
            result = run.maybe_run_validation_and_checkpoint(self.bundle, self.artifacts, 1199, self.output)
        self.assertIsNone(result)
        restart.assert_called_once()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_report_write_failure_does_not_advance_best_and_restarts_sumo(self):
        with patch.object(run, "run_with_agent_eval_mode", return_value=self.result()), \
             patch.object(run, "save_validation_result_atomic", side_effect=OSError("disk error")), \
             patch.object(run, "close_sumo_before_save", return_value=True), \
             patch.object(run, "restart_sumo_after_save") as restart:
            with self.assertRaisesRegex(OSError, "disk error"):
                run.maybe_run_validation_and_checkpoint(self.bundle, self.artifacts, 1199, self.output, True)
        self.assertEqual(self.artifacts.best_arrival_rate, float("-inf"))
        self.assertEqual(self.artifacts.validation_records, [])
        self.assertTrue(restart.call_args.args[1])

    def test_ranking_state_survives_autosave_artifact_roundtrip_and_rewards_json(self):
        self.validate(self.result(99, cost=8.))
        self.artifacts = run.normalize_training_artifacts(pickle.loads(pickle.dumps(self.artifacts)))
        result, _, _ = self.validate(self.result(99, cost=9.), episode=1349)
        self.assertFalse(result["saved_best_arrival_checkpoint"])
        run.save_training_rewards_json(self.artifacts, self.output / "training_rewards.json")
        report = self.read(self.output / "training_rewards.json")
        self.assertEqual(report["best_arrival_rate"], .99)
        self.assertEqual(report["best_arrival_cost"], 8.)
        self.assertEqual(report["best_arrival_episode"], 1199)
        old = SimpleNamespace(rewards=[])
        normalized = run.normalize_training_artifacts(old)
        self.assertEqual(normalized.best_arrival_rate, float("-inf"))

    def test_json_handles_numpy_paths_and_failed_replace_keeps_previous_report(self):
        path = self.output / "report.json"
        run.save_validation_result_atomic({"values": np.array([np.nan, 2.]), "path": self.output,
                                           "score": np.float32(3.5)}, path)
        before = path.read_bytes()
        report = self.read(path)
        self.assertEqual(report["values"], [None, 2.])
        self.assertEqual(report["path"], str(self.output))
        with patch.object(run, "safe_replace_file", side_effect=OSError("replace failed")):
            with self.assertRaises(OSError):
                run.save_validation_result_atomic({"new": True}, path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_existing_validation_schedule_is_unchanged(self):
        with patch.object(run, "run_with_agent_eval_mode") as evaluate:
            self.assertIsNone(run.maybe_run_validation_and_checkpoint(
                self.bundle, self.artifacts, 1049, self.output))
            self.assertIsNone(run.maybe_run_validation_and_checkpoint(
                self.bundle, self.artifacts, 1200, self.output))
        evaluate.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
