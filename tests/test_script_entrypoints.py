from __future__ import annotations

import unittest
from pathlib import Path

from rp.experiment import PROJECT_DIR

import extract_ppo_policy_warmstart as ppo_compat
import extract_sacd_weights_warmstart as sacd_compat
import summarize_teleports as teleport_compat
import verify_training_validation_split as split_compat
from scripts import extract_ppo_policy_warmstart as ppo_script
from scripts import extract_sacd_weights_warmstart as sacd_script
from scripts import summarize_teleports as teleport_script
from scripts import verify_training_validation_split as split_script


class ScriptCompatibilityTests(unittest.TestCase):
    def test_root_entrypoints_share_script_implementations(self):
        self.assertIs(ppo_compat.main, ppo_script.main)
        self.assertIs(sacd_compat.main, sacd_script.main)
        self.assertIs(teleport_compat.main, teleport_script.main)
        self.assertIs(split_compat.main, split_script.main)

    def test_split_preview_uses_the_checkout_root_directly(self):
        actual = split_script.scenario_file_preview(
            PROJECT_DIR,
            "env",
            "training_data",
            "1800",
            1,
            7,
        )
        self.assertEqual(
            actual,
            (
                PROJECT_DIR / "env" / "training_data" / "output_1800_7.rou.xml",
                PROJECT_DIR / "env" / "training_data" / "ev_1800_p01_7.rou.xml",
                PROJECT_DIR / "env" / "training_data_goal" / "goal_ev_1800_p01_7.json",
            ),
        )

    def test_validation_trigger_schedule_is_preserved(self):
        schedule = type(
            "Schedule",
            (),
            {"num_episodes": 100, "training_end": 40},
        )()
        self.assertEqual(
            split_script.validation_trigger_steps(schedule),
            list(range(40, 101, 5)),
        )

    def test_ppo_policy_extraction_schema_is_preserved(self):
        weights = {"layer": [1, 2]}
        self.assertIs(
            ppo_script.extract_policy_weights(
                {"agent_state": {"policy_weights": weights}}
            ),
            weights,
        )

    def test_sacd_weight_extraction_schema_is_preserved(self):
        state = {key: object() for key in sacd_script.SACD_WEIGHT_KEYS}
        self.assertEqual(sacd_script.extract_agent_state(state), state)

    def test_teleport_summary_schema_is_preserved(self):
        result = teleport_script.summarize_validation_result(
            {
                "agent_teleport_scenarios": [
                    {
                        "scenario_index": 1,
                        "probability_index": 2,
                        "seed": 51,
                        "teleport_started_ids": ["EV_1"],
                    }
                ]
            },
            ["1500", "1800", "2100"],
        )
        self.assertEqual(
            result,
            {
                "agent": {
                    "1800_02_51": {
                        "teleport_started_ids": ["EV_1"],
                        "teleport_ended_ids": [],
                    }
                },
                "dijkstra": {},
            },
        )


try:
    import plot_dijkstra_over_agent as plot_compat
    from scripts import plot_dijkstra_over_agent as plot_script

    PLOTTING_RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    PLOTTING_RUNTIME_AVAILABLE = False


if PLOTTING_RUNTIME_AVAILABLE:

    class PlotScriptCompatibilityTests(unittest.TestCase):
        def test_root_plot_entrypoint_shares_script_implementation(self):
            self.assertIs(plot_compat.main, plot_script.main)
            self.assertEqual(
                plot_script.moving_average([1.0, 3.0, 5.0], 2),
                [1.0, 2.0, 4.0],
            )


try:
    import compare_training_rewards_to_dijkstra as compare_compat
    import evaluate as evaluate_compat
    import route_planning_main as run_compat
    import validation as validation_compat
    from scripts import compare_training_rewards_to_dijkstra as compare_script
    from scripts import evaluate as evaluate_script
    from scripts import run_experiment as run_script
    from scripts import validation as validation_script

    SUMO_SCRIPT_RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    SUMO_SCRIPT_RUNTIME_AVAILABLE = False


if SUMO_SCRIPT_RUNTIME_AVAILABLE:

    class SumoScriptCompatibilityTests(unittest.TestCase):
        def test_root_entrypoints_share_script_implementations(self):
            self.assertIs(evaluate_compat.main, evaluate_script.main)
            self.assertIs(validation_compat.main, validation_script.main)
            self.assertIs(compare_compat.main, compare_script.main)
            self.assertIs(run_compat.main, run_script.main)

        def test_run_entrypoint_uses_the_checkout_root(self):
            project_root = Path(run_compat.__file__).resolve().parent
            self.assertEqual(run_script.THIS_DIR, project_root)
            self.assertEqual(run_script.THIS_DIR, PROJECT_DIR)

        def test_active_scripts_do_not_reconstruct_a_parent_checkout_layout(self):
            for script_module in (
                evaluate_script,
                validation_script,
                compare_script,
                run_script,
                split_script,
            ):
                with self.subTest(script=script_module.__name__):
                    source = Path(script_module.__file__).read_text(encoding="utf-8")
                    self.assertNotIn("parents[2]", source)
                    self.assertNotIn('repo_root / "route_planning"', source)

        def test_active_runners_do_not_assign_the_compatibility_traci_global(self):
            for relative_path in (
                Path("scripts/run_experiment.py"),
                Path("rp/experiment/dijkstra_runner.py"),
            ):
                with self.subTest(path=relative_path):
                    source = (PROJECT_DIR / relative_path).read_text(encoding="utf-8")
                    self.assertNotIn("utils_sumo.traci", source)


if __name__ == "__main__":
    unittest.main()
