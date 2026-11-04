from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import TrainingSchedule
from rp.experiment import scenario_loader
from rp.experiment import training as training_orchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


@contextmanager
def temporary_module(name: str, module: types.ModuleType):
    previous = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def load_module(
    module_name: str,
    filename: str,
    injected: dict[str, types.ModuleType] | None = None,
) -> types.ModuleType:
    injected = injected or {}
    previous_injected = {
        name: sys.modules.get(name, _MISSING)
        for name in injected
    }
    previous_loaded = sys.modules.get(module_name, _MISSING)
    try:
        sys.modules.update(injected)
        spec = importlib.util.spec_from_file_location(module_name, PROJECT_ROOT / filename)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_loaded is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_loaded
        for name, previous in previous_injected.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


sumo_runtime = load_module(
    "_characterization_sumo_runtime",
    "sumo_runtime.py",
    {"scenario_loader": scenario_loader},
)


class ScenarioLoaderCharacterizationTests(unittest.TestCase):
    def test_ev_ids_follow_real_xml_order_without_renumbering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "ev.rou.xml"
            path.write_text('<routes><vehicle id="EV_1"/><vehicle id="EV_9"/><vehicle id="EV_3"/></routes>', encoding="utf-8")
            actual = scenario_loader._build_ev_ids(path)
        self.assertEqual(actual, ["EV_1", "EV_9", "EV_3"])

    def test_goal_nodes_require_exact_ev_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            goal_file = Path(tmp_dir) / "goals.json"
            goal_file.write_text(json.dumps(["node-a", "node-b"]), encoding="utf-8")

            actual = scenario_loader._load_goal_nodes(
                goal_file,
                ["EV_0", "EV_1"],
            )

            with self.assertRaisesRegex(ValueError, "goals=2, evs=3"):
                scenario_loader._load_goal_nodes(
                    goal_file,
                    ["EV_0", "EV_1", "EV_2"],
                )

        self.assertEqual(
            actual,
            {
                "EV_0": "node-a",
                "EV_1": "node-b",
            },
        )

    def test_scenario_builders_keep_current_paths_and_route_argument_format(self) -> None:
        run_dir = Path("experiment-env")
        environment = SimpleNamespace(
            training_flows=("train-flow",),
            testing_flows=("test-flow",),
            validation_flows=("validation-flow",),
        )
        config = SimpleNamespace(
            environment=environment,
            paths=SimpleNamespace(run_dir=run_dir),
        )
        cases = [
            (
                scenario_loader.build_training_scenario,
                "train-flow",
                "training_data",
                "training_data_goal",
            ),
            (
                scenario_loader.build_testing_scenario,
                "test-flow",
                "testing_data",
                "testing_data_goal",
            ),
            (
                scenario_loader.build_validation_scenario,
                "validation-flow",
                "validation_data",
                "validation_data_goal",
            ),
        ]

        for builder, flow, data_dir, goal_dir in cases:
            with self.subTest(builder=builder.__name__):
                with (
                    patch.object(scenario_loader, "_build_ev_ids", return_value=["EV_0", "EV_1"]),
                    patch.object(
                        scenario_loader,
                        "_load_goal_nodes",
                        return_value={"EV_0": "goal-a", "EV_1": "goal-b"},
                    ) as goal_loader,
                ):
                    spec = builder(
                        config,
                        scenario_index=0,
                        probability_index=2,
                        seed=17,
                    )

                expected_general = run_dir / data_dir / f"output_{flow}_17.rou.xml"
                expected_ev = run_dir / data_dir / f"ev_{flow}_p02_17.rou.xml"
                expected_goal = run_dir / goal_dir / f"goal_ev_{flow}_p02_17.json"
                self.assertEqual(spec.general_route_file, expected_general)
                self.assertEqual(spec.ev_route_file, expected_ev)
                self.assertEqual(spec.goal_file, expected_goal)
                self.assertEqual(
                    spec.route_files_argument,
                    f"{expected_general}, {expected_ev}",
                )
                goal_loader.assert_called_once_with(
                    expected_goal,
                    ["EV_0", "EV_1"],
                )


class SumoCommandCharacterizationTests(unittest.TestCase):
    def test_load_arguments_keep_current_order_and_string_conversion(self) -> None:
        config = SimpleNamespace(
            paths=SimpleNamespace(config_file=Path("experiment-env/osm.sumocfg")),
            step_length=0.5,
        )
        scenario = scenario_loader.ScenarioSpec(
            general_route_file=Path("general.rou.xml"),
            ev_route_file=Path("ev.rou.xml"),
            goal_file=Path("goal.json"),
            ev_ids=["EV_0"],
            goal_nodes={"EV_0": "goal-a"},
            route_files_argument="general.rou.xml, ev.rou.xml",
        )

        self.assertEqual(
            sumo_runtime.build_load_args(config, scenario),
            [
                "-c",
                str(Path("experiment-env/osm.sumocfg")),
                "--step-length",
                "0.5",
                "--route-files",
                "general.rou.xml, ev.rou.xml",
                "-W",
                "true",
            ],
        )


class TrainingOrchestratorCharacterizationTests(unittest.TestCase):
    def test_epsilon_schedule_keeps_current_phase_boundaries(self) -> None:
        scheduler = training_orchestrator.EpsilonScheduler(
            schedule=TrainingSchedule(
                num_episodes=10,
                warmup_ratio=0.2,
                training_ratio=0.8,
            ),
            min_epsilon=0.1,
        )

        self.assertEqual(scheduler.value(0), 1.0)
        self.assertEqual(scheduler.value(1), 1.0)
        self.assertEqual(scheduler.value(2), 1.0)
        self.assertAlmostEqual(scheduler.value(5), 0.55)
        self.assertAlmostEqual(scheduler.value(7), 0.25)
        self.assertEqual(scheduler.value(8), 0.1)
        self.assertEqual(scheduler.value(100), 0.1)

    def test_extended_training_keeps_epsilon_at_minimum_after_original_decay(self) -> None:
        scheduler = training_orchestrator.EpsilonScheduler(
            schedule=TrainingSchedule(
                num_episodes=2500,
                warmup_episodes=225,
                epsilon_decay_end_episode=1200,
            ),
            min_epsilon=0.05,
        )

        self.assertEqual(scheduler.value(224), 1.0)
        self.assertEqual(scheduler.value(1200), 0.05)
        self.assertEqual(scheduler.value(2499), 0.05)

    def test_execution_mode_and_scenario_hit_counting_are_stable(self) -> None:
        self.assertEqual(
            training_orchestrator.build_execution_mode(False),
            training_orchestrator.ExecutionMode.TRAIN,
        )
        self.assertEqual(
            training_orchestrator.build_execution_mode(True),
            training_orchestrator.ExecutionMode.VALIDATE_ONLY,
        )

        progress = training_orchestrator.TrainingProgress()
        progress.record_scenario_hit(2)
        progress.record_scenario_hit(2)
        progress.record_scenario_hit(0)
        self.assertEqual(progress.scenario_hits, {2: 2, 0: 1})


if __name__ == "__main__":
    unittest.main()
