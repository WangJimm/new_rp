from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


class FakeSimulationAPI:
    def __init__(self):
        self.steps = 0

    def getMinExpectedNumber(self):
        return 0

    def getDepartedIDList(self):
        return []

    def getTime(self):
        return float(self.steps)

    def getArrivedIDList(self):
        return []


class FakeTraCI(types.ModuleType):
    def __init__(self):
        super().__init__("traci")
        self.constants = types.SimpleNamespace()
        self.simulation = FakeSimulationAPI()
        self.load_calls = []

    def load(self, arguments):
        self.load_calls.append(list(arguments))

    def simulationStep(self):
        self.simulation.steps += 1


def load_utils_sumo_with_dependency_stubs(fake_traci):
    fake_torch = types.ModuleType("torch")
    fake_torch.zeros = lambda *args, **kwargs: (args, kwargs)
    fake_torch_geometric = types.ModuleType("torch_geometric")
    fake_torch_geometric_data = types.ModuleType("torch_geometric.data")
    fake_torch_geometric_data.Data = object
    fake_torch_geometric.data = fake_torch_geometric_data
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")
    fake_matplotlib.pyplot = fake_pyplot

    injected = {
        "traci": fake_traci,
        "torch": fake_torch,
        "torch_geometric": fake_torch_geometric,
        "torch_geometric.data": fake_torch_geometric_data,
        "matplotlib": fake_matplotlib,
        "matplotlib.pyplot": fake_pyplot,
    }
    previous = {name: sys.modules.get(name, _MISSING) for name in injected}
    module_name = f"_legacy_dijkstra_characterization_{id(fake_traci)}"
    try:
        sys.modules.update(injected)
        spec = importlib.util.spec_from_file_location(
            module_name,
            PROJECT_ROOT / "utils_sumo.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load utils_sumo.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        for name, old_module in previous.items():
            if old_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class LegacyDijkstraCharacterizationTests(unittest.TestCase):
    def test_zero_step_scenario_keeps_load_arguments_and_counted_ev_paths(self):
        fake_traci = FakeTraCI()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)
        static_pack = {
            "edge_ids": ["edge"],
            "nid2idx": {"node": 0},
            "eid2eidx": {"edge": 0},
            "adj": {},
        }
        testing_paths = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            goal_directory = root / "scenario" / "testing_data_goal"
            goal_directory.mkdir(parents=True)
            goal_file = goal_directory / "goal_ev_free_0900_p01_7.json"
            goal_file.write_text(json.dumps(["goal-node"]), encoding="utf-8")
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(legacy, "count_evs_in_rou", return_value=1):
                    result = legacy.dijkstra_sim(
                        net=object(),
                        static_pack=static_pack,
                        ev_ids=["ignored-by-legacy-code"],
                        filesource=Path("base-env"),
                        rd3=7,
                        rd4=1,
                        rd5=0,
                        testing_paths=testing_paths,
                        using_env="scenario",
                        ev_route="custom.rou.xml",
                    )
            finally:
                os.chdir(previous_directory)

        self.assertEqual(result, 0)
        self.assertEqual(testing_paths, [[]])
        self.assertEqual(
            fake_traci.load_calls,
            [
                [
                    "-c",
                    str(Path("base-env") / "osm.sumocfg"),
                    "--step-length",
                    "0.5",
                    "--route-files",
                    f"{Path('base-env') / 'testing_data/output_free_0900_7.rou.xml'}, custom.rou.xml",
                    "--no-warnings",
                ]
            ],
        )

    def test_goal_count_must_match_counted_evs_exactly(self):
        fake_traci = FakeTraCI()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)
        static_pack = {"nid2idx": {}, "adj": {}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            goal_directory = root / "scenario" / "testing_data_goal"
            goal_directory.mkdir(parents=True)
            goal_file = goal_directory / "goal_ev_free_0900_p01_7.json"
            goal_file.write_text(json.dumps(["only-one-goal"]), encoding="utf-8")
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(legacy, "count_evs_in_rou", return_value=2):
                    with self.assertRaisesRegex(ValueError, "goals=1, evs=2"):
                        legacy.dijkstra_sim(
                            net=object(),
                            static_pack=static_pack,
                            ev_ids=None,
                            filesource=Path("base-env"),
                            rd3=7,
                            rd4=1,
                            rd5=0,
                            using_env="scenario",
                            ev_route="custom.rou.xml",
                        )
            finally:
                os.chdir(previous_directory)

        self.assertEqual(fake_traci.load_calls, [])

    def test_arrival_without_route_decision_has_zero_measured_time(self):
        fake_traci = FakeTraCI()
        fake_traci.simulation.getMinExpectedNumber = lambda: 1
        fake_traci.simulation.getArrivedIDList = lambda: ["EV_0"]
        fake_traci.vehicle = types.SimpleNamespace(getIDList=lambda: [])
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)
        static_pack = {"nid2idx": {}, "adj": {}}
        testing_paths = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            goal_directory = root / "scenario" / "testing_data_goal"
            goal_directory.mkdir(parents=True)
            (goal_directory / "goal_ev_free_0900_p01_7.json").write_text(
                json.dumps(["goal"]), encoding="utf-8"
            )
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(legacy, "count_evs_in_rou", return_value=1):
                    result = legacy.dijkstra_sim(
                        net=object(),
                        static_pack=static_pack,
                        ev_ids=None,
                        filesource=Path("base-env"),
                        rd3=7,
                        rd4=1,
                        rd5=0,
                        testing_paths=testing_paths,
                        using_env="scenario",
                        ev_route="custom.rou.xml",
                    )
            finally:
                os.chdir(previous_directory)

        self.assertEqual(result, 0.0)
        self.assertEqual(testing_paths, [[]])

    def test_simulation_stops_at_the_configured_step_limit(self):
        fake_traci = FakeTraCI()
        fake_traci.simulation.getMinExpectedNumber = lambda: 1
        fake_traci.vehicle = types.SimpleNamespace(getIDList=lambda: [])
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)
        static_pack = {"nid2idx": {}, "adj": {}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            goal_directory = root / "scenario" / "testing_data_goal"
            goal_directory.mkdir(parents=True)
            (goal_directory / "goal_ev_free_0900_p01_7.json").write_text(
                json.dumps(["goal"]), encoding="utf-8"
            )
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(legacy, "count_evs_in_rou", return_value=1):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "limit=2.*unfinished=\\['EV_0'\\]",
                    ):
                        legacy.dijkstra_sim(
                            net=object(),
                            static_pack=static_pack,
                            ev_ids=None,
                            filesource=Path("base-env"),
                            rd3=7,
                            rd4=1,
                            rd5=0,
                            using_env="scenario",
                            ev_route="custom.rou.xml",
                            max_simulation_steps=2,
                        )
            finally:
                os.chdir(previous_directory)

        self.assertEqual(fake_traci.simulation.steps, 2)

    def test_traffic_jam_speeds_are_restored_after_an_exception(self):
        fake_traci = FakeTraCI()
        fake_traci.simulation.getMinExpectedNumber = lambda: 1
        fake_traci.simulation.getTime = lambda: 120.0
        fake_traci.edge = types.SimpleNamespace(getLaneNumber=lambda _edge: 1)
        speed_calls = []
        fake_traci.lane = types.SimpleNamespace(
            getMaxSpeed=lambda _lane: 10.0,
            setMaxSpeed=lambda lane, speed: speed_calls.append((lane, speed)),
        )

        def fail_during_vehicle_query():
            raise RuntimeError("forced vehicle query failure")

        fake_traci.vehicle = types.SimpleNamespace(
            getIDList=fail_during_vehicle_query
        )
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)
        static_pack = {"nid2idx": {}, "adj": {}}

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            goal_directory = root / "scenario" / "testing_data_goal"
            goal_directory.mkdir(parents=True)
            (goal_directory / "goal_ev_free_0900_p01_7.json").write_text(
                json.dumps(["goal"]), encoding="utf-8"
            )
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(legacy, "count_evs_in_rou", return_value=1):
                    with self.assertRaisesRegex(
                        RuntimeError, "forced vehicle query failure"
                    ):
                        legacy.dijkstra_sim(
                            net=object(),
                            static_pack=static_pack,
                            ev_ids=None,
                            filesource=Path("base-env"),
                            rd3=7,
                            rd4=1,
                            rd5=0,
                            using_env="scenario",
                            ev_route="custom.rou.xml",
                            traffic_jam=True,
                        )
            finally:
                os.chdir(previous_directory)

        self.assertEqual(
            speed_calls,
            [
                ("5707608#2_0", 3.0),
                ("5707608#0_0", 3.0),
                ("5707608#2_0", 10.0),
                ("5707608#0_0", 10.0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
