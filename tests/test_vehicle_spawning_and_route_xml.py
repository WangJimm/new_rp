from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def load_utils_sumo_with_dependency_stubs(fake_traci):
    fake_traci.constants = types.SimpleNamespace()
    fake_torch = types.ModuleType("torch")
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
    module_name = f"_vehicle_spawning_characterization_{id(fake_traci)}"
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


class FakeRouteAPI:
    def __init__(self):
        self.calls = []

    def add(self, route_id, edges):
        self.calls.append((route_id, list(edges)))


class FakeVehicleAPI:
    def __init__(self):
        self.add_calls = []
        self.speed_mode_calls = []

    def add(self, *args, **kwargs):
        self.add_calls.append((args, kwargs))

    def setSpeedMode(self, vehicle_id, mode):
        self.speed_mode_calls.append((vehicle_id, mode))


def make_traci():
    fake = types.ModuleType("traci")
    fake.route = FakeRouteAPI()
    fake.vehicle = FakeVehicleAPI()
    return fake


class RouteXMLCharacterizationTests(unittest.TestCase):
    def test_route_xml_path_attributes_departures_and_edge_formats_are_stable(self):
        fake_traci = make_traci()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "scenario" / "new_training_data_3"
            output_directory.mkdir(parents=True)
            previous_directory = Path.cwd()
            try:
                os.chdir(root)
                result = legacy.save_routes_xml(
                    [["edge-a", "edge-b"], "edge-c"],
                    "sample",
                    start_depart=13.13,
                    depart_gap=5.0,
                    vtype_id="ev",
                    using_env="scenario",
                )
            finally:
                os.chdir(previous_directory)

            output = output_directory / "sample_training_set.rou.xml"
            self.assertIsNone(result)
            self.assertTrue(output.is_file())
            tree = ET.parse(output)
            vehicles = tree.getroot().findall("vehicle")
            self.assertEqual(
                [vehicle.attrib for vehicle in vehicles],
                [
                    {
                        "id": "EV_1",
                        "depart": "13.13",
                        "departLane": "best",
                        "departSpeed": "max",
                        "type": "ev",
                    },
                    {
                        "id": "EV_2",
                        "depart": "18.13",
                        "departLane": "best",
                        "departSpeed": "max",
                        "type": "ev",
                    },
                ],
            )
            self.assertEqual(
                [vehicle.find("route").attrib["edges"] for vehicle in vehicles],
                ["edge-a edge-b", "edge-c"],
            )


class VehicleSpawningCharacterizationTests(unittest.TestCase):
    def test_spawn_for_rl_uses_temporary_route_and_forwarded_depart_options(self):
        fake_traci = make_traci()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)

        legacy.spawn_for_rl(
            "EV_7",
            "edge-a",
            vtype="ambulance",
            depart_lane="2",
            depart_pos="10",
            depart_speed="5",
        )

        self.assertEqual(fake_traci.route.calls, [("tmp_EV_7", ["edge-a"])])
        self.assertEqual(
            fake_traci.vehicle.add_calls,
            [
                (
                    ("EV_7", "tmp_EV_7"),
                    {
                        "typeID": "ambulance",
                        "departLane": "2",
                        "departPos": "10",
                        "departSpeed": "5",
                    },
                )
            ],
        )

    def test_testing_spawn_uses_ev1_keys_string_departures_and_speed_mode_32(self):
        fake_traci = make_traci()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)

        legacy.spawn_evs_testing(
            base_depart=2,
            depart_gap=2.5,
            src_edges={"EV1": "edge-a", "EV2": "edge-b"},
            goal_nodes={"EV1": "goal-a", "EV2": "goal-b"},
        )

        self.assertEqual(
            fake_traci.route.calls,
            [("route_EV1", ["edge-a"]), ("route_EV2", ["edge-b"])],
        )
        self.assertEqual(
            [call[1]["depart"] for call in fake_traci.vehicle.add_calls],
            ["2.0", "4.5"],
        )
        self.assertEqual(
            fake_traci.vehicle.speed_mode_calls,
            [("EV1", 32), ("EV2", 32)],
        )

    def test_testing_spawn_rejects_non_mapping_or_misaligned_inputs(self):
        fake_traci = make_traci()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)

        with self.assertRaisesRegex(TypeError, "EV-keyed mappings"):
            legacy.spawn_evs_testing(
                src_edges=["edge-a"],
                goal_nodes=["goal-a"],
            )
        with self.assertRaisesRegex(ValueError, "identical vehicle IDs"):
            legacy.spawn_evs_testing(
                src_edges={"EV1": "edge-a"},
                goal_nodes={"EV2": "goal-a"},
            )
        self.assertEqual(fake_traci.route.calls, [])
        self.assertEqual(fake_traci.vehicle.add_calls, [])

    def test_generated_spawn_uses_reproducible_ids_waypoints_and_saved_routes(self):
        fake_traci = make_traci()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)
        saved = []

        class FakeRandom:
            @staticmethod
            def choices(values, k):
                self = values
                return [self[0]] * k

            @staticmethod
            def choice(values):
                return values[0]

        class FakeTime:
            @staticmethod
            def time():
                raise AssertionError("wall-clock time must not affect vehicle IDs")

        def fake_pathfinder(**kwargs):
            self.assertEqual(kwargs["cur_edge"], "edge-a")
            self.assertEqual(kwargs["goal_node"], "goal")
            return ["n0", "n1", "n2", "n3", "goal"], ["a", "b", "c"], 3.0

        def fake_save_routes(routes, filename, using_env):
            saved.append((routes, filename, using_env))

        with (
            patch.object(legacy, "random", FakeRandom),
            patch.object(legacy, "time", FakeTime),
            patch.object(legacy, "dijkstra_path", fake_pathfinder),
            patch.object(legacy, "save_routes_xml", fake_save_routes),
        ):
            goals, waypoints = legacy.spawn_evs(
                ["edge-a"],
                ["goal"],
                1,
                base_depart=4,
                depart_gap=5.0,
                net=object(),
                adj={"unused": []},
                filename="sample",
                using_env="scenario",
            )

        self.assertEqual(goals, {"EV_1": "goal"})
        self.assertEqual(waypoints, {"EV_1": ["n3", "goal"]})
        self.assertEqual(
            fake_traci.route.calls,
            [("route_EV_1", ["edge-a"])],
        )
        self.assertEqual(
            fake_traci.vehicle.add_calls[0][1],
            {
                "vehID": "EV_1",
                "routeID": "route_EV_1",
                "depart": "4.0",
                "departLane": "best",
                "departSpeed": "max",
            },
        )
        self.assertEqual(fake_traci.vehicle.speed_mode_calls, [("EV_1", 32)])
        self.assertEqual(saved, [([["edge-a"]], "sample", "scenario")])

    def test_generated_spawn_stops_after_bounded_route_attempts(self):
        fake_traci = make_traci()
        legacy = load_utils_sumo_with_dependency_stubs(fake_traci)
        pathfinder_calls = []
        saved = []

        class FakeRandom:
            @staticmethod
            def choices(values, k):
                return [values[0]] * k

            @staticmethod
            def choice(values):
                return values[0]

        def unreachable_pathfinder(**kwargs):
            pathfinder_calls.append(kwargs)
            return None, None, float("inf")

        with (
            patch.object(legacy, "random", FakeRandom),
            patch.object(legacy, "dijkstra_path", unreachable_pathfinder),
            patch.object(
                legacy,
                "save_routes_xml",
                lambda *args, **kwargs: saved.append((args, kwargs)),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "vehicle_id=EV_1.*attempts=3",
            ):
                legacy.spawn_evs(
                    ["edge-a"],
                    ["goal"],
                    1,
                    net=object(),
                    adj={},
                    max_source_attempts=3,
                )

        self.assertEqual(len(pathfinder_calls), 3)
        self.assertEqual(fake_traci.route.calls, [])
        self.assertEqual(fake_traci.vehicle.add_calls, [])
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
