from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import route_file_utils


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


class RouteFileUtilityTests(unittest.TestCase):
    def write_routes(self, directory: Path) -> Path:
        route_file = directory / "input.rou.xml"
        route_file.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vehicle id="EV_0"><route edges="edge-a edge-b edge-c"/></vehicle>
    <vehicle id="EV_extra"><route edges="edge-d"/></vehicle>
    <vehicle id="car_0"><route edges="edge-e edge-f"/></vehicle>
    <vehicle id="EV_no_route"/>
    <vehicle id="EV_empty"><route edges=""/></vehicle>
</routes>
""",
            encoding="utf-8",
        )
        return route_file

    def test_count_evs_uses_prefix_and_direct_vehicle_elements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            route_file = self.write_routes(Path(tmp_dir))

            self.assertEqual(route_file_utils.count_evs_in_rou(str(route_file)), 4)
            self.assertEqual(
                route_file_utils.count_evs_in_rou(str(route_file), prefix="car"),
                1,
            )

    def test_split_edges_discards_extra_whitespace(self) -> None:
        self.assertEqual(
            route_file_utils._split_edges("  edge-a\t edge-b\nedge-c  "),
            ["edge-a", "edge-b", "edge-c"],
        )

    def test_change_route_removes_only_target_first_edge_and_writes_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            route_file = self.write_routes(directory)
            output_file = directory / "changed.rou.xml"

            changed = route_file_utils.change_route(
                str(route_file),
                str(output_file),
                target_vid="EV_0",
                min_edges_left=1,
            )

            self.assertTrue(changed)
            output_root = ET.parse(output_file).getroot()
            output_routes = {
                vehicle.get("id"): vehicle.find("route").get("edges")
                for vehicle in output_root.findall("vehicle")
                if vehicle.find("route") is not None
            }
            self.assertEqual(output_routes["EV_0"], "edge-b edge-c")
            self.assertEqual(output_routes["car_0"], "edge-e edge-f")

            original_root = ET.parse(route_file).getroot()
            original_ev_route = original_root.find("vehicle[@id='EV_0']/route")
            self.assertIsNotNone(original_ev_route)
            self.assertEqual(original_ev_route.get("edges"), "edge-a edge-b edge-c")

    def test_change_route_returns_false_without_creating_output_when_too_short(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            route_file = self.write_routes(directory)
            output_file = directory / "unchanged.rou.xml"

            changed = route_file_utils.change_route(
                str(route_file),
                str(output_file),
                target_vid="EV_extra",
                min_edges_left=1,
            )

            self.assertFalse(changed)
            self.assertFalse(output_file.exists())

    def test_keep_first_edge_reports_modified_and_skipped_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            route_file = self.write_routes(directory)
            output_file = directory / "first-edge.rou.xml"

            modified, skipped = route_file_utils.keep_first_edge_in_rou_file(
                str(route_file),
                str(output_file),
            )

            self.assertEqual((modified, skipped), (3, 2))
            output_root = ET.parse(output_file).getroot()
            output_routes = {
                vehicle.get("id"): vehicle.find("route").get("edges")
                for vehicle in output_root.findall("vehicle")
                if vehicle.find("route") is not None
            }
            self.assertEqual(output_routes["EV_0"], "edge-a")
            self.assertEqual(output_routes["EV_extra"], "edge-d")
            self.assertEqual(output_routes["car_0"], "edge-e")


class UtilsSumoCompatibilityTests(unittest.TestCase):
    def load_utils_sumo_with_dependency_stubs(self):
        fake_traci = types.ModuleType("traci")
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
        module_name = "_route_file_compatibility_utils_sumo"
        previous_loaded = sys.modules.get(module_name, _MISSING)
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
            if previous_loaded is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_loaded
            for name, old_module in previous.items():
                if old_module is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

    def test_utils_sumo_reexports_route_file_api(self) -> None:
        utils_sumo = self.load_utils_sumo_with_dependency_stubs()

        self.assertIs(utils_sumo.count_evs_in_rou, route_file_utils.count_evs_in_rou)
        self.assertIs(utils_sumo._split_edges, route_file_utils._split_edges)
        self.assertIs(utils_sumo.change_route, route_file_utils.change_route)
        self.assertIs(
            utils_sumo.keep_first_edge_in_rou_file,
            route_file_utils.keep_first_edge_in_rou_file,
        )
        self.assertIs(
            utils_sumo.batch_keep_first_edge,
            route_file_utils.batch_keep_first_edge,
        )
        self.assertIs(
            utils_sumo.save_routes_xml,
            route_file_utils.save_routes_xml,
        )


if __name__ == "__main__":
    unittest.main()
