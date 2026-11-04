from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


class FakeEdgeAPI:
    def __init__(self, lane_counts=None, travel_times=None):
        self.lane_counts = lane_counts or {}
        self.travel_times = travel_times or {}

    def getLaneNumber(self, edge_id):
        return self.lane_counts[edge_id]

    def getTraveltime(self, edge_id):
        return self.travel_times[edge_id]


class FakeLaneAPI:
    def __init__(self, links=None):
        self.links = links or {}

    def getLinks(self, lane_id, extended=False):
        if not extended:
            raise AssertionError("legacy code requests extended links")
        return self.links.get(lane_id, [])


class FakeVehicleAPI:
    def __init__(self):
        self.road_id = "current"
        self.route = ("past", "current", "future")
        self.route_index = 1
        self.set_route_calls = []
        self.set_route_error = None

    def getRoadID(self, _vehicle_id):
        return self.road_id

    def getRoute(self, _vehicle_id):
        return self.route

    def getRouteIndex(self, _vehicle_id):
        return self.route_index

    def setRoute(self, vehicle_id, route):
        if self.set_route_error is not None:
            raise self.set_route_error
        self.set_route_calls.append((vehicle_id, list(route)))


def load_utils_sumo_with_traci(fake_traci):
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
    module_name = f"_traci_routing_characterization_{id(fake_traci)}"
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


class FakeNode:
    def __init__(self, node_id, outgoing=None):
        self.node_id = node_id
        self.outgoing = list(outgoing or [])

    def getID(self):
        return self.node_id

    def getOutgoing(self):
        return self.outgoing


class FakeNetEdge:
    def __init__(self, edge_id, target, function="normal"):
        self.edge_id = edge_id
        self.target = target
        self.function = function

    def getID(self):
        return self.edge_id

    def getToNode(self):
        return self.target

    def getFunction(self):
        return self.function


class FakeNet:
    def __init__(self, current_edge):
        self.current_edge = current_edge

    def getEdge(self, _edge_id):
        return self.current_edge


def make_traci(*, lane_counts=None, links=None, travel_times=None):
    fake = types.ModuleType("traci")
    fake.edge = FakeEdgeAPI(lane_counts, travel_times)
    fake.lane = FakeLaneAPI(links)
    fake.vehicle = FakeVehicleAPI()
    return fake


class ReachabilityCharacterizationTests(unittest.TestCase):
    def test_reachable_edges_uses_every_lane_and_current_id_parsing(self):
        fake = make_traci(
            lane_counts={"current": 2},
            links={
                "current_0": [("east_0",)],
                "current_1": [("north_0",), ("east_1",)],
            },
        )
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(legacy.reachable_edges_from("current"), {"east", "north"})

    def test_reachable_edges_preserves_underscores_in_edge_ids(self):
        fake = make_traci(
            lane_counts={"current": 1},
            links={"current_0": [("road_segment_0",), ("road_segment_1",)]},
        )
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(
            legacy.reachable_edges_from("current"),
            {"road_segment"},
        )

    def test_candidate_junctions_filter_unreachable_and_internal_edges(self):
        target_b = FakeNode("junction-b")
        target_a = FakeNode("junction-a")
        reachable_b = FakeNetEdge("edge-b", target_b)
        unreachable = FakeNetEdge("edge-x", target_a)
        internal = FakeNetEdge(":internal", target_a, function="internal")
        reachable_a = FakeNetEdge("edge-a", target_a)
        current_node = FakeNode(
            "current-target",
            [reachable_b, unreachable, internal, reachable_a],
        )
        current_edge = FakeNetEdge("current", current_node)
        fake = make_traci(
            lane_counts={"current": 1},
            links={"current_0": [("edge-b_0",), ("edge-a_0",), (":internal_0",)]},
        )
        legacy = load_utils_sumo_with_traci(fake)

        junctions, outgoing = legacy.candidate_next_junctions(
            FakeNet(current_edge),
            "current",
        )

        self.assertEqual(junctions, ["junction-a", "junction-b"])
        self.assertEqual(outgoing, [reachable_b, reachable_a])


class RouteApplicationCharacterizationTests(unittest.TestCase):
    def test_apply_next_junction_picks_first_minimum_and_appends_to_suffix(self):
        target = FakeNode("chosen")
        first = FakeNetEdge("first", target)
        tied = FakeNetEdge("tied", target)
        other = FakeNetEdge("other", FakeNode("other-target"))
        fake = make_traci(travel_times={"first": 4.0, "tied": 4.0})
        legacy = load_utils_sumo_with_traci(fake)

        selected = legacy.apply_next_junction(
            "ev1",
            "chosen",
            [other, first, tied],
        )

        self.assertEqual(selected, "first")
        self.assertEqual(
            fake.vehicle.set_route_calls,
            [("ev1", ["current", "future", "first"])],
        )

    def test_missing_choice_reports_route_and_candidates(self):
        candidate = FakeNetEdge("edge-a", FakeNode("junction-a"))
        fake = make_traci(travel_times={})
        legacy = load_utils_sumo_with_traci(fake)

        with self.assertRaisesRegex(
            RuntimeError,
            r"could not resolve outgoing edge: .*chosen_jn=missing.*edge-a",
        ):
            legacy.apply_next_junction("ev1", "missing", [candidate])

        self.assertEqual(fake.vehicle.set_route_calls, [])

    def test_invalid_route_index_is_rejected_before_setting_route(self):
        candidate = FakeNetEdge("edge-a", FakeNode("chosen"))
        fake = make_traci(travel_times={"edge-a": 1.0})
        fake.vehicle.route_index = -1
        legacy = load_utils_sumo_with_traci(fake)

        with self.assertRaisesRegex(RuntimeError, r"invalid route index: .*route_index=-1"):
            legacy.apply_next_junction("ev1", "chosen", [candidate])

        self.assertEqual(fake.vehicle.set_route_calls, [])

    def test_set_route_failure_is_wrapped_with_new_route_context(self):
        candidate = FakeNetEdge("edge-a", FakeNode("chosen"))
        fake = make_traci(travel_times={"edge-a": 1.0})
        fake.vehicle.set_route_error = ValueError("disconnected")
        legacy = load_utils_sumo_with_traci(fake)

        with self.assertRaisesRegex(
            RuntimeError,
            r"failed to set SUMO route: .*new_route=\['current', 'future', 'edge-a'\]",
        ) as caught:
            legacy.apply_next_junction("ev1", "chosen", [candidate])

        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_apply_full_route_honors_optional_suffix_limit(self):
        fake = make_traci()
        legacy = load_utils_sumo_with_traci(fake)

        legacy.apply_full_route("ev1", "current", ["a", "b", "c"], suffix_keep=2)
        legacy.apply_full_route("ev2", "start", ["x", "y"], suffix_keep=None)

        self.assertEqual(
            fake.vehicle.set_route_calls,
            [
                ("ev1", ["current", "a", "b"]),
                ("ev2", ["start", "x", "y"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
