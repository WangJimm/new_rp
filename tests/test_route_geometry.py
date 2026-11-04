from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import route_geometry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def load_utils_sumo_with_dependency_stubs():
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
    module_name = "_route_geometry_characterization_utils_sumo"
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


utils_sumo = load_utils_sumo_with_dependency_stubs()


class CompatibilityReexportTests(unittest.TestCase):
    def test_utils_sumo_reexports_route_geometry_helpers(self):
        for name in route_geometry.__all__:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(utils_sumo, name),
                    getattr(route_geometry, name),
                )


class FakeEdge:
    def __init__(self, edge_id, *, length=10.0, internal=False):
        self.edge_id = edge_id
        self.length = length
        self.internal = internal
        self.lanes = []

    def getID(self):
        return self.edge_id

    def getLength(self):
        return self.length

    def getFunction(self):
        return "internal" if self.internal else "normal"

    def getLanes(self):
        return self.lanes


class FakeLane:
    def __init__(self, lane_id, edge, *, length=None):
        self.lane_id = lane_id
        self.edge = edge
        self.length = edge.getLength() if length is None else length
        self.outgoing = []

    def getID(self):
        return self.lane_id

    def getEdge(self):
        return self.edge

    def getLength(self):
        return self.length

    def getOutgoing(self):
        return self.outgoing


class FakeLink:
    def __init__(self, to_lane, via_lane_id=None):
        self.to_lane = to_lane
        self.via_lane_id = via_lane_id

    def getToLane(self):
        return self.to_lane

    def getViaLaneID(self):
        return self.via_lane_id


class FakeNet:
    def __init__(self):
        self.edges = {}
        self.lanes = {}

    def add_edge(self, edge, *lanes):
        edge.lanes.extend(lanes)
        self.edges[edge.getID()] = edge
        for lane in lanes:
            self.lanes[lane.getID()] = lane

    def getEdge(self, edge_id):
        return self.edges[edge_id]

    def getLane(self, lane_id):
        return self.lanes[lane_id]


def add_edge(net, edge_id, *, length=10.0, internal=False, lane_count=1):
    edge = FakeEdge(edge_id, length=length, internal=internal)
    lanes = [
        FakeLane(f"{edge_id}_{index}", edge)
        for index in range(lane_count)
    ]
    net.add_edge(edge, *lanes)
    return edge, lanes


class LaneResolutionCharacterizationTests(unittest.TestCase):
    def test_lane_id_is_preferred_and_edge_id_falls_back_to_first_lane(self):
        net = FakeNet()
        edge, lanes = add_edge(net, "edge", lane_count=2)

        self.assertIs(utils_sumo._get_lane_from_id_or_edge(net, "edge_1"), lanes[1])
        self.assertIs(utils_sumo._get_lane_from_id_or_edge(net, "edge"), lanes[0])
        self.assertIsNone(utils_sumo._get_lane_from_id_or_edge(net, "missing"))
        self.assertIsNone(utils_sumo._get_lane_from_id_or_edge(net, None))

    def test_safe_get_lane_always_returns_a_lane_list(self):
        net = FakeNet()
        _edge, lanes = add_edge(net, "edge", lane_count=2)
        _underscored_edge, underscored_lanes = add_edge(
            net,
            "edge_with_underscore",
            lane_count=2,
        )

        self.assertEqual(utils_sumo._safe_get_lane(net, "edge_1"), [lanes[1]])
        self.assertEqual(utils_sumo._safe_get_lane(net, "edge"), lanes)
        self.assertEqual(
            utils_sumo._safe_get_lane(net, "edge_with_underscore"),
            underscored_lanes,
        )
        self.assertEqual(utils_sumo._safe_get_lane(net, "missing"), [])
        self.assertEqual(utils_sumo._safe_get_lane(net, None), [])


class InternalChainCharacterizationTests(unittest.TestCase):
    def test_direct_external_link_has_empty_internal_chain(self):
        net = FakeNet()
        _target, target_lanes = add_edge(net, "target")
        link = FakeLink(target_lanes[0])

        self.assertEqual(
            utils_sumo.trace_to_target_edge(net, link, "target"),
            ([], target_lanes[0]),
        )
        self.assertIsNone(utils_sumo.trace_to_target_edge(net, link, "other"))

    def test_trace_follows_via_lanes_and_records_internal_edges(self):
        net = FakeNet()
        _target, target_lanes = add_edge(net, "target")
        _internal_a, internal_a_lanes = add_edge(
            net,
            ":a",
            length=2.0,
            internal=True,
        )
        _internal_b, internal_b_lanes = add_edge(
            net,
            ":b",
            length=3.0,
            internal=True,
        )
        internal_a_lanes[0].outgoing = [FakeLink(internal_b_lanes[0], ":b_0")]
        internal_b_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        first = FakeLink(internal_a_lanes[0], ":a_0")

        self.assertEqual(
            utils_sumo.trace_to_target_edge(net, first, "target"),
            ([":a", ":b"], target_lanes[0]),
        )

    def test_chain_selection_uses_shortest_internal_length(self):
        net = FakeNet()
        from_edge, from_lanes = add_edge(net, "from", lane_count=2)
        _target, target_lanes = add_edge(net, "target")
        _long, long_lanes = add_edge(net, ":long", length=8.0, internal=True)
        _short, short_lanes = add_edge(net, ":short", length=3.0, internal=True)
        long_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        short_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        from_lanes[0].outgoing = [FakeLink(long_lanes[0], ":long_0")]
        from_lanes[1].outgoing = [FakeLink(short_lanes[0], ":short_0")]

        self.assertEqual(
            utils_sumo.internal_chain_between_edges(net, "from", "target"),
            {
                "from_edge": "from",
                "to_edge": "target",
                "internal_edges": [":short"],
                "internal_len": 3.0,
                "exit_lane_id": "target_0",
            },
        )

    def test_route_chain_stops_with_none_when_any_hop_is_unreachable(self):
        net = FakeNet()
        _first, first_lanes = add_edge(net, "first")
        _second, second_lanes = add_edge(net, "second")
        add_edge(net, "third")
        first_lanes[0].outgoing = [FakeLink(second_lanes[0])]

        self.assertEqual(
            utils_sumo.internal_chains_for_route(net, ["first", "second"]),
            [
                {
                    "from_edge": "first",
                    "to_edge": "second",
                    "internal_edges": [],
                    "internal_len": 0,
                    "exit_lane_id": "second_0",
                }
            ],
        )
        self.assertIsNone(
            utils_sumo.internal_chains_for_route(
                net,
                ["first", "second", "third"],
            )
        )

    def test_distance_from_edge_to_target_excludes_start_lane_length(self):
        net = FakeNet()
        _start, start_lanes = add_edge(net, "start", length=20.0)
        _target, target_lanes = add_edge(net, "target", length=11.0)
        _internal, internal_lanes = add_edge(
            net,
            ":internal",
            length=4.0,
            internal=True,
        )
        internal_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        start_lanes[0].outgoing = [FakeLink(internal_lanes[0], ":internal_0")]

        self.assertEqual(
            utils_sumo.distance_from_exit_lane_to_target_edge(
                net,
                "start",
                "target",
            ),
            (15.0, 4.0, 11.0, [":internal"], "target_0"),
        )

    def test_distance_accepts_a_start_lane_id(self):
        net = FakeNet()
        _start, start_lanes = add_edge(net, "start", length=20.0)
        _target, target_lanes = add_edge(net, "target", length=11.0)
        _internal, internal_lanes = add_edge(
            net,
            ":internal",
            length=4.0,
            internal=True,
        )
        internal_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        start_lanes[0].outgoing = [FakeLink(internal_lanes[0], ":internal_0")]

        self.assertEqual(
            utils_sumo.distance_from_exit_lane_to_target_edge(
                net,
                "start_0",
                "target",
            ),
            (15.0, 4.0, 11.0, [":internal"], "target_0"),
        )

    def test_distance_keeps_shorter_path_from_an_earlier_lane(self):
        net = FakeNet()
        _start, start_lanes = add_edge(net, "start", lane_count=2)
        _target, target_lanes = add_edge(net, "target", length=11.0)
        _short, short_lanes = add_edge(net, ":short", length=2.0, internal=True)
        _long, long_lanes = add_edge(net, ":long", length=8.0, internal=True)
        short_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        long_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        start_lanes[0].outgoing = [FakeLink(short_lanes[0], ":short_0")]
        start_lanes[1].outgoing = [FakeLink(long_lanes[0], ":long_0")]

        self.assertEqual(
            utils_sumo.distance_from_exit_lane_to_target_edge(
                net, "start", "target"
            ),
            (13.0, 2.0, 11.0, [":short"], "target_0"),
        )

    def test_distance_selects_shorter_path_from_a_later_lane(self):
        net = FakeNet()
        _start, start_lanes = add_edge(net, "start", lane_count=2)
        _target, target_lanes = add_edge(net, "target", length=11.0)
        _long, long_lanes = add_edge(net, ":long", length=8.0, internal=True)
        _short, short_lanes = add_edge(net, ":short", length=2.0, internal=True)
        long_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        short_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        start_lanes[0].outgoing = [FakeLink(long_lanes[0], ":long_0")]
        start_lanes[1].outgoing = [FakeLink(short_lanes[0], ":short_0")]

        self.assertEqual(
            utils_sumo.distance_from_exit_lane_to_target_edge(
                net, "start", "target"
            ),
            (13.0, 2.0, 11.0, [":short"], "target_0"),
        )

    def test_distance_keeps_reachable_path_when_last_lane_is_unreachable(self):
        net = FakeNet()
        _start, start_lanes = add_edge(net, "start", lane_count=2)
        _target, target_lanes = add_edge(net, "target", length=11.0)
        _internal, internal_lanes = add_edge(
            net, ":internal", length=4.0, internal=True
        )
        internal_lanes[0].outgoing = [FakeLink(target_lanes[0])]
        start_lanes[0].outgoing = [FakeLink(internal_lanes[0], ":internal_0")]

        self.assertEqual(
            utils_sumo.distance_from_exit_lane_to_target_edge(
                net, "start", "target"
            ),
            (15.0, 4.0, 11.0, [":internal"], "target_0"),
        )

    def test_distance_returns_none_when_all_lanes_are_unreachable(self):
        net = FakeNet()
        add_edge(net, "start", lane_count=2)
        add_edge(net, "target")

        self.assertIsNone(
            utils_sumo.distance_from_exit_lane_to_target_edge(
                net, "start", "target"
            )
        )


if __name__ == "__main__":
    unittest.main()
