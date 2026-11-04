from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


class FakeVehicleAPI:
    def __init__(self, *, road_id, route, route_index, lane_id, lane_position):
        self.road_id = road_id
        self.route = tuple(route)
        self.route_index = route_index
        self.lane_id = lane_id
        self.lane_position = lane_position

    def getRoadID(self, _vehicle_id):
        return self.road_id

    def getRoute(self, _vehicle_id):
        return self.route

    def getRouteIndex(self, _vehicle_id):
        return self.route_index

    def getLaneID(self, _vehicle_id):
        return self.lane_id

    def getLanePosition(self, _vehicle_id):
        return self.lane_position


class FakeLaneAPI:
    def __init__(self, lengths):
        self.lengths = lengths

    def getLength(self, lane_id):
        return self.lengths[lane_id]


def make_traci(*, road_id, route, route_index, lane_id, lane_position, lane_lengths):
    fake = types.ModuleType("traci")
    fake.constants = types.SimpleNamespace()
    fake.vehicle = FakeVehicleAPI(
        road_id=road_id,
        route=route,
        route_index=route_index,
        lane_id=lane_id,
        lane_position=lane_position,
    )
    fake.lane = FakeLaneAPI(lane_lengths)
    return fake


def load_utils_sumo_with_traci(fake_traci):
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
    module_name = f"_traci_distance_characterization_{id(fake_traci)}"
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

    def add(self, edge_id, *, length, internal=False, lane_length=None):
        edge = FakeEdge(edge_id, length=length, internal=internal)
        lane = FakeLane(
            f"{edge_id}_0",
            edge,
            length=lane_length,
        )
        edge.lanes.append(lane)
        self.edges[edge_id] = edge
        self.lanes[lane.getID()] = lane
        return edge, lane

    def getEdge(self, edge_id):
        return self.edges[edge_id]

    def getLane(self, lane_id):
        return self.lanes[lane_id]


def build_route_net():
    net = FakeNet()
    _a, lane_a = net.add("A", length=100.0)
    _b, lane_b = net.add("B", length=50.0)
    _c, lane_c = net.add("C", length=30.0)
    _i1, lane_i1 = net.add(":i1", length=4.0, internal=True)
    _i2, lane_i2 = net.add(":i2", length=6.0, internal=True)
    _i3, lane_i3 = net.add(":i3", length=2.0, internal=True)

    lane_a.outgoing = [FakeLink(lane_i1, ":i1_0")]
    lane_i1.outgoing = [FakeLink(lane_i2, ":i2_0")]
    lane_i2.outgoing = [FakeLink(lane_b)]
    lane_b.outgoing = [FakeLink(lane_i3, ":i3_0")]
    lane_i3.outgoing = [FakeLink(lane_c)]
    return net


class RemainingRouteCharacterizationTests(unittest.TestCase):
    def test_single_edge_remaining_distance_is_clamped_at_zero(self):
        cases = (
            (8.0, 2.0),
            (10.0, 0.0),
            (12.0, 0.0),
        )

        for lane_position, expected in cases:
            with self.subTest(lane_position=lane_position):
                fake = make_traci(
                    road_id="A",
                    route=["A"],
                    route_index=0,
                    lane_id="A_0",
                    lane_position=lane_position,
                    lane_lengths={"A_0": 10.0},
                )
                legacy = load_utils_sumo_with_traci(fake)

                self.assertEqual(
                    legacy.remaining_routes_len(FakeNet(), "EV_0"), expected
                )

    def test_single_edge_route_mismatch_is_rejected(self):
        fake = make_traci(
            road_id="B",
            route=["A"],
            route_index=0,
            lane_id="B_0",
            lane_position=0.0,
            lane_lengths={"B_0": 10.0},
        )
        legacy = load_utils_sumo_with_traci(fake)

        with self.assertRaisesRegex(RuntimeError, r"route mismatch: .*cur_edge=B"):
            legacy.remaining_routes_len(FakeNet(), "EV_0")

    def test_external_edge_adds_remaining_edge_and_all_following_hops(self):
        net = build_route_net()
        fake = make_traci(
            road_id="A",
            route=["A", "B", "C"],
            route_index=0,
            lane_id="A_0",
            lane_position=20.0,
            lane_lengths={"A_0": 100.0},
        )
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(legacy.remaining_routes_len(net, "EV_0"), 172.0)

    def test_internal_edge_uses_current_lane_remainder_and_later_hops(self):
        net = build_route_net()
        fake = make_traci(
            road_id=":i1",
            route=["A", "B", "C"],
            route_index=0,
            lane_id=":i1_0",
            lane_position=1.0,
            lane_lengths={":i1_0": 4.0},
        )
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(legacy.remaining_routes_len(net, "EV_0"), 91.0)


class PositionDistanceCharacterizationTests(unittest.TestCase):
    def test_distance_to_next_external_includes_current_remainder_chain_and_exit(self):
        net = FakeNet()
        _start, start_lane = net.add("start", length=20.0)
        _target, target_lane = net.add("target", length=11.0)
        _internal, internal_lane = net.add(":internal", length=4.0, internal=True)
        start_lane.outgoing = [FakeLink(internal_lane, ":internal_0")]
        internal_lane.outgoing = [FakeLink(target_lane)]
        fake = make_traci(
            road_id="start",
            route=["start", "target"],
            route_index=0,
            lane_id="start_0",
            lane_position=5.0,
            lane_lengths={"start_0": 20.0},
        )
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(
            legacy.distance_to_next_external_on(net, "EV_0", "target"),
            (30.0, 4.0, 11.0, [":internal"], "target_0"),
        )

    def test_remaining_to_exit_handles_gateway_exit_and_internal_positions(self):
        net = build_route_net()

        gateway_traci = make_traci(
            road_id="A",
            route=["A", "B"],
            route_index=0,
            lane_id="A_0",
            lane_position=20.0,
            lane_lengths={"A_0": 100.0},
        )
        gateway = load_utils_sumo_with_traci(gateway_traci)
        self.assertEqual(
            gateway.remaining_to_exit_external_end(
                net,
                "EV_0",
                [":i1", ":i2"],
                "B_0",
            ),
            140.0,
        )

        exit_traci = make_traci(
            road_id="B",
            route=["B"],
            route_index=0,
            lane_id="B_0",
            lane_position=15.0,
            lane_lengths={"B_0": 50.0},
        )
        at_exit = load_utils_sumo_with_traci(exit_traci)
        self.assertEqual(
            at_exit.remaining_to_exit_external_end(
                net,
                "EV_0",
                [":i1", ":i2"],
                "B_0",
            ),
            35.0,
        )

        internal_traci = make_traci(
            road_id=":i1",
            route=["A", "B"],
            route_index=0,
            lane_id=":i1_0",
            lane_position=1.0,
            lane_lengths={":i1_0": 4.0},
        )
        internal = load_utils_sumo_with_traci(internal_traci)
        self.assertEqual(
            internal.remaining_to_exit_external_end(
                net,
                "EV_0",
                [":i1", ":i2"],
                "B_0",
            ),
            59.0,
        )


if __name__ == "__main__":
    unittest.main()
