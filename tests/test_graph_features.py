from __future__ import annotations

import math
import types
import unittest
from collections import deque


try:
    import numpy as np
    import torch
    from torch_geometric.data import Data

    import graph_features
    import utils_sumo
    from rp import graph as graph_package

    GRAPH_RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    GRAPH_RUNTIME_AVAILABLE = False


class FakeNode:
    def __init__(self, node_id, *, node_type="priority"):
        self.node_id = node_id
        self.node_type = node_type
        self.incoming = []
        self.outgoing = []

    def getID(self):
        return self.node_id

    def getType(self):
        return self.node_type

    def getIncoming(self):
        return self.incoming

    def getOutgoing(self):
        return self.outgoing


class FakeLane:
    def __init__(self, lane_id):
        self.lane_id = lane_id

    def getID(self):
        return self.lane_id


class FakeEdge:
    def __init__(
        self,
        edge_id,
        from_node,
        to_node,
        *,
        length=100.0,
        speed=10.0,
        lane_count=1,
        internal=False,
    ):
        self.edge_id = edge_id
        self.from_node = from_node
        self.to_node = to_node
        self.length = length
        self.speed = speed
        self.internal = internal
        self.lanes = [FakeLane(f"{edge_id}_{index}") for index in range(lane_count)]

    def getID(self):
        return self.edge_id

    def getFromNode(self):
        return self.from_node

    def getToNode(self):
        return self.to_node

    def getLength(self):
        return self.length

    def getSpeed(self):
        return self.speed

    def getLanes(self):
        return self.lanes

    def getFunction(self):
        return "internal" if self.internal else "normal"


class FakeNet:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.edge_by_id = {edge.getID(): edge for edge in edges}

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges

    def getEdge(self, edge_id):
        return self.edge_by_id[edge_id]


def make_simple_net():
    source = FakeNode("source", node_type="traffic_light")
    target = FakeNode("target")
    edge = FakeEdge(
        "edge",
        source,
        target,
        length=120.0,
        speed=12.0,
        lane_count=2,
    )
    source.outgoing = [edge]
    target.incoming = [edge]
    return FakeNet([source, target], [edge]), edge


class FakeTrafficLightAPI:
    def __init__(self):
        self.ids = []

    def getIDList(self):
        return self.ids

    def getControlledLinks(self, _tls_id):
        return []


class FakeSimulationAPI:
    def getTime(self):
        return 25.0


class FakeEdgeAPI:
    def __init__(self, values=None, failing=None, subscriptions=None):
        self.values = values or {}
        self.failing = set(failing or [])
        self.subscriptions = subscriptions or {}

    def _value(self, edge_id, index):
        if edge_id in self.failing:
            raise FakeTraCIException(edge_id)
        return self.values[edge_id][index]

    def getTraveltime(self, edge_id):
        return self._value(edge_id, 0)

    def getLastStepMeanSpeed(self, edge_id):
        return self._value(edge_id, 1)

    def getLastStepOccupancy(self, edge_id):
        return self._value(edge_id, 2)

    def getAllSubscriptionResults(self):
        return self.subscriptions


class FakeTraCIException(Exception):
    pass


def make_traci(*, values=None, failing=None, subscriptions=None):
    return types.SimpleNamespace(
        TraCIException=FakeTraCIException,
        trafficlight=FakeTrafficLightAPI(),
        simulation=FakeSimulationAPI(),
        edge=FakeEdgeAPI(values, failing, subscriptions),
    )


class FakeEMA:
    def __init__(self, offset):
        self.offset = offset
        self.calls = []

    def update(self, edge_id, value, current_time):
        self.calls.append((edge_id, value, current_time))
        return value + self.offset


if GRAPH_RUNTIME_AVAILABLE:

    class DynamicNodeFeatureCharacterizationTests(unittest.TestCase):
        def test_utils_sumo_reexports_pure_graph_helpers(self):
            self.assertIs(
                utils_sumo.dynamic_node_feature_update,
                graph_features.dynamic_node_feature_update,
            )
            self.assertIs(
                utils_sumo.get_edge_index_v2,
                graph_features.get_edge_index_v2,
            )
            self.assertIs(
                graph_features.dynamic_node_feature_update,
                graph_package.dynamic_node_feature_update,
            )
            self.assertIs(
                graph_features.get_edge_index_v2,
                graph_package.get_edge_index_v2,
            )

        def test_undefined_tls_node_feature_api_fails_explicitly(self):
            self.assertNotIn("tls_node_features", graph_package.__all__)
            with self.assertRaisesRegex(NotImplementedError, "no feature schema"):
                graph_features.tls_node_features("node")
            with self.assertRaisesRegex(NotImplementedError, "no feature schema"):
                utils_sumo.tls_node_features("node")

        def test_dynamic_node_columns_and_state_mutations_are_stable(self):
            data = Data(
                x=torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                ),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                edge_attr=torch.ones((2, 3), dtype=torch.float32),
            )
            ever_visit = torch.zeros((3, 1), dtype=torch.float32)
            visit_age = torch.tensor([[0.5], [0.2], [0.1]], dtype=torch.float32)

            updated, returned_ever, returned_age = utils_sumo.dynamic_node_feature_update(
                data,
                target=1,
                recent_node=deque([0, 2]),
                ever_visit=ever_visit,
                visit_age=visit_age,
            )

            self.assertEqual(tuple(updated.x.shape), (3, 7))
            torch.testing.assert_close(updated.x[:, 3], torch.tensor([0.0, 1.0, 0.0]))
            torch.testing.assert_close(updated.x[:, 4], torch.tensor([0.0, 0.0, 1.0]))
            torch.testing.assert_close(updated.x[:, 5], torch.tensor([1.0, 0.0, 1.0]))
            torch.testing.assert_close(updated.x[:, 6], torch.tensor([0.475, 0.19, 1.0]))
            self.assertIs(returned_ever, ever_visit)
            self.assertIs(returned_age, visit_age)

        def test_empty_recent_history_has_explicit_initial_features(self):
            data = Data(
                x=torch.zeros((2, 3), dtype=torch.float32),
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                edge_attr=torch.ones((1, 3), dtype=torch.float32),
            )
            ever_visit = torch.zeros((2, 1), dtype=torch.float32)
            visit_age = torch.tensor([[0.5], [0.2]], dtype=torch.float32)

            updated, returned_ever, returned_age = (
                utils_sumo.dynamic_node_feature_update(
                    data,
                    target=1,
                    recent_node=deque(),
                    ever_visit=ever_visit,
                    visit_age=visit_age,
                )
            )

            torch.testing.assert_close(updated.x[:, 3], torch.tensor([0.0, 1.0]))
            torch.testing.assert_close(updated.x[:, 4], torch.zeros(2))
            torch.testing.assert_close(updated.x[:, 5], torch.zeros(2))
            torch.testing.assert_close(updated.x[:, 6], torch.tensor([0.475, 0.19]))
            self.assertIs(returned_ever, ever_visit)
            self.assertIs(returned_age, visit_age)


    class StaticGraphCharacterizationTests(unittest.TestCase):
        def test_graph_schema_node_features_and_adjacency_are_stable(self):
            net, _edge = make_simple_net()

            data, edge_ids, node_index, adjacency, edge_lengths = (
                utils_sumo.get_edge_index_v2(net)
            )

            self.assertEqual(edge_ids, ["edge"])
            self.assertEqual(node_index, {"source": 0, "target": 1})
            self.assertEqual(dict(adjacency), {"source": [("target", "edge")]})
            self.assertEqual(edge_lengths, {"edge": 120.0})
            torch.testing.assert_close(
                data.x,
                torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            )
            torch.testing.assert_close(data.edge_index, torch.tensor([[0], [1]]))
            torch.testing.assert_close(
                data.edge_attr,
                torch.tensor([[120.0, 12.0, 2.0]]),
            )

        def test_filtered_nodes_and_edges_share_one_aligned_edge_set(self):
            source = FakeNode("source")
            target = FakeNode("target")
            internal_node = FakeNode(":internal")
            kept = FakeEdge("kept", source, target, length=12.0)
            filtered = FakeEdge("filtered", source, internal_node, length=99.0)
            source.outgoing = [kept, filtered]
            target.incoming = [kept]
            internal_node.incoming = [filtered]
            net = FakeNet([source, target, internal_node], [kept, filtered])

            data, edge_ids, node_index, adjacency, edge_lengths = (
                utils_sumo.get_edge_index_v2(net)
            )

            self.assertEqual(node_index, {"source": 0, "target": 1})
            self.assertEqual(edge_ids, ["kept"])
            self.assertEqual(dict(adjacency), {"source": [("target", "kept")]})
            self.assertEqual(edge_lengths, {"kept": 12.0})
            self.assertEqual(data.edge_index.shape[1], len(edge_ids))
            self.assertEqual(data.edge_attr.shape[0], len(edge_ids))

        def test_prepare_static_keeps_five_normalized_static_edge_columns(self):
            net, _edge = make_simple_net()
            fake_traci = make_traci()
            previous = utils_sumo.traci
            utils_sumo.traci = fake_traci
            try:
                static_pack = utils_sumo.prepare_static(net)
            finally:
                utils_sumo.traci = previous

            self.assertEqual(static_pack["edge_ids"], ["edge"])
            self.assertEqual(static_pack["eid2eidx"], {"edge": 0})
            self.assertEqual(tuple(static_pack["static_edge_attr_norm"].shape), (1, 5))
            torch.testing.assert_close(
                static_pack["static_edge_attr_norm"],
                torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]),
            )
            self.assertEqual(
                {key: tuple(float(v) for v in value) for key, value in static_pack["edge_mm"].items()},
                {
                    "length": (120.0, 120.0),
                    "speed": (12.0, 12.0),
                    "lanes": (2.0, 2.0),
                },
            )


    class RuntimeEdgeFeatureCharacterizationTests(unittest.TestCase):
        def test_subscription_features_keep_speed_occupancy_vehicle_order(self):
            constants = utils_sumo.tc
            fake_traci = make_traci(
                subscriptions={
                    "edge": {
                        constants.LAST_STEP_MEAN_SPEED: 8.0,
                        constants.LAST_STEP_OCCUPANCY: 0.25,
                        constants.LAST_STEP_VEHICLE_NUMBER: 3.0,
                    }
                }
            )
            previous = utils_sumo.traci
            utils_sumo.traci = fake_traci
            try:
                result = utils_sumo.build_dynamic_features(["edge", "missing"])
            finally:
                utils_sumo.traci = previous

            torch.testing.assert_close(
                result,
                torch.tensor([[8.0, 0.25, 3.0], [0.0, 0.0, 0.0]]),
            )

        def test_collection_applies_ema_and_preserves_nan_on_traci_failure(self):
            fake_traci = make_traci(
                values={"ok": (10.0, 5.0, 0.25)},
                failing={"bad"},
            )
            previous = utils_sumo.traci
            utils_sumo.traci = fake_traci
            travel_time = FakeEMA(1.0)
            mean_speed = FakeEMA(2.0)
            occupancy = FakeEMA(0.1)
            try:
                result = utils_sumo.collect_dynamic_edge_traveltime(
                    ["ok", "bad"],
                    travel_time,
                    mean_speed,
                    occupancy,
                )
            finally:
                utils_sumo.traci = previous

            torch.testing.assert_close(result[0], torch.tensor([11.0, 7.0, 0.35]))
            self.assertTrue(torch.isnan(result[1]).all())
            self.assertEqual(travel_time.calls, [("ok", 10.0, 25.0)])

        def test_runtime_modes_keep_fixed_eight_column_schema(self):
            data_static = Data(
                x=torch.tensor([[1.0, float("nan"), 3.0]]),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                edge_attr=torch.zeros((1, 3)),
            )
            static_pack = {
                "data_static": data_static,
                "edge_ids": ["edge"],
                "eid2eidx": {"edge": 0},
                "static_edge_attr_norm": torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            }
            fake_traci = make_traci(values={"edge": (99.0, 15.0, 0.4)})
            previous = utils_sumo.traci
            utils_sumo.traci = fake_traci
            try:
                results = {
                    mode: utils_sumo.build_data_runtime(
                        static_pack,
                        FakeEMA(0.0),
                        FakeEMA(0.0),
                        FakeEMA(0.0),
                        edge_feature_mode=mode,
                    )
                    for mode in (
                        "all",
                        "static_only",
                        "dynamic_only",
                        "travel_time_only",
                        "avg_speed_only",
                        "occupancy_only",
                        "static_travel_time",
                        "static_occupancy",
                        "static_travel_time_occupancy",
                        "none",
                    )
                }
            finally:
                utils_sumo.traci = previous

            for result in results.values():
                self.assertEqual(tuple(result.edge_attr.shape), (1, 8))
                torch.testing.assert_close(result.x, torch.tensor([[1.0, 0.0, 3.0]]))

            all_features = results["all"].edge_attr[0]
            self.assertAlmostEqual(float(all_features[0]), math.log1p(99.0) / math.log1p(600.0))
            torch.testing.assert_close(all_features[1:], torch.tensor([0.5, 0.4, 0.1, 0.2, 0.3, 0.4, 0.5]))
            torch.testing.assert_close(results["none"].edge_attr, torch.zeros((1, 8)))
            self.assertEqual(
                torch.nonzero(results["occupancy_only"].edge_attr[0]).flatten().tolist(),
                [2],
            )
            self.assertEqual(
                torch.nonzero(results["static_occupancy"].edge_attr[0]).flatten().tolist(),
                [2, 3, 4, 5, 6, 7],
            )

        def test_unknown_runtime_mode_is_rejected(self):
            static_pack = {
                "data_static": Data(
                    x=torch.zeros((1, 3)),
                    edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                    edge_attr=torch.zeros((1, 3)),
                ),
                "edge_ids": ["edge"],
                "static_edge_attr_norm": torch.zeros((1, 5)),
            }

            with self.assertRaisesRegex(ValueError, "Unknown edge_feature_mode: invalid"):
                utils_sumo.build_data_runtime(
                    static_pack,
                    FakeEMA(0.0),
                    FakeEMA(0.0),
                    FakeEMA(0.0),
                    edge_feature_mode="invalid",
                )


if __name__ == "__main__":
    unittest.main()
