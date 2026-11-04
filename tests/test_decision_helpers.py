from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from rp.sumo import decisions as decision_helpers


class FakeNode:
    def __init__(self, node_id):
        self.node_id = node_id

    def getID(self):
        return self.node_id


class FakeEdge:
    def __init__(self, *, function="normal", source="source", target="target"):
        self.function = function
        self.source = FakeNode(source)
        self.target = FakeNode(target)

    def getFunction(self):
        return self.function

    def getFromNode(self):
        return self.source

    def getToNode(self):
        return self.target


class FakeNet:
    def __init__(self, edge):
        self.edge = edge

    def getEdge(self, _edge_id):
        return self.edge


class FakeVehicleAPI:
    def __init__(self, road_id, *, lane_id="lane", lane_position=0.0):
        self.road_id = road_id
        self.lane_id = lane_id
        self.lane_position = lane_position

    def getRoadID(self, _vehicle_id):
        return self.road_id

    def getLaneID(self, _vehicle_id):
        return self.lane_id

    def getLanePosition(self, _vehicle_id):
        return self.lane_position


def make_traci(road_id, *, lane_length=100.0, lane_position=0.0):
    return types.SimpleNamespace(
        vehicle=FakeVehicleAPI(road_id, lane_position=lane_position),
        lane=types.SimpleNamespace(getLength=lambda _lane_id: lane_length),
    )


class GoalJunctionCharacterizationTests(unittest.TestCase):
    def test_vehicle_not_on_a_normal_edge_has_not_reached_goal(self):
        net = FakeNet(FakeEdge())
        for road_id in ("", "!pending"):
            with self.subTest(road_id=road_id):
                self.assertFalse(
                    decision_helpers.reached_goal_junction(
                        "ev",
                        "target",
                        net,
                        traci_api=make_traci(road_id),
                    )
                )

    def test_internal_edge_uses_its_source_junction(self):
        edge = FakeEdge(function="internal", source="goal", target="other")
        self.assertTrue(
            decision_helpers.reached_goal_junction(
                "ev",
                "goal",
                FakeNet(edge),
                traci_api=make_traci(":internal"),
            )
        )

    def test_external_edge_requires_matching_target_and_distance_threshold(self):
        edge = FakeEdge(target="goal")
        self.assertTrue(
            decision_helpers.reached_goal_junction(
                "ev",
                "goal",
                FakeNet(edge),
                eps=1.0,
                traci_api=make_traci("edge", lane_position=99.0),
            )
        )
        self.assertFalse(
            decision_helpers.reached_goal_junction(
                "ev",
                "goal",
                FakeNet(edge),
                eps=1.0,
                traci_api=make_traci("edge", lane_position=98.999),
            )
        )

    def test_external_edge_uses_an_explicit_position_tolerance(self):
        edge = FakeEdge(target="goal")
        net = FakeNet(edge)

        self.assertTrue(
            decision_helpers.reached_goal_junction(
                "ev",
                "goal",
                net,
                position_tolerance=0.001,
                traci_api=make_traci("edge", lane_position=100.0005),
            )
        )
        self.assertFalse(
            decision_helpers.reached_goal_junction(
                "ev",
                "goal",
                net,
                position_tolerance=0.001,
                traci_api=make_traci("edge", lane_position=100.01),
            )
        )
        self.assertFalse(
            decision_helpers.reached_goal_junction(
                "ev",
                "different-goal",
                FakeNet(edge),
                traci_api=make_traci("edge", lane_position=100.0),
            )
        )


class FakeTorch:
    def __init__(self):
        self.calls = []

    def zeros(self, size, *, device):
        self.calls.append((size, device))
        return [0.0] * size


class ActionMaskCharacterizationTests(unittest.TestCase):
    def test_reachable_junction_indices_are_marked_on_cpu(self):
        fake_torch = FakeTorch()
        with patch.object(
            decision_helpers,
            "candidate_next_junctions",
            return_value=(["junction-b", "junction-a"], [object(), object()]),
        ):
            result = decision_helpers.action_mask_dqn(
                object(),
                "current",
                {"junction-a": 1, "junction-b": 3},
                n=5,
                traci_api=object(),
                torch_api=fake_torch,
            )

        self.assertEqual(result, [0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertEqual(fake_torch.calls, [(5, "cpu")])

    def test_default_mask_size_is_derived_from_the_topology(self):
        fake_torch = FakeTorch()
        with patch.object(
            decision_helpers,
            "candidate_next_junctions",
            return_value=(["junction-b"], [object()]),
        ):
            result = decision_helpers.action_mask_dqn(
                object(),
                "current",
                {"junction-a": 0, "junction-b": 1},
                traci_api=object(),
                torch_api=fake_torch,
            )

        self.assertEqual(result, [0.0, 1.0])
        self.assertEqual(fake_torch.calls, [(2, "cpu")])

    def test_missing_reachable_junction_has_scenario_context(self):
        with patch.object(
            decision_helpers,
            "candidate_next_junctions",
            return_value=(["missing-junction"], [object()]),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "cur_edge=current.*missing-junction",
            ):
                decision_helpers.action_mask_dqn(
                    object(),
                    "current",
                    {"known": 0},
                    traci_api=object(),
                    torch_api=FakeTorch(),
                )


if __name__ == "__main__":
    unittest.main()
