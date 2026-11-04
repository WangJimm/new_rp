from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import network_topology


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
    module_name = "_topology_characterization_utils_sumo"
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


utils_sumo = load_utils_sumo_with_dependency_stubs()


class CompatibilityReexportTests(unittest.TestCase):
    def test_utils_sumo_reexports_extracted_topology_helpers(self) -> None:
        helper_names = (
            "is_external_node",
            "is_external_edge",
            "is_decisionable_edge",
            "edge_allows",
            "expand_to_decision_edges",
            "build_macro_candidates",
            "_is_internal_edge",
            "is_all_internal_to",
            "is_internal_edge",
        )
        for name in helper_names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(utils_sumo, name),
                    getattr(network_topology, name),
                )

        self.assertEqual(utils_sumo.MIN_DECISION_LEN, network_topology.MIN_DECISION_LEN)
        self.assertEqual(utils_sumo.MAX_SKIP_HOPS, network_topology.MAX_SKIP_HOPS)
        self.assertIs(utils_sumo.VCLASS_FILTER, network_topology.VCLASS_FILTER)


class FakeLane:
    def __init__(self, allowed: bool = True):
        self.allowed = allowed

    def allows(self, _vehicle_class: str) -> bool:
        return self.allowed


class FakeNode:
    def __init__(self, node_id: str, node_type: str = "priority"):
        self.node_id = node_id
        self.node_type = node_type
        self.outgoing = []

    def getID(self) -> str:
        return self.node_id

    def getType(self) -> str:
        return self.node_type

    def getOutgoing(self):
        return self.outgoing


class FakeEdge:
    def __init__(
        self,
        edge_id: str,
        to_node: FakeNode,
        *,
        length: float = 10.0,
        internal: bool = False,
        allowed: bool = True,
    ):
        self.edge_id = edge_id
        self.to_node = to_node
        self.length = length
        self.internal = internal
        self.lanes = [FakeLane(allowed)]

    def getID(self) -> str:
        return self.edge_id

    def getFunction(self) -> str:
        return "internal" if self.internal else "normal"

    def isInternal(self) -> bool:
        return self.internal

    def getLength(self) -> float:
        return self.length

    def getToNode(self) -> FakeNode:
        return self.to_node

    def getLanes(self):
        return self.lanes


class FakeNet:
    def __init__(self, edges: dict[str, FakeEdge]):
        self.edges = edges

    def getEdge(self, edge_id: str) -> FakeEdge:
        return self.edges[edge_id]


class EdgeClassificationCharacterizationTests(unittest.TestCase):
    def test_external_node_and_edge_require_non_internal_type_and_id(self) -> None:
        self.assertTrue(utils_sumo.is_external_node(FakeNode("node")))
        self.assertFalse(utils_sumo.is_external_node(FakeNode(":node")))
        self.assertFalse(utils_sumo.is_external_node(FakeNode("node", "internal")))

        node = FakeNode("target")
        self.assertTrue(utils_sumo.is_external_edge(FakeEdge("edge", node)))
        self.assertFalse(utils_sumo.is_external_edge(FakeEdge(":edge", node)))
        self.assertFalse(
            utils_sumo.is_external_edge(FakeEdge("edge", node, internal=True))
        )

    def test_decisionable_edge_uses_two_meter_inclusive_boundary(self) -> None:
        node = FakeNode("target")
        self.assertFalse(
            utils_sumo.is_decisionable_edge(FakeEdge("short", node, length=1.999))
        )
        self.assertTrue(
            utils_sumo.is_decisionable_edge(FakeEdge("boundary", node, length=2.0))
        )
        self.assertFalse(
            utils_sumo.is_decisionable_edge(
                FakeEdge(":internal", node, length=20.0, internal=True)
            )
        )

    def test_edge_allows_uses_first_lane_and_fails_open(self) -> None:
        node = FakeNode("target")
        self.assertTrue(utils_sumo.edge_allows(FakeEdge("edge", node), None))
        self.assertFalse(
            utils_sumo.edge_allows(
                FakeEdge("edge", node, allowed=False),
                "passenger",
            )
        )

        edge_without_lanes = FakeEdge("edge", node)
        edge_without_lanes.lanes = []
        self.assertTrue(utils_sumo.edge_allows(edge_without_lanes, "passenger"))

    def test_internal_edge_falls_back_to_colon_prefix(self) -> None:
        edge = FakeEdge(":fallback", FakeNode("target"), internal=True)
        edge.getFunction = lambda: (_ for _ in ()).throw(RuntimeError("unsupported"))

        self.assertTrue(utils_sumo.is_internal_edge(edge))
        self.assertTrue(utils_sumo._is_internal_edge(edge))


class DecisionEdgeExpansionCharacterizationTests(unittest.TestCase):
    def build_chain(self):
        target = FakeNode("target")
        middle = FakeNode("middle")
        entry = FakeNode("entry")
        real_edge = FakeEdge("real-edge", target, length=10.0)
        short_edge = FakeEdge("short-edge", middle, length=1.0)
        internal_edge = FakeEdge(":internal", entry, length=0.5, internal=True)
        entry.outgoing = [short_edge]
        middle.outgoing = [real_edge]
        return internal_edge, short_edge, real_edge

    def test_expansion_crosses_internal_and_short_edges_to_first_real_edge(self) -> None:
        internal_edge, _short_edge, _real_edge = self.build_chain()

        self.assertEqual(
            utils_sumo.expand_to_decision_edges(None, internal_edge),
            [("target", "real-edge", [":internal", "short-edge"])],
        )

    def test_expansion_falls_back_to_start_edge_when_no_candidate_exists(self) -> None:
        dead_end_node = FakeNode("dead-end")
        start = FakeEdge(":dead", dead_end_node, length=0.5, internal=True)

        self.assertEqual(
            utils_sumo.expand_to_decision_edges(None, start),
            [("dead-end", ":dead", [])],
        )

    def test_vehicle_class_filter_can_force_fallback(self) -> None:
        target = FakeNode("target")
        entry = FakeNode("entry")
        blocked = FakeEdge("blocked", target, allowed=False)
        start = FakeEdge(":start", entry, internal=True)
        entry.outgoing = [blocked]

        self.assertEqual(
            utils_sumo.expand_to_decision_edges(
                None,
                start,
                vclass="passenger",
            ),
            [("entry", ":start", [])],
        )

    def test_macro_candidates_combine_direct_and_expanded_edges_in_order(self) -> None:
        internal_edge, _short_edge, real_edge = self.build_chain()
        direct = FakeEdge("direct", FakeNode("direct-target"), length=5.0)

        macros = utils_sumo.build_macro_candidates(
            None,
            "current-edge",
            [direct, internal_edge, real_edge],
        )

        self.assertEqual(
            macros,
            [
                {"node": "direct-target", "first_edge": "direct", "prefix": []},
                {
                    "node": "target",
                    "first_edge": "real-edge",
                    "prefix": [":internal", "short-edge"],
                },
            ],
        )


class InternalOnlyReachabilityCharacterizationTests(unittest.TestCase):
    def test_current_exit_already_at_target_has_zero_length(self) -> None:
        target = FakeNode("target")
        current = FakeEdge("current", target)

        self.assertEqual(
            utils_sumo.is_all_internal_to(
                FakeNet({"current": current}),
                "current",
                "target",
            ),
            (True, 0.0),
        )

    def test_internal_only_path_accumulates_edge_lengths(self) -> None:
        target = FakeNode("target")
        middle = FakeNode("middle")
        source = FakeNode("source")
        second = FakeEdge(":second", target, length=3.0, internal=True)
        first = FakeEdge(":first", middle, length=2.0, internal=True)
        source.outgoing = [first]
        middle.outgoing = [second]
        current = FakeEdge("current", source)

        self.assertEqual(
            utils_sumo.is_all_internal_to(
                FakeNet({"current": current}),
                "current",
                "target",
            ),
            (True, 5.0),
        )

    def test_external_edges_do_not_count_as_internal_only_path(self) -> None:
        target = FakeNode("target")
        source = FakeNode("source")
        source.outgoing = [FakeEdge("external", target, length=2.0)]
        current = FakeEdge("current", source)

        self.assertEqual(
            utils_sumo.is_all_internal_to(
                FakeNet({"current": current}),
                "current",
                "target",
            ),
            (False, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
