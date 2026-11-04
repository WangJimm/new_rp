from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import routing_algorithms


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
    module_name = "_routing_characterization_utils_sumo"
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


class FakeNode:
    def __init__(self, node_id: str):
        self.node_id = node_id

    def getID(self) -> str:
        return self.node_id


class FakeEdge:
    def __init__(self, to_node: str):
        self.to_node = FakeNode(to_node)

    def getToNode(self) -> FakeNode:
        return self.to_node


class FakeNet:
    def __init__(self, current_to_node: str):
        self.edges = {"current-edge": FakeEdge(current_to_node)}

    def getEdge(self, edge_id: str) -> FakeEdge:
        return self.edges[edge_id]


class DijkstraCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.net = FakeNet("A")
        self.adj = {
            "A": [("B", "edge-ab"), ("C", "edge-ac")],
            "B": [("C", "edge-bc"), ("D", "edge-bd")],
            "C": [("D", "edge-cd")],
        }
        self.costs = {
            "edge-ab": 1.0,
            "edge-ac": 5.0,
            "edge-bc": 1.0,
            "edge-bd": 10.0,
            "edge-cd": 1.0,
        }

    def cost(self, edge_id: str) -> float:
        return self.costs[edge_id]

    def test_source_already_at_goal_has_empty_path(self) -> None:
        self.assertEqual(
            utils_sumo.dijkstra_next_hop(
                self.net,
                self.adj,
                "current-edge",
                "A",
                self.cost,
            ),
            ("A", None),
        )
        self.assertEqual(
            utils_sumo.dijkstra_path(
                self.net,
                self.adj,
                "current-edge",
                "A",
                self.cost,
            ),
            (["A"], [], 0.0),
        )

    def test_cheapest_multi_hop_path_and_first_hop_are_reconstructed(self) -> None:
        self.assertEqual(
            utils_sumo.dijkstra_next_hop(
                self.net,
                self.adj,
                "current-edge",
                "D",
                self.cost,
            ),
            ("B", "edge-ab"),
        )
        self.assertEqual(
            utils_sumo.dijkstra_path(
                self.net,
                self.adj,
                "current-edge",
                "D",
                self.cost,
            ),
            (["A", "B", "C", "D"], ["edge-ab", "edge-bc", "edge-cd"], 3.0),
        )

    def test_unreachable_goal_returns_none_tuple(self) -> None:
        self.assertEqual(
            utils_sumo.dijkstra_next_hop(
                self.net,
                self.adj,
                "current-edge",
                "Z",
                self.cost,
            ),
            (None, None),
        )
        self.assertEqual(
            utils_sumo.dijkstra_path(
                self.net,
                self.adj,
                "current-edge",
                "Z",
                self.cost,
            ),
            (None, None, None),
        )

    def test_equal_cost_paths_keep_the_first_discovered_predecessor(self) -> None:
        adj = {
            "A": [("B", "edge-ab"), ("C", "edge-ac")],
            "B": [("D", "edge-bd")],
            "C": [("D", "edge-cd")],
        }
        costs = {
            "edge-ab": 1.0,
            "edge-ac": 1.0,
            "edge-bd": 1.0,
            "edge-cd": 1.0,
        }

        self.assertEqual(
            utils_sumo.dijkstra_path(
                self.net,
                adj,
                "current-edge",
                "D",
                costs.__getitem__,
            ),
            (["A", "B", "D"], ["edge-ab", "edge-bd"], 2.0),
        )

    def test_cost_by_length_returns_the_stored_edge_value(self) -> None:
        self.assertEqual(utils_sumo.cost_by_length({"edge": 12.5}, "edge"), 12.5)

    def test_utils_sumo_reexports_the_extracted_algorithms(self) -> None:
        self.assertIs(utils_sumo.cost_by_length, routing_algorithms.cost_by_length)
        self.assertIs(utils_sumo.dijkstra_next_hop, routing_algorithms.dijkstra_next_hop)
        self.assertIs(utils_sumo.dijkstra_path, routing_algorithms.dijkstra_path)


if __name__ == "__main__":
    unittest.main()
