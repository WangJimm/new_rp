"""Regression tests for turn restrictions in planned SUMO routes; no simulation."""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rp.routing.algorithms import dijkstra_next_hop, dijkstra_path
from rp.experiment.dijkstra_runner import DijkstraBaselineRunner, DijkstraEVState


class Edge:
    def __init__(self, name, target, allowed=("passenger", "bus")):
        self.name, self.target, self.allowed = name, target, allowed
        self.turns = {}

    def getID(self):
        return self.name

    def getToNode(self):
        return SimpleNamespace(getID=lambda: self.target)

    def allows(self, vclass):
        return vclass in self.allowed

    def getAllowedOutgoing(self, vclass):
        return {e: [object()] for e, classes in self.turns.items()
                if vclass is None or (vclass in classes and e.allows(vclass))}


class TurnAwareRoutingTests(unittest.TestCase):
    def setUp(self):
        # "forbidden" shares junction A with current, but is accessible only
        # after arriving on return. A node-only visited set loses this detour.
        self.edges = {name: Edge(name, target) for name, target in (
            ("current", "A"), ("out", "B"), ("return", "A"),
            ("forbidden", "G"), ("bus-only", "G"),
        )}
        e = self.edges
        e["current"].turns = {e["out"]: ("passenger", "bus"), e["bus-only"]: ("bus",)}
        e["out"].turns = {e["return"]: ("passenger", "bus")}
        e["return"].turns = {e["forbidden"]: ("passenger", "bus")}
        self.net = SimpleNamespace(getEdge=e.__getitem__)
        self.adj = {"A": [("G", "forbidden"), ("B", "out"), ("G", "bus-only")],
                    "B": [("A", "return")]}
        self.costs = {"out": 4., "return": 3., "forbidden": 2., "bus-only": 1.}

    def route(self, **kwargs):
        return dijkstra_path(self.net, self.adj, "current", "G", self.costs.__getitem__, **kwargs)

    def test_forbidden_first_turn_requires_revisiting_junction(self):
        self.assertEqual(self.route(), (["A", "B", "A", "G"], ["out", "return", "forbidden"], 9.))
        self.assertEqual(dijkstra_next_hop(self.net, self.adj, "current", "G", self.costs.__getitem__),
                         ("B", "out"))

    def test_no_connection_is_unreachable_even_if_junctions_touch(self):
        self.edges["out"].turns.clear()
        self.assertEqual(self.route(), (None, None, None))

    def test_vehicle_class_is_checked_on_each_turn(self):
        self.assertEqual(self.route(vclass="bus"), (["A", "G"], ["bus-only"], 1.))
        self.assertEqual(self.route(vclass=None), (["A", "G"], ["bus-only"], 1.))
        self.edges["current"].allowed = ("bus",)
        self.assertEqual(self.route(), (None, None, None))

    def test_permissions_are_checked_at_intermediate_turns(self):
        self.edges["return"].turns[self.edges["forbidden"]] = ("bus",)
        self.assertEqual(self.route(), (None, None, None))

    def test_caller_adjacency_still_excludes_edges(self):
        self.adj["A"] = [("G", "forbidden")]
        self.assertEqual(self.route(), (None, None, None))

    def test_dynamic_cost_callback_selects_cheapest_legal_alternative(self):
        self.edges["current"].turns[self.edges["bus-only"]] = ("passenger", "bus")
        self.costs["bus-only"] = 20.
        self.assertEqual(self.route(), (["A", "B", "A", "G"], ["out", "return", "forbidden"], 9.))
        self.costs["bus-only"] = 1.
        self.assertEqual(self.route(), (["A", "G"], ["bus-only"], 1.))

    def test_baseline_uses_actual_vehicle_class_and_reports_route_context(self):
        vehicle = SimpleNamespace(getRoadID=lambda _: "current", getVehicleClass=lambda _: "bus")
        runner = DijkstraBaselineRunner(traci=SimpleNamespace(vehicle=vehicle), net=self.net,
            static_pack={"adj": self.adj}, runtime_config=object(), cost_fn=self.costs.__getitem__)
        state = DijkstraEVState()
        with patch("rp.experiment.dijkstra_runner.remaining_routes_len", return_value=0.), \
             patch("rp.experiment.dijkstra_runner.apply_full_route") as apply:
            runner._maybe_assign_dijkstra_route("EV_2", state, "G", 2.)
        self.assertEqual(apply.call_args.args[:3], ("EV_2", "current", ["bus-only"]))
        self.assertTrue(state.decided)
        with patch("rp.experiment.dijkstra_runner.remaining_routes_len", return_value=0.), \
             patch("rp.experiment.dijkstra_runner.apply_full_route", side_effect=ValueError("bad turn")):
            with self.assertRaisesRegex(RuntimeError, "Dijkstra baseline.*EV_2.*goal_node=G.*bad turn"):
                runner._maybe_assign_dijkstra_route("EV_2", DijkstraEVState(), "G", 2.)


class BerlinTurnRegressionTests(unittest.TestCase):
    def test_reported_ev2_has_a_legal_detour(self):
        try:
            from sumolib.net import readNet
        except ImportError:
            self.skipTest("sumolib is required for the real-network regression")
        net = readNet(str(Path(__file__).resolve().parents[1] / "new_env/env_static_tls.net.xml"))
        adj = {n.getID(): [(e.getToNode().getID(), e.getID()) for e in n.getOutgoing()
                          if not e.getID().startswith(":")] for n in net.getNodes()}
        start = net.getEdge("1147628417")
        self.assertNotIn(net.getEdge("1451112465"), start.getAllowedOutgoing("passenger"))
        nodes, path, cost = dijkstra_path(net, adj, start.getID(), "5984003368",
                                         lambda e: net.getEdge(e).getLength())
        self.assertTrue(path)
        self.assertEqual(path[0], "1451112467")
        self.assertEqual(nodes[-1], "5984003368")
        for a, b in zip([start.getID()] + path, path):
            self.assertIn(net.getEdge(b), net.getEdge(a).getAllowedOutgoing("passenger"))
        self.assertAlmostEqual(cost, sum(net.getEdge(e).getLength() for e in path))


if __name__ == "__main__":
    unittest.main()
