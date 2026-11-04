from __future__ import annotations

import unittest

import network_topology
import route_geometry
import routing_algorithms
from rp import routing
from rp.routing import geometry, topology


class RoutingPackageApiTests(unittest.TestCase):
    def test_public_api_exports_stable_routing_operations(self):
        expected = {
            "cost_by_length",
            "dijkstra_next_hop",
            "dijkstra_path",
            "distance_from_exit_lane_to_target_edge",
            "internal_chain_between_edges",
            "internal_chains_for_route",
            "trace_to_target_edge",
            "build_macro_candidates",
            "is_external_edge",
            "is_external_node",
            "is_internal_edge",
        }
        self.assertTrue(expected.issubset(set(routing.__all__)))

    def test_private_compatibility_helpers_are_not_public_package_api(self):
        self.assertNotIn("_safe_get_lane", routing.__all__)
        self.assertNotIn("_get_lane_from_id_or_edge", routing.__all__)
        self.assertNotIn("_is_internal_edge", routing.__all__)

    def test_root_modules_are_identity_preserving_compatibility_shims(self):
        self.assertIs(routing_algorithms.dijkstra_path, routing.dijkstra_path)
        self.assertIs(network_topology.is_external_edge, routing.is_external_edge)
        self.assertIs(route_geometry.trace_to_target_edge, routing.trace_to_target_edge)
        self.assertIs(network_topology._is_internal_edge, topology._is_internal_edge)
        self.assertIs(route_geometry._safe_get_lane, geometry._safe_get_lane)


if __name__ == "__main__":
    unittest.main()
