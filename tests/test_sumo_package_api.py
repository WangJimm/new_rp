from __future__ import annotations

import unittest

import decision_helpers
import route_file_utils
import sumo_runtime
import traci_distance
import traci_routing
import traffic_signals
import vehicle_spawning
from rp import sumo
from rp.sumo import route_files
from rp.sumo import traffic_signals as signal_implementation


class SumoPackageApiTests(unittest.TestCase):
    def test_public_api_exports_stable_sumo_operations(self):
        expected = {
            "action_mask_dqn",
            "reached_goal_junction",
            "remaining_routes_len",
            "apply_full_route",
            "apply_next_junction",
            "candidate_next_junctions",
            "build_load_args",
            "compute_signal_backbone_for_edges",
            "spawn_evs",
            "count_evs_in_rou",
            "save_routes_xml",
        }
        self.assertTrue(expected.issubset(set(sumo.__all__)))

    def test_private_compatibility_helpers_are_not_public_package_api(self):
        self.assertNotIn("_pick_active_logic", sumo.__all__)
        self.assertNotIn("_split_edges", sumo.__all__)

    def test_root_modules_are_identity_preserving_compatibility_shims(self):
        self.assertIs(decision_helpers.action_mask_dqn, sumo.action_mask_dqn)
        self.assertIs(traci_distance.remaining_routes_len, sumo.remaining_routes_len)
        self.assertIs(traci_routing.apply_full_route, sumo.apply_full_route)
        self.assertIs(traffic_signals.lane2edge, sumo.lane2edge)
        self.assertIs(vehicle_spawning.spawn_evs, sumo.spawn_evs)
        self.assertIs(sumo_runtime.build_load_args, sumo.build_load_args)
        self.assertIs(route_file_utils.save_routes_xml, sumo.save_routes_xml)
        self.assertIs(
            traffic_signals._pick_active_logic,
            signal_implementation._pick_active_logic,
        )
        self.assertIs(route_file_utils._split_edges, route_files._split_edges)


if __name__ == "__main__":
    unittest.main()
