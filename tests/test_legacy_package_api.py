from __future__ import annotations

import unittest

import legacy_dijkstra_sim
from legacy import dijkstra_sim as public_dijkstra_sim
from legacy.dijkstra_sim import dijkstra_sim as implementation_dijkstra_sim


class LegacyPackageApiTests(unittest.TestCase):
    def test_root_shim_and_package_share_the_same_implementation(self):
        self.assertIs(legacy_dijkstra_sim.dijkstra_sim, implementation_dijkstra_sim)
        self.assertIs(public_dijkstra_sim, implementation_dijkstra_sim)

    def test_legacy_package_only_advertises_the_compatibility_entrypoint(self):
        import legacy

        self.assertEqual(legacy.__all__, ["dijkstra_sim"])


if __name__ == "__main__":
    unittest.main()
