from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


class FakePhase:
    def __init__(self, duration, state):
        self.duration = duration
        self.state = state


class FakeLogic:
    def __init__(self, program_id, phases):
        self.programID = program_id
        self.phases = list(phases)

    def getPhases(self):
        return self.phases


class FakeTrafficLightAPI:
    def __init__(self):
        self.ids = []
        self.programs = {}
        self.logics = {}
        self.controlled_links = {}
        self.phases = {}
        self.next_switches = {}

    def getIDList(self):
        return self.ids

    def getProgram(self, tls_id):
        return self.programs[tls_id]

    def getAllProgramLogics(self, tls_id):
        return self.logics[tls_id]

    def getControlledLinks(self, tls_id):
        return self.controlled_links[tls_id]

    def getPhase(self, tls_id):
        return self.phases[tls_id]

    def getNextSwitch(self, tls_id):
        return self.next_switches[tls_id]


class FakeSimulationAPI:
    def __init__(self, current_time=0.0):
        self.current_time = current_time

    def getTime(self):
        return self.current_time


def make_traci(current_time=0.0):
    fake = types.ModuleType("traci")
    fake.constants = types.SimpleNamespace()
    fake.trafficlight = FakeTrafficLightAPI()
    fake.simulation = FakeSimulationAPI(current_time)
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
    module_name = f"_traffic_signal_characterization_{id(fake_traci)}"
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


def configure_tls(fake_traci, *, phases=None, controlled_links=None):
    tls = fake_traci.trafficlight
    tls.ids = ["tls-1"]
    tls.programs["tls-1"] = "active"
    tls.logics["tls-1"] = [
        FakeLogic("inactive", [FakePhase(99, "r")]),
        FakeLogic("active", phases or [FakePhase(10, "G")]),
    ]
    tls.controlled_links["tls-1"] = controlled_links or []
    tls.phases["tls-1"] = 0
    tls.next_switches["tls-1"] = 10.0


class SignalProgramCharacterizationTests(unittest.TestCase):
    def test_lane_to_edge_removes_only_the_final_lane_suffix(self):
        legacy = load_utils_sumo_with_traci(make_traci())

        self.assertEqual(legacy.lane2edge("road_segment_12"), "road_segment")
        self.assertEqual(legacy.lane2edge(":junction_0_1"), ":junction_0")

    def test_active_logic_matches_program_and_falls_back_to_first(self):
        fake = make_traci()
        configure_tls(fake)
        legacy = load_utils_sumo_with_traci(fake)

        selected = legacy._pick_active_logic("tls-1")
        self.assertEqual(selected.programID, "active")

        fake.trafficlight.programs["tls-1"] = "missing"
        self.assertEqual(legacy._pick_active_logic("tls-1").programID, "inactive")

        fake.trafficlight.logics["tls-1"] = []
        self.assertIsNone(legacy._pick_active_logic("tls-1"))

    def test_green_ratio_uses_cycle_weighting_and_mean_across_links(self):
        fake = make_traci()
        configure_tls(
            fake,
            phases=[FakePhase(10, "Gr"), FakePhase(20, "rG")],
        )
        legacy = load_utils_sumo_with_traci(fake)

        cycle, mean_ratio = legacy.green_ratio_for_link_indices("tls-1", [0, 1])

        self.assertEqual(cycle, 30.0)
        self.assertAlmostEqual(mean_ratio, 0.5)
        self.assertEqual(
            legacy.green_ratio_for_link_indices("tls-1", []),
            (30.0, 0.0),
        )

    def test_zero_duration_cycle_returns_zero_ratio(self):
        fake = make_traci()
        configure_tls(fake, phases=[FakePhase(0, "G")])
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(
            legacy.green_ratio_for_link_indices("tls-1", [0]),
            (0.0, 0.0),
        )


class SignalLinkCharacterizationTests(unittest.TestCase):
    def test_controlled_link_table_keeps_link_indices_per_incoming_edge(self):
        fake = make_traci()
        configure_tls(
            fake,
            controlled_links=[
                [("in_edge_0", "out_0", "via")],
                [("other_0", "out_1", "via"), ("in_edge_1", "out_2", "via")],
            ],
        )
        legacy = load_utils_sumo_with_traci(fake)

        table = legacy.build_inedge_to_tls_and_indices()

        self.assertEqual(table["in_edge"]["tls-1"], [0, 1])
        self.assertEqual(table["other"]["tls-1"], [1])

    def test_signal_backbone_uses_defaults_for_missing_or_uncontrolled_tls(self):
        fake = make_traci()
        configure_tls(
            fake,
            phases=[FakePhase(10, "G"), FakePhase(20, "r")],
            controlled_links=[[("incoming_0", "out_0", "via")]],
        )
        legacy = load_utils_sumo_with_traci(fake)

        cycles, ratios = legacy.compute_signal_backbone_for_edges(
            ["incoming", "missing", "incoming"],
            ["tls-1", "tls-1", None],
        )

        self.assertEqual(cycles, [30.0, 0.0, 0.0])
        self.assertAlmostEqual(ratios[0], 1.0 / 3.0)
        self.assertEqual(ratios[1:], [1.0, 1.0])

    def test_movement_lookup_matches_exact_edge_ids(self):
        fake = make_traci()
        configure_tls(
            fake,
            controlled_links=[
                [("edge10_0", "outbound_0", "via")],
                [("edge1_0", "out_0", "via")],
            ],
        )
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(
            legacy.find_tls_and_link_index_for_movement("edge1", "out"),
            ("tls-1", 1),
        )
        self.assertEqual(
            legacy.find_tls_and_link_index_for_movement("edge1", "outbound"),
            (None, None),
        )
        self.assertEqual(
            legacy.find_tls_and_link_index_for_movement("missing", "out"),
            (None, None),
        )


class SignalTimingCharacterizationTests(unittest.TestCase):
    def test_current_green_returns_only_current_phase_remainder(self):
        fake = make_traci(current_time=7.0)
        configure_tls(fake, phases=[FakePhase(10, "G"), FakePhase(20, "r")])
        fake.trafficlight.next_switches["tls-1"] = 12.0
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(
            legacy.time_to_green_and_remaining_green("tls-1", 0),
            (0.0, 5.0),
        )

    def test_red_phase_accumulates_until_next_green(self):
        fake = make_traci(current_time=8.0)
        configure_tls(
            fake,
            phases=[
                FakePhase(10, "r"),
                FakePhase(4, "r"),
                FakePhase(6, "G"),
            ],
        )
        fake.trafficlight.next_switches["tls-1"] = 10.0
        legacy = load_utils_sumo_with_traci(fake)

        self.assertEqual(
            legacy.time_to_green_and_remaining_green("tls-1", 0),
            (6.0, 6.0),
        )

    def test_never_green_returns_infinity(self):
        fake = make_traci(current_time=20.0)
        configure_tls(fake, phases=[FakePhase(10, "r"), FakePhase(5, "r")])
        fake.trafficlight.next_switches["tls-1"] = 10.0
        legacy = load_utils_sumo_with_traci(fake)

        wait, remaining = legacy.time_to_green_and_remaining_green("tls-1", 0)

        self.assertTrue(math.isinf(wait))
        self.assertEqual(remaining, 0.0)


if __name__ == "__main__":
    unittest.main()
