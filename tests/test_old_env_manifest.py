from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path

from rp.experiment.config import build_selected_env_config, validate_environment_inputs
from rp.experiment.scenario_loader import build_training_scenario, build_validation_scenario
from rp.experiment.scenario_selection import (
    iter_training_scenario_keys, iter_validation_scenario_keys, iter_testing_scenario_keys,
    build_training_scenario_schedule,
)

GOALS = ["42496394", "42490303", "42474000"]

class OldEnvManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / "env"
        self.env.mkdir()
        (self.env / "env_static_tls.net.xml").write_text(
            "<net>" + "".join(f'<junction id="{g}"/>' for g in GOALS) + "</net>",
            encoding="utf-8")
        (self.env / "osm.sumocfg").write_text(
            '<configuration><input><net-file value="env_static_tls.net.xml"/></input></configuration>',
            encoding="utf-8")
        self.rows = []
        for split, seed in [("training", 10), ("validation", 20), ("testing", 30)]:
            data = self.env / f"{split}_data"
            goals = self.env / f"{split}_data_goal"
            data.mkdir()
            goals.mkdir()
            for flow in [1500, 1800, 2100]:
                general = data / f"output_{flow}_{seed}.rou.xml"
                general.write_text("<routes/>", encoding="utf-8")
                for penetration in ["p01", "p02"]:
                    ev = data / f"ev_{flow}_{penetration}_{seed}.rou.xml"
                    ev.write_text('<routes><vehicle id="EV_7"/><vehicle id="EV_2"/></routes>',
                                  encoding="utf-8")
                    (goals / f"goal_ev_{flow}_{penetration}_{seed}.json").write_text(
                        json.dumps(GOALS[:2]), encoding="utf-8")
                    self.rows.append({
                        "split": split, "flow_veh_h": flow, "background_seed": seed,
                        "penetration": penetration, "ev_vehicle_count": 2,
                        "general_route_file": general.relative_to(self.env).as_posix(),
                        "ev_route_file": ev.relative_to(self.env).as_posix(),
                    })

    def write_manifest(self):
        with (self.env / "scenario_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_manifest_enables_all_demands_and_uses_passed_project_root(self):
        self.write_manifest()
        config = build_selected_env_config(self.root, "env")
        self.assertEqual(config.paths.run_dir, self.env)
        self.assertEqual(tuple(config.environment.training_flows), ("1500", "1800", "2100"))
        self.assertEqual(tuple(config.environment.cs_locations), tuple(GOALS))
        self.assertEqual(config.environment.scenario_manifest, self.env / "scenario_manifest.csv")
        for iterator, seed in [
            (iter_training_scenario_keys, 10),
            (iter_validation_scenario_keys, 20),
            (iter_testing_scenario_keys, 30),
        ]:
            keys = iterator(config)
            self.assertEqual(len(keys), 6)
            self.assertEqual({k.seed for k in keys}, {seed})
        validate_environment_inputs(config)

    def test_no_manifest_retains_legacy_seed_ranges(self):
        config = build_selected_env_config(self.root, "env")
        self.assertEqual(config.paths.run_dir, self.env)
        self.assertFalse(config.environment.scenario_catalog)
        self.assertEqual(list(config.environment.training_flows), ["1800"])
        self.assertEqual(len(iter_training_scenario_keys(config)), 40)
        self.assertEqual({k.seed for k in iter_training_scenario_keys(config)}, set(range(1, 21)))

    def test_actual_ev_ids_goals_and_sumo_seed_follow_manifest(self):
        self.write_manifest()
        config = build_selected_env_config(self.root, "env")
        spec = build_training_scenario(config, 0, 1, 10)
        self.assertEqual(spec.ev_ids, ["EV_7", "EV_2"])
        self.assertEqual(spec.goal_nodes, dict(zip(spec.ev_ids, GOALS[:2])))
        self.assertEqual(spec.sumo_seed, 10)
        with self.assertRaisesRegex(ValueError, "absent from training"):
            build_training_scenario(config, 0, 1, 20)

    def test_schedule_is_repeatable_and_excludes_held_out_seeds(self):
        self.write_manifest()
        config = build_selected_env_config(self.root, "env")
        first = build_training_scenario_schedule(config, 100, seed=42)
        self.assertEqual(first, build_training_scenario_schedule(config, 100, seed=42))
        self.assertEqual({x.seed for x in first}, {10})

    def test_old_manifest_preflight_rejects_missing_goals(self):
        self.write_manifest()
        config = build_selected_env_config(self.root, "env")
        (self.env / "validation_data_goal/goal_ev_1500_p01_20.json").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_environment_inputs(config)

    def test_old_manifest_preflight_rejects_foreign_goal(self):
        self.write_manifest()
        (self.env / "training_data_goal/goal_ev_1500_p01_10.json").write_text(
            json.dumps(["1419061109", GOALS[0]]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected goals"):
            validate_environment_inputs(build_selected_env_config(self.root, "env"))

    def test_old_manifest_rejects_cross_demand_seed_leakage(self):
        for row in self.rows:
            if row["split"] == "validation" and row["flow_veh_h"] == 1500:
                row["flow_veh_h"] = 2400
                row["background_seed"] = 10
                row["general_route_file"] = row["general_route_file"].replace("1500", "2400").replace("_20.", "_10.")
                row["ev_route_file"] = row["ev_route_file"].replace("1500", "2400").replace("_20.", "_10.")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "more than one partition"):
            build_selected_env_config(self.root, "env")

if __name__ == "__main__":
    unittest.main()
