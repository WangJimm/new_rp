from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from config import build_static_env_config
from scenario_loader import build_training_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = PROJECT_ROOT / "env"

FIXED_SCENARIO_FILES = {
    "env/osm.sumocfg": "a489df35b17decd92e8fb488b60bcbacd78cecf1c9efa8bdd749b9106c003227",
    "env/env_static_tls.net.xml": "81e71d20052ad5340080a27197b34888748576a4afdce5b8dec43336c704a4c4",
    "env/training_data/output_1800_1.rou.xml": "8376f4c0b842269dc15707cb0fe34df8c20c58a71d00c335491efdac26ce2985",
    "env/training_data/ev_1800_p01_1.rou.xml": "f5f642ea1c6d09724e96658c8da8af19ca6c72a053b30b5fea4eb853f0caf088",
    "env/training_data_goal/goal_ev_1800_p01_1.json": "655262f7b658479c201a7ec80df2da74d5bcd4cfa642251729094159628e1c21",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_sumo_binary() -> str | None:
    configured_binary = os.environ.get("SUMO_BINARY")
    if configured_binary and Path(configured_binary).is_file():
        return configured_binary

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        for relative in (Path("bin/sumo.exe"), Path("Scripts/sumo.exe"), Path("bin/sumo")):
            candidate = Path(sumo_home) / relative
            if candidate.is_file():
                return str(candidate)

    return shutil.which("sumo") or shutil.which("sumo.exe")


SUMO_BINARY = find_sumo_binary()


class FixedScenarioAssetCharacterizationTests(unittest.TestCase):
    def test_seed_1_training_scenario_files_match_snapshot(self) -> None:
        for relative_path, expected_hash in FIXED_SCENARIO_FILES.items():
            with self.subTest(path=relative_path):
                path = PROJECT_ROOT / relative_path
                self.assertTrue(path.is_file(), f"Missing fixed scenario asset: {path}")
                self.assertEqual(sha256(path), expected_hash)

    def test_seed_1_training_scenario_builds_from_real_route_and_goal_files(self) -> None:
        config = build_static_env_config(PROJECT_ROOT)

        scenario = build_training_scenario(
            config,
            scenario_index=0,
            probability_index=1,
            seed=1,
        )

        expected_general = ENV_DIR / "training_data/output_1800_1.rou.xml"
        expected_ev = ENV_DIR / "training_data/ev_1800_p01_1.rou.xml"
        self.assertEqual(scenario.general_route_file, expected_general)
        self.assertEqual(scenario.ev_route_file, expected_ev)
        self.assertEqual(len(scenario.ev_ids), 9)
        self.assertEqual(len(scenario.goal_nodes), 9)
        self.assertEqual(scenario.ev_ids[0], "EV_0")
        self.assertEqual(scenario.goal_nodes["EV_0"], "42496394")
        self.assertEqual(
            scenario.route_files_argument,
            f"{expected_general}, {expected_ev}",
        )

    def test_all_route_and_goal_assets_have_exact_ev_cardinality(self) -> None:
        checked = 0
        for data_kind in ("training", "testing", "validation"):
            route_dir = ENV_DIR / f"{data_kind}_data"
            goal_dir = ENV_DIR / f"{data_kind}_data_goal"
            if not goal_dir.is_dir():
                continue
            for goal_file in goal_dir.glob("*.json"):
                route_name = goal_file.name.removeprefix("goal_").replace(
                    ".json", ".rou.xml"
                )
                route_file = route_dir / route_name
                with self.subTest(goal_file=goal_file.name):
                    self.assertTrue(route_file.is_file())
                    goals = json.loads(goal_file.read_text(encoding="utf-8"))
                    vehicles = ET.parse(route_file).getroot().findall("vehicle")
                    self.assertEqual(len(goals), len(vehicles))
                checked += 1
        self.assertEqual(checked, 220)


@unittest.skipUnless(SUMO_BINARY, "SUMO binary is not installed or configured")
class SumoProcessSmokeTests(unittest.TestCase):
    def test_seed_1_training_scenario_loads_and_runs_five_seconds(self) -> None:
        general_route = ENV_DIR / "training_data/output_1800_1.rou.xml"
        ev_route = ENV_DIR / "training_data/ev_1800_p01_1.rou.xml"
        command = [
            str(SUMO_BINARY),
            "-c",
            str(ENV_DIR / "osm.sumocfg"),
            "--net-file",
            str(ENV_DIR / "env_static_tls.net.xml"),
            "--route-files",
            f"{general_route},{ev_route}",
            "--begin",
            "0",
            "--end",
            "5",
            "--no-step-log",
            "true",
            "--verbose",
            "false",
            "-W",
            "true",
        ]

        result = subprocess.run(
            command,
            cwd=ENV_DIR,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"SUMO smoke test failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
