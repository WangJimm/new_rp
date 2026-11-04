from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOT_COMPATIBILITY_MODULES = {
    "config",
    "debug_utils",
    "decision_helpers",
    "dijkstra_refactored",
    "episode_runner",
    "graph_features",
    "legacy_dijkstra_sim",
    "network_topology",
    "plotting_utils",
    "route_file_utils",
    "route_geometry",
    "route_planning_main",
    "routing_algorithms",
    "runtime_state",
    "scenario_loader",
    "scenario_selection",
    "sumo_runtime",
    "tensor_utils",
    "traci_distance",
    "traci_routing",
    "traffic_signals",
    "training_orchestrator",
    "vehicle_spawning",
}

ALLOWED_FACADE_IMPORTS = {
    Path("scripts/run_experiment.py"),
    Path("rp/experiment/dijkstra_runner.py"),
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_root_python_files_are_shims_except_deprecated_facade(self):
        for path in PROJECT_ROOT.glob("*.py"):
            if path.name == "utils_sumo.py":
                continue
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            with self.subTest(path=path.name):
                self.assertIn("Compatibility", first_line)

    def test_package_and_script_code_do_not_import_root_shims(self):
        violations: list[str] = []
        for source_root in (PROJECT_ROOT / "rp", PROJECT_ROOT / "scripts"):
            for path in source_root.rglob("*.py"):
                relative = path.relative_to(PROJECT_ROOT)
                imported = imported_roots(path)
                forbidden = imported & ROOT_COMPATIBILITY_MODULES
                if forbidden:
                    violations.append(f"{relative}: {sorted(forbidden)}")
                if "utils_sumo" in imported and relative not in ALLOWED_FACADE_IMPORTS:
                    violations.append(f"{relative}: unexpected utils_sumo import")
        self.assertEqual(violations, [])

    def test_facade_declares_replacement_packages_without_runtime_import(self):
        tree = ast.parse(
            (PROJECT_ROOT / "utils_sumo.py").read_text(encoding="utf-8")
        )
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"DEPRECATED_FACADE", "FACADE_REPLACEMENTS"}
        }
        self.assertIs(assignments["DEPRECATED_FACADE"], True)
        self.assertEqual(
            assignments["FACADE_REPLACEMENTS"]["sumo"],
            "rp.sumo",
        )


if __name__ == "__main__":
    unittest.main()
