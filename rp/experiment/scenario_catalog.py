"""Read explicit scenario partitions without inferring continuous seed ranges."""
from __future__ import annotations
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("training", "validation", "testing")

@dataclass(frozen=True)
class CatalogScenario:
    split: str
    flow_name: str
    probability_index: int
    seed: int
    general_route_file: Path
    ev_route_file: Path
    goal_file: Path
    ev_count: int


def load_scenario_catalog(manifest: Path) -> tuple[CatalogScenario, ...]:
    """Validate filenames, split membership, paired variants and reused seeds."""
    root = manifest.parent.resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "flow_veh_h", "background_seed", "penetration",
                    "general_route_file", "ev_route_file", "ev_vehicle_count"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing scenario manifest columns: {manifest}")
        rows = list(reader)
    records = []
    seen = set()
    seed_splits = defaultdict(set)
    paired = defaultdict(set)
    for row in rows:
        split, flow, penetration = row["split"], row["flow_veh_h"], row["penetration"]
        if split not in SPLITS or penetration not in ("p01", "p02") or not flow.isdigit():
            raise ValueError(f"Invalid scenario manifest row: {row}")
        seed, count = int(row["background_seed"]), int(row["ev_vehicle_count"])
        probability = int(penetration[1:])
        key = (flow, probability, seed)
        if key in seen or count <= 0 or seed < 0:
            raise ValueError(f"Duplicate or invalid scenario: {key}")
        seen.add(key)
        seed_splits[seed].add(split)
        paired[(split, flow, seed)].add(probability)
        general = root / f"{split}_data" / f"output_{flow}_{seed}.rou.xml"
        ev = root / f"{split}_data" / f"ev_{flow}_{penetration}_{seed}.rou.xml"
        for column, expected in (("general_route_file", general), ("ev_route_file", ev)):
            actual = (root / row[column]).resolve()
            if not actual.is_relative_to(root) or actual != expected.resolve():
                raise ValueError(f"Manifest {column} does not match split/filename: {row}")
        goal = root / f"{split}_data_goal" / f"goal_ev_{flow}_{penetration}_{seed}.json"
        records.append(CatalogScenario(split, flow, probability, seed, general, ev, goal, count))
    if {record.split for record in records} != set(SPLITS):
        raise ValueError(f"Manifest must contain all three partitions: {manifest}")
    if any(len(splits) != 1 for splits in seed_splits.values()):
        raise ValueError("A generator seed occurs in more than one partition")
    if any(probabilities != {1, 2} for probabilities in paired.values()):
        raise ValueError("Each background scenario must include both p01 and p02")
    return tuple(sorted(records, key=lambda r: (SPLITS.index(r.split), r.seed, r.probability_index, int(r.flow_name))))


def configured_scenarios(config, split: str) -> tuple[CatalogScenario, ...]:
    catalog = getattr(config.environment, "scenario_catalog", ())
    return tuple(record for record in catalog if record.split == split)
