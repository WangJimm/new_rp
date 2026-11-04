from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


from .config import RuntimeConfig
from .scenario_catalog import configured_scenarios


@dataclass(frozen=True)
class ScenarioSpec:
    general_route_file: Path
    ev_route_file: Path
    goal_file: Path
    ev_ids: list[str]
    goal_nodes: dict[str, str]
    route_files_argument: str
    sumo_seed: int | None = None


def _build_ev_ids(ev_route_file: Path) -> list[str]:
    ids = [vehicle.attrib["id"] for vehicle in ET.parse(ev_route_file).getroot().findall("vehicle")
           if vehicle.get("id", "").startswith("EV")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"EV IDs must be present and unique: {ev_route_file}")
    return ids


def _load_goal_nodes(goal_file: Path, ev_ids: list[str]) -> dict[str, str]:
    with open(goal_file, "r", encoding="utf-8") as f:
        goal_nodes = json.load(f)
    if not isinstance(goal_nodes, list):
        raise TypeError(f"Goal file must contain a JSON list: {goal_file}")
    if len(goal_nodes) != len(ev_ids):
        raise ValueError(
            "Goal/EV count mismatch: "
            f"goals={len(goal_nodes)}, evs={len(ev_ids)}, goal_file={goal_file}"
        )
    return {ev_id: goal_nodes[idx] for idx, ev_id in enumerate(ev_ids)}



def _manifest_scenario(config, split, scenario_index, probability_index, seed):
    if not getattr(config.environment, "scenario_catalog", ()):
        return None
    flows = getattr(config.environment, f"{split}_flows")
    if not 0 <= scenario_index < len(flows):
        raise ValueError(f"Invalid {split} flow index: {scenario_index}")
    flow = flows[scenario_index]
    records = [r for r in configured_scenarios(config, split)
               if (r.flow_name, r.probability_index, r.seed) == (flow, probability_index, seed)]
    if len(records) != 1:
        raise ValueError(f"Scenario is absent from {split} manifest: flow={flow}, p={probability_index}, seed={seed}")
    record = records[0]
    for path in (record.general_route_file, record.ev_route_file, record.goal_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} scenario input: {path}")
    ev_ids = _build_ev_ids(record.ev_route_file)
    if len(ev_ids) != record.ev_count:
        raise ValueError(f"EV count disagrees with scenario manifest: {record.ev_route_file}")
    goals = _load_goal_nodes(record.goal_file, ev_ids)
    return ScenarioSpec(record.general_route_file, record.ev_route_file, record.goal_file,
                        ev_ids, goals, f"{record.general_route_file}, {record.ev_route_file}", sumo_seed=seed)


def build_training_scenario(
    config: RuntimeConfig,
    scenario_index: int,
    probability_index: int,
    seed: int,
) -> ScenarioSpec:
    manifest_scenario = _manifest_scenario(config, "training", scenario_index, probability_index, seed)
    if manifest_scenario is not None:
        return manifest_scenario
    env = config.environment
    flow_name = env.training_flows[scenario_index]
    route_root = config.paths.run_dir

    general_route_file = config.paths.run_dir / "training_data" / f"output_{flow_name}_{seed}.rou.xml"
    ev_route_file = route_root / "training_data" / f"ev_{flow_name}_p0{probability_index}_{seed}.rou.xml"
    goal_file = route_root / "training_data_goal" / f"goal_ev_{flow_name}_p0{probability_index}_{seed}.json"
    ev_ids = _build_ev_ids(ev_route_file)
    goal_nodes = _load_goal_nodes(goal_file, ev_ids)

    return ScenarioSpec(
        general_route_file=general_route_file,
        ev_route_file=ev_route_file,
        goal_file=goal_file,
        ev_ids=ev_ids,
        goal_nodes=goal_nodes,
        route_files_argument=f"{general_route_file}, {ev_route_file}",
    )


def build_testing_scenario(
    config: RuntimeConfig,
    scenario_index: int,
    probability_index: int,
    seed: int,
) -> ScenarioSpec:
    manifest_scenario = _manifest_scenario(config, "testing", scenario_index, probability_index, seed)
    if manifest_scenario is not None:
        return manifest_scenario
    env = config.environment
    flow_name = env.testing_flows[scenario_index]
    route_root = config.paths.run_dir

    general_route_file = config.paths.run_dir / "testing_data" / f"output_{flow_name}_{seed}.rou.xml"
    ev_route_file = route_root / "testing_data" / f"ev_{flow_name}_p0{probability_index}_{seed}.rou.xml"
    goal_file = route_root / "testing_data_goal" / f"goal_ev_{flow_name}_p0{probability_index}_{seed}.json"
    ev_ids = _build_ev_ids(ev_route_file)
    goal_nodes = _load_goal_nodes(goal_file, ev_ids)

    return ScenarioSpec(
        general_route_file=general_route_file,
        ev_route_file=ev_route_file,
        goal_file=goal_file,
        ev_ids=ev_ids,
        goal_nodes=goal_nodes,
        route_files_argument=f"{general_route_file}, {ev_route_file}",
    )


def build_validation_scenario(
    config: RuntimeConfig,
    scenario_index: int,
    probability_index: int,
    seed: int,
) -> ScenarioSpec:
    manifest_scenario = _manifest_scenario(config, "validation", scenario_index, probability_index, seed)
    if manifest_scenario is not None:
        return manifest_scenario
    env = config.environment
    flow_name = env.validation_flows[scenario_index]
    route_root = config.paths.run_dir

    general_route_file = config.paths.run_dir / "validation_data" / f"output_{flow_name}_{seed}.rou.xml"
    ev_route_file = route_root / "validation_data" / f"ev_{flow_name}_p0{probability_index}_{seed}.rou.xml"
    goal_file = route_root / "validation_data_goal" / f"goal_ev_{flow_name}_p0{probability_index}_{seed}.json"
    ev_ids = _build_ev_ids(ev_route_file)
    goal_nodes = _load_goal_nodes(goal_file, ev_ids)

    return ScenarioSpec(
        general_route_file=general_route_file,
        ev_route_file=ev_route_file,
        goal_file=goal_file,
        ev_ids=ev_ids,
        goal_nodes=goal_nodes,
        route_files_argument=f"{general_route_file}, {ev_route_file}",
    )


__all__ = [
    "ScenarioSpec",
    "build_training_scenario",
    "build_testing_scenario",
    "build_validation_scenario",
]
