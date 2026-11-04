from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SCENARIOS = ["1500", "1800", "2100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize teleport vehicle IDs from validation result JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("validation_results.json"),
        help="Path to validation result JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("teleport_summary.json"),
        help="Path to write summarized teleport JSON.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario names ordered by scenario_index. Default: 1500 1800 2100.",
    )
    return parser.parse_args()


def build_scenario_key(record: dict[str, Any], scenarios: list[str]) -> str:
    scenario_index = int(record["scenario_index"])
    probability_index = int(record["probability_index"])
    seed = int(record["seed"])

    if scenario_index < 0 or scenario_index >= len(scenarios):
        scenario_name = f"scenario{scenario_index}"
    else:
        scenario_name = scenarios[scenario_index]

    return f"{scenario_name}_{probability_index:02d}_{seed}"


def summarize_records(records: list[dict[str, Any]], scenarios: list[str]) -> dict[str, dict[str, list[str]]]:
    summary: dict[str, dict[str, list[str]]] = {}
    for record in records:
        key = build_scenario_key(record, scenarios)
        summary[key] = {
            "teleport_started_ids": list(record.get("teleport_started_ids", [])),
            "teleport_ended_ids": list(record.get("teleport_ended_ids", [])),
        }
    return summary


def summarize_validation_result(data: dict[str, Any], scenarios: list[str]) -> dict[str, Any]:
    return {
        "agent": summarize_records(data.get("agent_teleport_scenarios", []), scenarios),
        "dijkstra": summarize_records(data.get("dijkstra_teleport_scenarios", []), scenarios),
    }


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = summarize_validation_result(data, list(args.scenarios))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved teleport summary: {args.output}")


if __name__ == "__main__":
    main()
