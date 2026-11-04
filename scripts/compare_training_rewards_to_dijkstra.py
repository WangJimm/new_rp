from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import traci

from rp.experiment import (
    PROJECT_DIR,
    ENV_CHOICES,
    build_selected_env_config,
    validate_environment_inputs,
    build_small_env_config,
    build_static_env_config,
    run_dijkstra_baseline_for_training_key,
)
from .run_experiment import (
    bootstrap_sumo_environment,
    configure_console_encoding,
    load_runtime_bundle,
    start_sumo_session,
    stop_sumo_session,
)
from rp.experiment import build_training_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay training scenarios with Dijkstra and compare per-episode costs."
    )
    parser.add_argument("--env", choices=ENV_CHOICES, default="env", help="Dataset environment.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to training_rewards.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write comparison results.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start from this record index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to process.",
    )
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="Use sumo-gui instead of sumo.",
    )
    parser.add_argument(
        "--restart-interval",
        type=int,
        default=100,
        help="Restart SUMO after this many records. Use 0 to disable periodic restarts.",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> list[dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Input JSON does not contain a valid 'records' list.")
    return records


def slice_records(records: list[dict[str, Any]], start: int, limit: int | None) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")
    sliced = records[start:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def compute_agent_cost(config, record: dict[str, Any]) -> tuple[float, int]:
    scenario = build_training_scenario(
        config,
        scenario_index=int(record["scenario_index"]),
        probability_index=int(record["probability_index"]),
        seed=int(record["seed"]),
    )
    reward = float(record["reward"])
    ev_count = len(scenario.ev_ids)
    agent_cost = config.success_bonus * ev_count - reward
    return agent_cost, ev_count


def build_episode_result(
    config,
    bundle,
    record: dict[str, Any],
) -> dict[str, Any]:
    scenario_index = int(record["scenario_index"])
    probability_index = int(record["probability_index"])
    seed = int(record["seed"])
    episode = int(record["episode"])
    reward = float(record["reward"])
    average_reward = record.get("average_reward")

    agent_cost, ev_count = compute_agent_cost(config, record)
    dijkstra_result = run_dijkstra_baseline_for_training_key(
        traci=traci,
        net=bundle.net,
        static_pack=bundle.static_pack,
        runtime_config=config,
        scenario_index=scenario_index,
        probability_index=probability_index,
        seed=seed,
    )
    dijkstra_cost = float(dijkstra_result.total_time_spend)
    ratio = dijkstra_cost / agent_cost if agent_cost > 0 else None

    return {
        "episode": episode,
        "scenario_index": scenario_index,
        "probability_index": probability_index,
        "seed": seed,
        "ev_count": ev_count,
        "reward": reward,
        "average_reward": average_reward,
        "agent_cost": agent_cost,
        "dijkstra_cost": dijkstra_cost,
        "dijkstra_over_agent": ratio,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [row["dijkstra_over_agent"] for row in results if row["dijkstra_over_agent"] is not None]
    return {
        "episode_count": len(results),
        "avg_agent_cost": sum(row["agent_cost"] for row in results) / len(results) if results else None,
        "avg_dijkstra_cost": sum(row["dijkstra_cost"] for row in results) / len(results) if results else None,
        "avg_dijkstra_over_agent": sum(ratios) / len(ratios) if ratios else None,
        "better_or_equal_count": sum(
            1 for row in results if row["dijkstra_over_agent"] is not None and row["dijkstra_over_agent"] >= 1.0
        ),
    }


def restart_runtime(config):
    stop_sumo_session()
    start_sumo_session(config)
    return load_runtime_bundle(config, logger=None)


def main() -> None:
    configure_console_encoding()
    args = parse_args()

    project_root = PROJECT_DIR
    # config = build_small_env_config(repo_root)
    config = build_selected_env_config(project_root, env_name=args.env) 

    if args.use_gui:
        config = replace(config, use_gui=True)
    if args.restart_interval < 0:
        raise ValueError("--restart-interval must be >= 0")
    

    records = load_records(args.input)
    selected_records = slice_records(records, args.start, args.limit)
    if not selected_records:
        raise ValueError("No records selected. Check --start / --limit.")

    validate_environment_inputs(config)
    bootstrap_sumo_environment(config)
    start_sumo_session(config)
    try:
        bundle = load_runtime_bundle(config, logger=None)
        results: list[dict[str, Any]] = []

        for idx, record in enumerate(selected_records, start=1):
            if args.restart_interval > 0 and idx > 1 and (idx - 1) % args.restart_interval == 0:
                bundle = restart_runtime(config)
                print(f"[sumo restart] restarted before record {idx}/{len(selected_records)}")
            result = build_episode_result(config, bundle, record)
            results.append(result)
            print(
                f"[{idx}/{len(selected_records)}] ep={result['episode']} "
                f"agent={result['agent_cost']:.6f} dijkstra={result['dijkstra_cost']:.6f} "
                f"ratio={result['dijkstra_over_agent'] if result['dijkstra_over_agent'] is not None else 'NA'}"
            )

        payload = {
            "input": str(args.input),
            "start": args.start,
            "limit": args.limit,
            "restart_interval": args.restart_interval,
            "success_bonus": config.success_bonus,
            "cost_formula": "agent_cost = success_bonus * ev_count - reward",
            "ratio_formula": "dijkstra_over_agent = dijkstra_cost / agent_cost",
            "summary": summarize(results),
            "episodes": results,
        }

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"saved comparison result: {args.output}")
    finally:
        stop_sumo_session()


if __name__ == "__main__":
    main()


# """
# 使用方式:

#   C:\Users\ntpum\anaconda3\envs\osm\python.exe c:/Users/ntpum/OneDrive/桌面/Wang/route_planning/compare_training_rewards_to_dijkstra.py `
#   --input .\outputs\gine_action_static_only\training_rewards.json `
#   --start 0 `
#   --limit 10 `
#   --output ".\outputs\gine_action_static_only\training_vs_dijkstra_sample.json"

# C:\Users\ntpum\anaconda3\envs\osm\python.exe c:/Users/ntpum/OneDrive/桌面/Wang/route_planning/compare_training_rewards_to_dijkstra.py `
# --input .\outputs\env\gine_action_static_occupancy\training_rewards.json `
# --start 498 `
# --output ".\outputs\env\gine_action_static_occupancy\training_vs_dijkstra_sample.json"

# """


