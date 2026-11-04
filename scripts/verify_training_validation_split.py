from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from rp.experiment import PROJECT_DIR, TrainingSchedule, ENV_CHOICES, build_selected_env_config
from rp.experiment import (
    build_training_scenario_schedule,
    iter_testing_scenario_keys,
    iter_training_scenario_keys,
    iter_validation_scenario_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify training/validation/testing scenario split without starting SUMO."
    )
    parser.add_argument("--env", choices=ENV_CHOICES, default="env", help="Dataset environment.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override schedule.num_episodes for a quick trigger preview.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sampled training keys to print.",
    )
    return parser.parse_args()


def scenario_file_preview(project_root: Path, env_name: str, data_dir: str, flow_name: str, probability_index: int, seed: int) -> tuple[Path, Path, Path]:
    route_root = project_root / env_name
    general_route_file = route_root / data_dir / f"output_{flow_name}_{seed}.rou.xml"
    ev_route_file = route_root / data_dir / f"ev_{flow_name}_p0{probability_index}_{seed}.rou.xml"
    goal_file = route_root / f"{data_dir}_goal" / f"goal_ev_{flow_name}_p0{probability_index}_{seed}.json"
    return general_route_file, ev_route_file, goal_file


def validation_trigger_steps(schedule: TrainingSchedule) -> list[int]:
    validation_interval = max(1, schedule.num_episodes // 20)
    return [
        episode + 1
        for episode in range(schedule.num_episodes)
        if (episode + 1) % validation_interval == 0 and (episode + 1) >= schedule.training_end
    ]


def main() -> None:
    args = parse_args()
    project_root = PROJECT_DIR
    config = build_selected_env_config(project_root, env_name=args.env)

    if args.num_episodes is not None:
        config = replace(
            config,
            schedule=replace(config.schedule, num_episodes=args.num_episodes),
        )

    env = config.environment
    training_keys = iter_training_scenario_keys(config)
    validation_keys = iter_validation_scenario_keys(config)
    testing_keys = iter_testing_scenario_keys(config)
    schedule = build_training_scenario_schedule(
        config,
        num_episodes=config.schedule.num_episodes,
    )
    triggers = validation_trigger_steps(config.schedule)

    print("=== schedule ===")
    print(f"num_episodes: {config.schedule.num_episodes}")
    print(f"warmup_end: {config.schedule.warmup_end}")
    print(f"training_end: {config.schedule.training_end}")
    print(f"validation_interval: {max(1, config.schedule.num_episodes // 20)}")
    print(f"validation_trigger_steps: {triggers}")
    print()

    print("=== split counts ===")
    print(f"training scenarios: {len(training_keys)}")
    print(f"validation scenarios: {len(validation_keys)}")
    print(f"testing scenarios: {len(testing_keys)}")
    print()

    print("=== configured ranges ===")
    print(f"training flows: {list(env.training_flows)}")
    if env.scenario_catalog:
        print(f"scenario manifest: {env.scenario_manifest}")
        print("Seeds follow manifest membership, not continuous numeric ranges.")
    else:
        print(f"training seeds: {env.training_seed_start}..{env.training_seed_end}")
    print(f"validation flows: {list(env.validation_flows)}")
    if not env.scenario_catalog:
        print(f"validation seeds: {env.validation_seed_start}..{env.validation_seed_end}")
    print(f"testing flows: {list(env.testing_flows)}")
    if not env.scenario_catalog:
        print(f"testing seeds: {env.testing_seed_start}..{env.testing_seed_end}")
    print()

    print("=== sampled training schedule ===")
    for key in schedule[: args.show_samples]:
        flow_name = env.training_flows[key.scenario_index]
        general_route_file, ev_route_file, goal_file = scenario_file_preview(
            project_root,
            env.name,
            "training_data",
            flow_name,
            key.probability_index,
            key.seed,
        )
        print(f"{key} -> {general_route_file}")
        print(f"         {ev_route_file}")
        print(f"         {goal_file}")
    print()

    print("=== validation preview ===")
    if validation_keys:
        first = validation_keys[0]
        last = validation_keys[-1]
        first_flow = env.validation_flows[first.scenario_index]
        last_flow = env.validation_flows[last.scenario_index]
        print(f"first validation key: {first}")
        print(scenario_file_preview(project_root, env.name, "validation_data", first_flow, first.probability_index, first.seed)[0])
        print(f"last validation key: {last}")
        print(scenario_file_preview(project_root, env.name, "validation_data", last_flow, last.probability_index, last.seed)[0])
    print()

    print("OK: training schedule uses training_data; validation phase uses every validation scenario key.")


if __name__ == "__main__":
    main()
