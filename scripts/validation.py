from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from rp.experiment import (
    PROJECT_DIR,
    ENV_CHOICES,
    build_selected_env_config,
    validate_environment_inputs,
    RP_VARIANT_CHOICES,
    build_env_config,
    build_small_env_config,
    build_static_env_config,
)
from .run_experiment import (
    add_bottleneck_args,
    apply_episode_limit_args,
    add_reproducibility_args,
    apply_reproducibility_args,
    seed_everything,
    default_evaluation_checkpoint,
    apply_bottleneck_args,
    bootstrap_sumo_environment,
    configure_console_encoding,
    load_runtime_bundle,
    run_validate_only_flow,
    start_sumo_session,
    stop_sumo_session,
    run_training_data_validation_phase,
    create_training_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained route-planning model.")
    parser.add_argument("--env", choices=ENV_CHOICES, default="env", help="Dataset environment.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint. For new_env the default is its best full-arrival lowest-cost checkpoint under outputs/new_env/<rp_variant>/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write validation results.",
    )
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="Use sumo-gui instead of sumo.",
    )
    parser.add_argument(
        "--use-testing-data",
        action="store_true",
        help="Validate on testing data instead of validation data.",
    )
    parser.add_argument(
        "--rp_variant",
        choices=RP_VARIANT_CHOICES,
        default="proposed",
        help="Route-planning ablation variant.",
    )
    parser.add_argument("--max-decisions-per-ev", type=int, default=None)
    parser.add_argument("--max-simulation-steps", type=int, default=None)
    add_reproducibility_args(parser)
    add_bottleneck_args(parser)
    return parser.parse_args()


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    return value


def main() -> None:
    configure_console_encoding()
    args = parse_args()

    project_root = PROJECT_DIR
    # config = build_small_env_config(repo_root)
    # config = build_env_config(repo_root) 
    config = build_selected_env_config(project_root, env_name=args.env) 
    config = replace(config, rp_variant=args.rp_variant)
    

    if args.use_gui:
        config = replace(config, use_gui=True)
    config = apply_bottleneck_args(config, args)
    config = apply_episode_limit_args(config, args)
    config = apply_reproducibility_args(config, args)
    seed_everything(config.training_seed, deterministic=config.deterministic)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = default_evaluation_checkpoint(config)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}; specify --checkpoint after training.")
    validate_environment_inputs(config)
    bootstrap_sumo_environment(config)
    session_info = start_sumo_session(config)
    try:
        bundle = load_runtime_bundle(config, logger=None)
        result = run_validate_only_flow(
            bundle,
            checkpoint_path=checkpoint,
            use_testing_scenarios=args.use_testing_data,
        )


        # result = run_validate_only_flow( # 測試training data
        #     bundle,
        #     checkpoint_path=checkpoint,
        #     use_training_scenarios=True,
        # )
        

        print(f"agent type: {type(bundle.agent).__name__}")
        print(f"checkpoint: {checkpoint}")
        print(f"sumo command: {session_info['cmd']}")
        print(f"better ratio: {result.get('better_ratio')}")
        print(f"total time: {result.get('total_time')}")
        print(f"total dijkstra time: {result.get('total_dijkstra_time')}")
        print(f"total count: {result.get('total_count')}")

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(make_json_safe(result), f, ensure_ascii=False, indent=2)
            print(f"saved validation result: {args.output}")
    finally:
        stop_sumo_session()


# """
# 基本執行:
# C:\Users\ntpum\anaconda3\envs\osm\python.exe c:/Users/ntpum/OneDrive/桌面/Wang/route_planning/validation.py 
# 讀取para:
# --checkpoint outputs\dqngnn_final.pt 
# 輸出結果:
# --output outputs\validation_result.json

# C:\Users\ntpum\anaconda3\envs\osm\python.exe c:/Users/ntpum/OneDrive/桌面/Wang/route_planning/validation.py --checkpoint outputs\dqngnn_best_better_ratio.pt --output outputs\best_better_ratio_validation_result.json

# C:\Users\ntpum\anaconda3\envs\osm\python.exe c:/Users/ntpum/OneDrive/桌面/Wang/route_planning/validation.py --checkpoint outputs\new_gine_action_occupancy_only\dqngnn_best_better_ratio.pt --output outputs\new_gine_action_occupancy_only\best_better_ratio_validation_result.json

# """
if __name__ == "__main__":
    main()
