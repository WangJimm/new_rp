from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one isolated old/refactor validation snapshot."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", default="proposed")
    parser.add_argument(
        "--source",
        choices=("training", "testing", "validation"),
        default="validation",
    )
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--probability-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--random-seed", type=int, default=20260823)
    parser.add_argument("--allow-cuda", action="store_true")
    return parser.parse_args()


def load_runtime_module(project_root: Path):
    sys.path.insert(0, str(project_root))
    if (project_root / "scripts" / "run_experiment.py").is_file():
        return importlib.import_module("scripts.run_experiment")
    return importlib.import_module("route_planning_main")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not args.allow_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(project_root)

    runtime = load_runtime_module(project_root)
    config_module = importlib.import_module("config")
    scenario_module = importlib.import_module("scenario_loader")
    selection_module = importlib.import_module("scenario_selection")

    import numpy as np
    import torch

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    config = config_module.build_static_env_config(project_root)
    config = replace(config, rp_variant=args.variant, use_gui=False)
    scenario_builder = getattr(
        scenario_module,
        f"build_{args.source}_scenario",
    )
    scenario = scenario_builder(
        config,
        scenario_index=args.scenario_index,
        probability_index=args.probability_index,
        seed=args.seed,
    )
    key = selection_module.ScenarioKey(
        scenario_index=args.scenario_index,
        probability_index=args.probability_index,
        seed=args.seed,
    )

    comparison_log_root = output.parent / "sumo_logs" / args.label
    runtime.sumo_log_root = lambda _config: comparison_log_root
    runtime.bootstrap_sumo_environment(config)
    started = time.perf_counter()
    runtime.start_sumo_session(config)
    try:
        bundle = runtime.load_runtime_bundle(config, logger=None)
        runtime.maybe_load_checkpoint(
            bundle.agent,
            checkpoint,
            validate_only=True,
        )
        artifacts = runtime.create_training_artifacts()
        result = runtime.run_with_agent_eval_mode(
            bundle,
            lambda: runtime.run_validation_for_scenario_pairs(
                bundle,
                artifacts,
                [(key, scenario)],
                desc=f"{args.label}-{args.source}",
                scenario_source=args.source,
            ),
        )
    finally:
        runtime.stop_sumo_session()
    elapsed = time.perf_counter() - started

    topology_payload = {
        "edge_ids": bundle.static_pack["edge_ids"],
        "nid2idx": bundle.static_pack["nid2idx"],
    }
    topology_json = json.dumps(
        topology_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    scenario_semantics_json = json.dumps(
        {
            "ev_ids": scenario.ev_ids,
            "goal_nodes": scenario.goal_nodes,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = {
        "label": args.label,
        "project_root": str(project_root),
        "variant": args.variant,
        "source": args.source,
        "scenario_key": {
            "scenario_index": args.scenario_index,
            "probability_index": args.probability_index,
            "seed": args.seed,
        },
        "random_seed": args.random_seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "assets": {
            "network": sha256(Path(config.paths.net_file)),
            "general_route": sha256(Path(scenario.general_route_file)),
            "ev_route": sha256(Path(scenario.ev_route_file)),
            "goals": sha256(Path(scenario.goal_file)),
        },
        "scenario_semantics_sha256": hashlib.sha256(
            scenario_semantics_json
        ).hexdigest(),
        "topology": {
            "node_count": len(bundle.static_pack["nid2idx"]),
            "edge_count": len(bundle.static_pack["edge_ids"]),
            "sha256": hashlib.sha256(topology_json).hexdigest(),
        },
        "result": result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(f"snapshot written: {output}")


if __name__ == "__main__":
    main()
