"""SUMO process argument construction without experiment-layer imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rp.experiment import RuntimeConfig, ScenarioSpec


def build_load_args(config: RuntimeConfig, scenario: ScenarioSpec) -> list[str]:
    args = [
        "-c",
        str(config.paths.config_file),
        "--step-length",
        str(config.step_length),
        "--route-files",
        scenario.route_files_argument,
        "-W",
        "true",
    ]
    seed = getattr(scenario, "sumo_seed", None)
    if seed is None:
        seed = getattr(config, "training_seed", None)
    if seed is not None:
        args.extend(["--seed", str(seed)])
    return args


__all__ = ["build_load_args"]
