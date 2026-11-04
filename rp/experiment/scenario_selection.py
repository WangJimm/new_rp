from __future__ import annotations
import random
from dataclasses import dataclass

from .config import RuntimeConfig
from .scenario_catalog import configured_scenarios


@dataclass(frozen=True)
class ScenarioKey:
    scenario_index: int
    probability_index: int
    seed: int



def _catalog_keys(config: RuntimeConfig, split: str) -> list[ScenarioKey]:
    flows = getattr(config.environment, f"{split}_flows")
    return [ScenarioKey(flows.index(r.flow_name), r.probability_index, r.seed)
            for r in configured_scenarios(config, split)]


def choose_training_scenario_key(config: RuntimeConfig, rng: random.Random | None = None) -> ScenarioKey:
    rng = rng or random.Random()
    env = config.environment
    if getattr(env, "scenario_catalog", ()):
        return rng.choice(_catalog_keys(config, "training"))
    return ScenarioKey(
        scenario_index=rng.randint(0, len(env.training_flows) - 1),
        probability_index=rng.randint(1, 2),
        seed=rng.randint(env.training_seed_start, env.training_seed_end),
    )


def build_training_scenario_schedule(config: RuntimeConfig, num_episodes: int, seed: int = 42) -> list[ScenarioKey]:
    rng = random.Random(seed)
    return [choose_training_scenario_key(config, rng=rng) for _ in range(num_episodes)]


def iter_training_scenario_keys(config: RuntimeConfig) -> list[ScenarioKey]:
    env = config.environment
    if getattr(env, "scenario_catalog", ()):
        return _catalog_keys(config, "training")
    keys: list[ScenarioKey] = []
    for seed in range(env.training_seed_start, env.training_seed_end + 1):
        for probability_index in range(1, 3):
            for scenario_index in range(len(env.training_flows)):
                keys.append(
                    ScenarioKey(
                        scenario_index=scenario_index,
                        probability_index=probability_index,
                        seed=seed,
                    )
                )
    return keys


def iter_testing_scenario_keys(config: RuntimeConfig) -> list[ScenarioKey]:
    env = config.environment
    if getattr(env, "scenario_catalog", ()):
        return _catalog_keys(config, "testing")
    keys: list[ScenarioKey] = []
    for seed in range(env.testing_seed_start, env.testing_seed_end + 1):
        for probability_index in range(1, 3):
            for scenario_index in range(len(env.testing_flows)):
                keys.append(
                    ScenarioKey(
                        scenario_index=scenario_index,
                        probability_index=probability_index,
                        seed=seed,
                    )
                )
    return keys


def iter_validation_scenario_keys(config: RuntimeConfig) -> list[ScenarioKey]:
    env = config.environment
    if getattr(env, "scenario_catalog", ()):
        return _catalog_keys(config, "validation")
    seed_start = env.validation_seed_start if env.validation_seed_start is not None else env.testing_seed_start
    seed_end = env.validation_seed_end if env.validation_seed_end is not None else env.testing_seed_end
    keys: list[ScenarioKey] = []
    for seed in range(seed_start, seed_end + 1):
        for probability_index in range(1, 3):
            for scenario_index in range(len(env.validation_flows)):
                keys.append(
                    ScenarioKey(
                        scenario_index=scenario_index,
                        probability_index=probability_index,
                        seed=seed,
                    )
                )
    return keys


__all__ = [
    "ScenarioKey",
    "choose_training_scenario_key",
    "build_training_scenario_schedule",
    "iter_training_scenario_keys",
    "iter_testing_scenario_keys",
    "iter_validation_scenario_keys",
]
