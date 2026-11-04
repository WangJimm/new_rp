from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import RuntimeConfig, TrainingSchedule
from .scenario_loader import (
    ScenarioSpec,
    build_testing_scenario,
    build_training_scenario,
    build_validation_scenario,
)
from .scenario_selection import (
    ScenarioKey,
    choose_training_scenario_key,
    iter_testing_scenario_keys,
    iter_training_scenario_keys,
    iter_validation_scenario_keys,
)


class ExecutionMode(str, Enum):
    VALIDATE_ONLY = "validate_only"
    TRAIN = "train"


@dataclass(frozen=True)
class EpsilonScheduler:
    schedule: TrainingSchedule
    min_epsilon: float

    def value(self, episode: int) -> float:
        if episode < self.schedule.warmup_end:
            return 1.0
        if episode < self.schedule.training_end:
            span = max(1, self.schedule.training_end - self.schedule.warmup_end)
            frac = (episode - self.schedule.warmup_end) / span
            return max(self.min_epsilon, 1.0 - frac * (1.0 - self.min_epsilon))
        return self.min_epsilon


def dijkstra_guidance_probability(config: RuntimeConfig, episode: int) -> float:
    """Return the teacher-action probability within an epsilon exploration step."""
    if not config.edqn_dijkstra_guidance_enabled:
        return 0.0
    start_probability = min(1.0, max(0.0, float(config.edqn_dijkstra_guide_probability)))
    hold_episodes = max(0, int(config.edqn_dijkstra_guide_hold_episodes))
    decay_episodes = max(1, int(config.edqn_dijkstra_guide_decay_episodes))
    if episode < hold_episodes:
        return start_probability
    decay_progress = (episode - hold_episodes) / decay_episodes
    return start_probability * max(0.0, 1.0 - decay_progress)


@dataclass
class TrainingProgress:
    rewards: list[float] = field(default_factory=list)
    rolling_rewards: list[float] = field(default_factory=list)
    scenario_hits: dict[int, int] = field(default_factory=dict)
    best_validation_reward: float = float("-inf")
    last_episode: int = 0

    def record_scenario_hit(self, scenario_index: int) -> None:
        self.scenario_hits[scenario_index] = self.scenario_hits.get(scenario_index, 0) + 1


def build_execution_mode(read_tmp_file: bool) -> ExecutionMode:
    return ExecutionMode.VALIDATE_ONLY if read_tmp_file else ExecutionMode.TRAIN


def resolve_training_scenario(config: RuntimeConfig, key: ScenarioKey | None = None) -> tuple[ScenarioKey, ScenarioSpec]:
    key = key or choose_training_scenario_key(config)
    spec = build_training_scenario(
        config,
        scenario_index=key.scenario_index,
        probability_index=key.probability_index,
        seed=key.seed,
    )
    return key, spec


def resolve_testing_scenarios(config: RuntimeConfig) -> list[tuple[ScenarioKey, ScenarioSpec]]:
    pairs: list[tuple[ScenarioKey, ScenarioSpec]] = []
    for key in iter_testing_scenario_keys(config):
        spec = build_testing_scenario(
            config,
            scenario_index=key.scenario_index,
            probability_index=key.probability_index,
            seed=key.seed,
        )
        pairs.append((key, spec))
    return pairs


def resolve_validation_scenarios(config: RuntimeConfig) -> list[tuple[ScenarioKey, ScenarioSpec]]:
    pairs: list[tuple[ScenarioKey, ScenarioSpec]] = []
    for key in iter_validation_scenario_keys(config):
        spec = build_validation_scenario(
            config,
            scenario_index=key.scenario_index,
            probability_index=key.probability_index,
            seed=key.seed,
        )
        pairs.append((key, spec))
    return pairs


def resolve_training_scenarios(config: RuntimeConfig) -> list[tuple[ScenarioKey, ScenarioSpec]]:
    pairs: list[tuple[ScenarioKey, ScenarioSpec]] = []
    for key in iter_training_scenario_keys(config):
        spec = build_training_scenario(
            config,
            scenario_index=key.scenario_index,
            probability_index=key.probability_index,
            seed=key.seed,
        )
        pairs.append((key, spec))
    return pairs


__all__ = [
    "ExecutionMode",
    "EpsilonScheduler",
    "TrainingProgress",
    "build_execution_mode",
    "resolve_training_scenario",
    "resolve_testing_scenarios",
    "resolve_validation_scenarios",
    "resolve_training_scenarios",
]
