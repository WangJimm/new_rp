from __future__ import annotations

import os
import sys
import time
import json
import pickle
import argparse
import shutil
import copy
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

# Training and evaluation only save plots; never create a Tk/Qt GUI.
import matplotlib
matplotlib.use("Agg", force=True)

import traci
import numpy as np
import torch
from sumolib import checkBinary
from sumolib.net import readNet

THIS_DIR = Path(__file__).resolve().parents[1]
ROUTE_PLANNING_DIR = THIS_DIR
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROUTE_PLANNING_DIR) not in sys.path:
    sys.path.append(str(ROUTE_PLANNING_DIR))

from rp.experiment import (
    RP_VARIANT_CHOICES,
    ENV_CHOICES,
    build_selected_env_config,
    validate_environment_inputs,
    RuntimeConfig,
    TrainingSchedule,
    build_small_env_config,
    build_env_config,
    build_static_env_config,
    resolve_rp_variant_spec,
)
from rp.experiment import (
    AgentLearningError,
    DQNGNNEpisodeRunner,
    EpisodeConfig,
    EpisodeRunInput,
    resolve_ev_ids,
)
from rp.experiment import (
    EMAFeature,
    EpisodeLogger,
    ScenarioKey,
    ScenarioSpec,
    build_testing_scenario,
    build_training_scenario,
    build_training_scenario_schedule,
    choose_training_scenario_key,
    iter_testing_scenario_keys,
)
from rp.sumo import build_load_args as build_shared_load_args
TRAINING_STATE_PICKLE_MAGIC = b"RP_TRAINING_STATE_PICKLE_V1\n"
TENSOR_NDARRAY_MARKER = "__rp_tensor_ndarray_v1__"
PRINT_SAVE_STAGES = False
PRINT_SUMO_SESSION_EVENTS = False
PRINT_RECOVERABLE_EPISODE_FAILURES = False


from rp.experiment import (
    EpsilonScheduler,
    TrainingProgress,
    build_execution_mode,
    dijkstra_guidance_probability,
    resolve_training_scenarios,
    resolve_testing_scenarios,
    resolve_validation_scenarios,
    resolve_training_scenario,
)
from rp.graph import prepare_static
from rp.reporting import simple_plot
from rp.experiment import (
    run_dijkstra_baseline_for_testing_key,
    run_dijkstra_baseline_for_training_key,
    run_dijkstra_baseline_for_validation_key,
)
from DQN.dqngnn_re import DQNGNNAgentRefactored, DQNGNNShapeConfig
from DQN.dqngnn_action_re import DQNGNNActionAgentRefactored, DQNGNNActionShapeConfig
from DQN.gcn_re import GCNGNNAgentRefactored
from DQN.dqn_re import DQNAgentRefactored, DQNShapeConfig
from DQN.edqn import EDQNAgentRefactored, EDQNShapeConfig
from DQN.ppo_rllib import RLLibPPOAgentRefactored, RLLibPPOShapeConfig
from DQN.ppo_torch import TorchPPOAgentRefactored, TorchPPOShapeConfig
from DQN.ppo_route import RoutePPOAgent, RoutePPOShapeConfig
from rp.experiment.reproducibility import (
    add_reproducibility_args, apply_reproducibility_args, seed_everything,
    capture_rng_state, restore_rng_state, write_run_manifest,
)
from DQN.sacd import SACDAgentRefactored, SACDShapeConfig
from tqdm import tqdm



@dataclass
class RefactorRuntimeBundle:
    config: RuntimeConfig
    net: Any
    static_pack: dict[str, Any]
    agent: Any
    rp_variant: str = "proposed"
    agent_type: str | None = None
    edge_feature_mode: str | None = None
    logger: EpisodeLogger | None = None


@dataclass
class TrainingArtifacts:
    rewards: list[float] = field(default_factory=list)
    average_rewards: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    average_costs: list[float] = field(default_factory=list)
    losses: list[float | None] = field(default_factory=list)
    average_losses: list[float] = field(default_factory=list)
    loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    reward_records: list[dict[str, Any]] = field(default_factory=list)
    failed_episode_records: list[dict[str, Any]] = field(default_factory=list)
    scenario_schedule_seed: int | None = None
    rp_variant: str | None = None
    agent_type: str | None = None
    edge_feature_mode: str | None = None
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    cost_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    reward_per_ev: dict[Any, Any] = field(default_factory=dict)
    rd_cnt: list[int] = field(default_factory=lambda: [0 for _ in range(10)])
    best_validation_reward: float = float("inf")
    best_better_ratio: float = float("-inf")
    best_better_ratio_cost: float = float("inf")
    best_arrival_rate: float = float("-inf")
    best_arrival_cost: float = float("inf")
    best_arrival_episode: int | None = None
    bad_validation_count: int = 0
    validation_records: list[dict[str, Any]] = field(default_factory=list)
    warm_start_checkpoint: str | None = None
    warm_start_checkpoint_episode: int | None = None

@dataclass(frozen=True)
class TrainingEpisodeResult:
    episode: int
    scenario_key: ScenarioKey
    scenario: ScenarioSpec
    load_args: list[str]
    epsilon: float
    dijkstra_guidance_probability: float
    exploration_decision_count: int
    dijkstra_guidance_available_count: int
    dijkstra_guided_decision_count: int
    train_enabled: bool
    memory_before: int
    memory_after: int
    total_costs: list[float]
    episode_cost_minutes: float
    ev_costs_minutes: list[float]
    penalized_episode_cost_minutes: float
    penalized_ev_costs_minutes: list[float]
    success_count: int
    failure_count: int
    success_rate: float
    failure_reasons: dict[str, str | None]
    simulation_steps: int
    transition_steps: int
    truncated_by_max_steps: bool
    update_attempts: int
    learn_calls: int
    gradient_steps: int
    update_metrics: dict[str, float]
    last_loss: float | None
    mean_loss: float | None
    teleport_start_count: int
    teleport_end_count: int
    teleport_started_ids: list[str]
    teleport_ended_ids: list[str]
    ev_teleport_started_ids: list[str]
    ev_teleport_ended_ids: list[str]
    bottleneck_enabled: bool
    bottleneck_triggered: bool
    bottleneck_active_steps: int
    bottleneck_events: list[dict[str, Any]]

    @property
    def reward_sum(self) -> float:
        return sum(self.total_costs)

    @property
    def cost_sum(self) -> float:
        return self.episode_cost_minutes

    @property
    def memory_delta(self) -> int:
        return self.memory_after - self.memory_before


class RecoverableEpisodeError(Exception):
    def __init__(
        self,
        message: str,
        *,
        episode: int,
        scenario_key: ScenarioKey,
        load_args: list[str],
        original_error: BaseException,
    ) -> None:
        super().__init__(message)
        self.episode = episode
        self.scenario_key = scenario_key
        self.load_args = load_args
        self.original_error = original_error
    

class TrainingStateResumeUnavailable(RuntimeError):
    pass


BACKUP_GENERATIONS = 5


def bootstrap_sumo_environment(config: RuntimeConfig) -> None:
    home = config.paths.sumo_home
    os.environ["SUMO_HOME"] = str(home)
    tools = home / "tools"
    if tools.is_dir() and str(tools) not in sys.path:
        sys.path.append(str(tools))
    bin_dirs = [home / "bin", home / "Scripts", Path(sys.prefix) / "bin", Path(sys.prefix) / "Scripts"]
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for folder in reversed(bin_dirs):
        if folder.is_dir() and str(folder) not in path_parts:
            path_parts.insert(0, str(folder))
    os.environ["PATH"] = os.pathsep.join(path_parts)
    suffix = ".exe" if os.name == "nt" else ""
    for name, variable in (("sumo", "SUMO_BINARY"), ("sumo-gui", "GUISIM_BINARY")):
        found = next((folder / (name + suffix) for folder in bin_dirs
                      if (folder / (name + suffix)).is_file()), None)
        if found is not None:
            os.environ[variable] = str(found)


def build_sumo_command(
    config: RuntimeConfig,
    error_log: Path | None = None,
    message_log: Path | None = None,
) -> list[str]:
    binary_name = "sumo-gui" if config.use_gui else "sumo"
    cmd = [
        checkBinary(binary_name),
        "-c",
        str(config.paths.config_file),
        "-W",
        "--step-length",
        str(config.step_length),
        "--no-step-log",
    ]
    if error_log is not None:
        cmd.extend(["--error-log", str(error_log)])
    if message_log is not None:
        cmd.extend(["--message-log", str(message_log)])
    return cmd


def sumo_log_root(config: RuntimeConfig) -> Path:
    root = THIS_DIR / "logs"
    return root if config.environment.name == "env" else root / config.environment.name


def start_sumo_session(config: RuntimeConfig) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = sumo_log_root(config) / f"run_{stamp}"
    os.makedirs(log_dir, exist_ok=True)
    error_log = log_dir / "sumo_error.log"
    msg_log = log_dir / "sumo_msg.log"
    cmd = build_sumo_command(config, error_log=error_log, message_log=msg_log)
    traci.start(cmd, stdout=open(os.devnull, "w"))
    return {
        "cmd": cmd,
        "log_dir": log_dir,
        "error_log": error_log,
        "msg_log": msg_log,
    }


def stop_sumo_session() -> None:
    try:
        traci.close()
    except Exception:
        pass


def restart_sumo_session(config: RuntimeConfig, reason: str | None = None) -> dict[str, Any]:
    stop_sumo_session()
    time.sleep(0.2)
    session_info = start_sumo_session(config)
    if reason and PRINT_SUMO_SESSION_EVENTS:
        print(f"[sumo restart] {reason}", flush=True)
        print(f"[sumo restart] log dir: {session_info['log_dir']}", flush=True)
    return session_info




def log_save_stage(stage: str, episode: int | None = None, path: Path | None = None) -> None:
    if not PRINT_SAVE_STAGES:
        return
    parts = ["[save stage]", stage]
    if episode is not None:
        parts.append(f"episode={episode}")
    if path is not None:
        parts.append(f"path={path}")
    print(" ".join(parts), flush=True)


def close_sumo_before_save(enabled: bool, reason: str) -> bool:
    if not enabled:
        return False
    stop_sumo_session()
    if PRINT_SUMO_SESSION_EVENTS:
        print(f"[sumo close] {reason}", flush=True)
    return True


def restart_sumo_after_save(config: RuntimeConfig, enabled: bool, reason: str) -> None:
    if enabled:
        log_save_stage("sumo restart start")
        restart_sumo_session(config, reason=reason)
        log_save_stage("sumo restart done")

def describe_runtime(config: RuntimeConfig) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_start, validation_interval = resolve_training_validation_schedule(config)
    return (
        f"env={config.environment.name}, "
        f"data={config.paths.run_dir}, "
        f"net={config.paths.net_file.name}, "
        f"episodes={config.schedule.num_episodes}, "
        f"epsilon_warmup_end={config.schedule.warmup_end}, "
        f"epsilon_decay_end={config.schedule.training_end}, "
        f"validation_start_episode={validation_start}, "
        f"validation_interval={validation_interval}, "
        f"device={device}, "
        f"use_gnn={config.use_gnn}, "
        f"revisit_penalty={config.revisit_penalty if config.revisit_penalty_enabled else 'disabled'}, "
        f"edqn_dijkstra_guidance={config.edqn_dijkstra_guidance_enabled}, "
        f"guide_probability={config.edqn_dijkstra_guide_probability}, "
        f"guide_hold={config.edqn_dijkstra_guide_hold_episodes}, "
        f"guide_decay={config.edqn_dijkstra_guide_decay_episodes}, "
        f"bottleneck={config.enable_bottleneck}, "
        f"bottleneck_edges={list(config.bottleneck_candidate_edges)}"
    )


def summarize_static_graph(config: RuntimeConfig) -> dict[str, int]:
    net = readNet(str(config.paths.net_file), withInternal=True)
    static_pack = prepare_static(net, traci_api=traci, prefer_controlled_tls=config.environment.name == "new_env")
    return {
        "num_edges": len(static_pack["edge_ids"]),
        "num_nodes": len(static_pack["nid2idx"]),
        "max_neighbors": max(len(v) for v in static_pack["adj"].values()),
    }


def build_load_args(config: RuntimeConfig, scenario: ScenarioSpec) -> list[str]:
    return build_shared_load_args(config, scenario)


def build_runner_input(
    scenario: ScenarioSpec,
    evaluate: bool,
    train: bool,
    agent_path: list[list[str]] | None = None,
    debug: bool = False,
    dijkstra_guidance_probability: float = 0.0,
) -> EpisodeRunInput:
    return EpisodeRunInput(
        ev_ids=scenario.ev_ids,
        goal_nodes=scenario.goal_nodes,
        evaluate=evaluate,
        train=train,
        agent_path=agent_path,
        debug=debug,
        dijkstra_guidance_probability=dijkstra_guidance_probability,
    )


def build_episode_runner(
    traci_module: Any,
    net,
    static_pack: dict[str, Any],
    agent: Any,
    runtime_config: RuntimeConfig,
    logger: EpisodeLogger | None = None,
) -> DQNGNNEpisodeRunner:
    half_life_sec = 5.0
    return DQNGNNEpisodeRunner(
        traci=traci_module,
        net=net,
        static_pack=static_pack,
        agent=agent,
        travel_time_ema=EMAFeature(half_life_sec=half_life_sec),
        mean_speed_ema=EMAFeature(half_life_sec=half_life_sec),
        occupancy_ema=EMAFeature(half_life_sec=half_life_sec),
        config=EpisodeConfig(
            max_decisions_per_ev=runtime_config.max_decisions_per_ev,
            max_simulation_steps=runtime_config.max_simulation_steps,
            decision_distance_threshold=runtime_config.decision_distance_threshold,
            goal_arrive_distance=runtime_config.goal_arrive_distance,
            # Keep the trusted eDQN objective identical across all baselines:
            # elapsed travel time plus the same fixed terminal terms.
            success_bonus=runtime_config.success_bonus,
            uturn_penalty=runtime_config.uturn_penalty,
            dead_end_penalty=runtime_config.dead_end_penalty,
            progress_reward_enabled=False,
            traffic_penalty_enabled=False,
            revisit_penalty_enabled=runtime_config.revisit_penalty_enabled,
            revisit_penalty=runtime_config.revisit_penalty,
            potential_shaping_enabled=runtime_config.potential_shaping_enabled,
            potential_shaping_gamma=runtime_config.potential_shaping_gamma,
            potential_distance_scale=runtime_config.potential_distance_scale,
            simulation_step_sec=runtime_config.step_length,
            memory_learning_threshold=runtime_config.memory_learning_threshold,
            memory_push_learning_threshold=runtime_config.memory_push_learning_threshold,
            enable_bottleneck=runtime_config.enable_bottleneck,
            bottleneck_start_time_sec=runtime_config.bottleneck_start_time_sec,
            bottleneck_end_time_sec=runtime_config.bottleneck_end_time_sec,
            bottleneck_speed_factor=runtime_config.bottleneck_speed_factor,
            bottleneck_candidate_edges=tuple(runtime_config.bottleneck_candidate_edges),
        ),
        logger=logger,
    )


def preview_episode_execution(config: RuntimeConfig, scenario: ScenarioSpec) -> dict[str, Any]:
    net = readNet(str(config.paths.net_file), withInternal=True)
    static_pack = prepare_static(net, traci_api=traci, prefer_controlled_tls=config.environment.name == "new_env")
    return {
        "load_args": build_load_args(config, scenario),
        "ev_count": len(scenario.ev_ids),
        "goal_count": len(scenario.goal_nodes),
        "max_neighbors": max(len(v) for v in static_pack["adj"].values()),
    }


def load_runtime_bundle(config: RuntimeConfig, logger: EpisodeLogger | None = None) -> RefactorRuntimeBundle:
    net = readNet(str(config.paths.net_file), withInternal=True)
    static_pack = prepare_static(net, traci_api=traci, prefer_controlled_tls=config.environment.name == "new_env")

    variant_spec = resolve_rp_variant_spec(config.rp_variant)
    agent_type = variant_spec.agent_type
    edge_feature_mode = variant_spec.edge_feature_mode

    if agent_type == "legacy":
        agent = build_agent_from_static_pack(static_pack, use_gnn=config.use_gnn)
    elif agent_type == "gine":
        agent = build_refactored_dqngnn_agent_from_static_pack(
            static_pack,
            edge_feature_mode=edge_feature_mode,
        )
    elif agent_type == "gcn":
        agent = build_refactored_gcn_agent_from_static_pack(static_pack)
    elif agent_type == "dqn":
        agent = build_refactored_dqn_agent_from_static_pack(static_pack)
    elif agent_type == "edqn":
        agent = build_refactored_edqn_agent_from_static_pack(static_pack)
    elif agent_type == "ppo_rllib":
        agent = build_refactored_ppo_rllib_agent_from_static_pack(static_pack)
    elif agent_type == "ppo_torch":
        agent = build_refactored_ppo_torch_agent_from_static_pack(static_pack)
    elif agent_type == "ppo_route":
        shape = RoutePPOShapeConfig(len(static_pack["nid2idx"]), len(static_pack["edge_ids"]),
                                    max(len(v) for v in static_pack["adj"].values()))
        agent = RoutePPOAgent(shape)
        agent.configure_route_context(net, static_pack, config.max_decisions_per_ev)
    elif agent_type == "sacd":
        agent = build_refactored_sacd_agent_from_static_pack(
            static_pack,
            use_action_features=config.sacd_use_action_features,
            replay_alpha=config.sacd_replay_alpha,
            target_entropy_ratio=config.sacd_target_entropy_ratio,
            use_epsilon_exploration=config.sacd_use_epsilon_exploration,
        )
    elif agent_type == "gine_action":
        agent = build_refactored_dqngnn_action_agent_from_static_pack(
            static_pack,
            edge_feature_mode=edge_feature_mode,
        )
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    agent.agent_type = agent_type
    if agent_type not in {"gine", "gine_action"}:
        agent.edge_feature_mode = None
    return RefactorRuntimeBundle(
        config=config,
        net=net,
        static_pack=static_pack,
        agent=agent,
        rp_variant=variant_spec.name,
        agent_type=agent_type,
        edge_feature_mode=getattr(agent, "edge_feature_mode", None),
        logger=logger,
    )


def build_agent_from_static_pack(static_pack: dict[str, Any], use_gnn: bool = True, writer_name: str = "refactor") -> Any:
    try:
        from DQN.dqn import DQN
    except ModuleNotFoundError as exc:
        raise RuntimeError("Legacy DQN agent is unavailable in this checkout; use rp_variant='dqn_baseline' instead.") from exc

    node_in = static_pack["data_static"].x.size(1)
    edge_in = 3 + static_pack["static_edge_attr_norm"].size(1)
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    return DQN(
        len(static_pack["nid2idx"]) * 2,
        max_neighbors,
        node_in + 4,
        edge_in,
        use_gnn=use_gnn,
        writer_name=writer_name,
        num_nodes=len(static_pack["nid2idx"]),
    )


def build_refactored_dqngnn_agent_from_static_pack(
    static_pack: dict[str, Any],
    edge_feature_mode: str = "all",
) -> DQNGNNAgentRefactored:
    node_in = static_pack["data_static"].x.size(1)
    edge_in = 3 + static_pack["static_edge_attr_norm"].size(1)
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    shape = DQNGNNShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
        node_in=node_in + 4,
        edge_in=edge_in,
    )
    return DQNGNNAgentRefactored(shape, edge_feature_mode=edge_feature_mode)


def build_refactored_dqngnn_action_agent_from_static_pack(
    static_pack: dict[str, Any],
    edge_feature_mode: str = "all",
) -> DQNGNNActionAgentRefactored:
    node_in = static_pack["data_static"].x.size(1)
    edge_in = 3 + static_pack["static_edge_attr_norm"].size(1)
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    shape = DQNGNNActionShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
        node_in=node_in + 4,
        edge_in=edge_in,
    )
    return DQNGNNActionAgentRefactored(shape, edge_feature_mode=edge_feature_mode)


def build_refactored_gcn_agent_from_static_pack(static_pack: dict[str, Any]) -> GCNGNNAgentRefactored:
    node_in = static_pack["data_static"].x.size(1)
    edge_in = 3 + static_pack["static_edge_attr_norm"].size(1)
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    shape = DQNGNNShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
        node_in=node_in + 4,
        edge_in=edge_in,
    )
    return GCNGNNAgentRefactored(shape)


def build_refactored_dqn_agent_from_static_pack(static_pack: dict[str, Any]) -> DQNAgentRefactored:
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    shape = DQNShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
    )
    return DQNAgentRefactored(shape)


def build_refactored_edqn_agent_from_static_pack(static_pack: dict[str, Any]) -> EDQNAgentRefactored:
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    shape = EDQNShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
    )
    return EDQNAgentRefactored(shape)


def build_refactored_ppo_rllib_agent_from_static_pack(static_pack: dict[str, Any]) -> RLLibPPOAgentRefactored:
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    shape = RLLibPPOShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
    )
    return RLLibPPOAgentRefactored(
        shape,
        lr=1e-4,
        train_batch_size=4096,
        sgd_minibatch_size=512,
        entropy_coeff=0.01,
        clip_param=0.2,
        gae_lambda=0.95,
    )


def build_refactored_ppo_torch_agent_from_static_pack(static_pack: dict[str, Any]) -> TorchPPOAgentRefactored:
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    shape = TorchPPOShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
    )
    return TorchPPOAgentRefactored(
        shape,
        lr=1e-4,
        train_batch_size=4096,
        sgd_minibatch_size=512,
        num_sgd_iter=10,
        entropy_coeff=0.01,
        clip_param=0.2,
        gae_lambda=0.95,
    )


def build_refactored_sacd_agent_from_static_pack(
    static_pack: dict[str, Any],
    *,
    use_action_features: bool = False,
    replay_alpha: float = 0.0,
    target_entropy_ratio: float = 0.30,
    use_epsilon_exploration: bool = False,
) -> SACDAgentRefactored:
    max_neighbors = max(len(v) for v in static_pack["adj"].values())
    edge_feature_dim = 3 + int(static_pack["static_edge_attr_norm"].size(1))
    action_feature_dim = edge_feature_dim + 7 if use_action_features else 0
    shape = SACDShapeConfig(
        num_nodes=len(static_pack["nid2idx"]),
        max_neighbors=max_neighbors,
        action_feature_dim=action_feature_dim,
    )
    return SACDAgentRefactored(
        shape,
        replay_alpha=replay_alpha,
        target_entropy_ratio=target_entropy_ratio,
        use_epsilon_exploration=use_epsilon_exploration,
    )


def maybe_load_checkpoint(agent: Any, checkpoint_path: Path | None, validate_only: bool = False) -> int:
    if checkpoint_path is None:
        return 0
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return agent.load_full_model(filename=str(checkpoint_path), buffer_name=None)


def create_training_artifacts() -> TrainingArtifacts:
    return TrainingArtifacts()


def normalize_training_artifacts(artifacts: TrainingArtifacts) -> TrainingArtifacts:
    defaults = TrainingArtifacts()
    for field_name, default_value in vars(defaults).items():
        if not hasattr(artifacts, field_name):
            setattr(artifacts, field_name, default_value)
    return artifacts


def resolve_variant_output_dir(config: RuntimeConfig, output_dir: Path | None = None) -> Path:
    if output_dir is not None:
        return output_dir
    root = THIS_DIR / "outputs"
    if config.environment.name != "env":
        root = root / config.environment.name
    return root / config.rp_variant


def default_training_state_path(config: RuntimeConfig, output_dir: Path | None = None) -> Path:
    return resolve_variant_output_dir(config, output_dir) / "training_state_latest.pt"



def environment_matches_config(config: RuntimeConfig, payload: dict[str, Any]) -> bool:
    saved_name = payload.get("environment_name", "env")
    current_name = getattr(getattr(config, "environment", None), "name", "env")
    return saved_name == current_name


def default_evaluation_checkpoint(config: RuntimeConfig) -> Path:
    output = resolve_variant_output_dir(config)
    if config.environment.name == "env":
        return output / "dqngnn_final.pt"
    prefix = resolve_rp_variant_spec(config.rp_variant).agent_type
    return output / f"{prefix}_best_100pct_arrival_lowest_cost.pt"


def dijkstra_guidance_settings(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "enabled": bool(config.edqn_dijkstra_guidance_enabled),
        "probability": float(config.edqn_dijkstra_guide_probability),
        "hold_episodes": int(config.edqn_dijkstra_guide_hold_episodes),
        "decay_episodes": int(config.edqn_dijkstra_guide_decay_episodes),
    }


def dijkstra_guidance_settings_match(config: RuntimeConfig, payload: dict[str, Any]) -> bool:
    saved = payload.get("dijkstra_guidance")
    if saved is None:
        return not config.edqn_dijkstra_guidance_enabled
    return saved == dijkstra_guidance_settings(config)


def training_schedule_settings(config: RuntimeConfig) -> dict[str, int]:
    return {
        "num_episodes": int(config.schedule.num_episodes),
        "epsilon_warmup_end": int(config.schedule.warmup_end),
        "epsilon_decay_end": int(config.schedule.training_end),
    }


def resolve_training_validation_schedule(config: RuntimeConfig) -> tuple[int, int]:
    """Return the first one-based validation episode and interval.

    Omitted overrides preserve the original global-multiple schedule.
    An explicit start anchors subsequent validations to that episode.
    """
    interval = getattr(config, "validation_interval", None)
    if interval is None:
        interval = max(1, config.schedule.num_episodes // 20)
    if interval <= 0:
        raise ValueError("--validation-interval must be positive.")
    start = getattr(config, "validation_start_episode", None)
    if start is None:
        minimum = max(1, config.schedule.training_end)
        start = ((minimum + interval - 1) // interval) * interval
    if start <= 0:
        raise ValueError("--validation-start-episode must be positive.")
    return int(start), int(interval)


def validation_schedule_settings(config: RuntimeConfig) -> dict[str, int]:
    start, interval = resolve_training_validation_schedule(config)
    return {"start_episode": start, "interval": interval}


def training_validation_due(config: RuntimeConfig, episode: int) -> bool:
    start, interval = resolve_training_validation_schedule(config)
    completed = episode + 1
    return completed >= start and (completed - start) % interval == 0


def training_schedule_settings_match(config: RuntimeConfig, payload: dict[str, Any]) -> bool:
    saved = payload.get("training_schedule")
    if saved is None:
        legacy = {
            "num_episodes": int(config.schedule.num_episodes),
            "epsilon_warmup_end": int(config.schedule.num_episodes * config.schedule.warmup_ratio),
            "epsilon_decay_end": int(config.schedule.num_episodes * config.schedule.training_ratio),
        }
        return training_schedule_settings(config) == legacy
    return saved == training_schedule_settings(config)


def training_completed_marker_path(output_dir: Path) -> Path:
    return output_dir / "training_completed.json"


def save_training_completed_marker(
    *,
    output_dir: Path,
    config: RuntimeConfig,
    artifacts: TrainingArtifacts,
    next_episode: int,
) -> None:
    log_save_stage("completed_marker start", episode=next_episode, path=training_completed_marker_path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = training_completed_marker_path(output_dir)
    tmp_path = marker_path.with_name(f"{marker_path.stem}.{os.getpid()}.tmp")
    payload = {
        "completed": True,
        "environment_name": config.environment.name,
        "completed_at": time.strftime("%Y%m%d-%H%M%S"),
        "rp_variant": config.rp_variant,
        "next_episode": int(next_episode),
        "num_episodes": int(config.schedule.num_episodes),
        "training_schedule": training_schedule_settings(config),
        "validation_schedule": validation_schedule_settings(config),
        "num_rewards": len(artifacts.rewards),
        "last_average_reward": artifacts.average_rewards[-1] if artifacts.average_rewards else None,
        "last_average_cost": artifacts.average_costs[-1] if artifacts.average_costs else None,
        "arrival_gate_required": 1.0,
        "best_100pct_arrival_lowest_cost": artifacts.best_validation_reward,
        "best_100pct_arrival_better_ratio": artifacts.best_better_ratio,
        "revisit_penalty_enabled": config.revisit_penalty_enabled,
        "revisit_penalty": config.revisit_penalty,
        "dijkstra_guidance": dijkstra_guidance_settings(config),
        "warm_start_checkpoint": artifacts.warm_start_checkpoint,
        "warm_start_checkpoint_episode": artifacts.warm_start_checkpoint_episode,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        fsync_file_object(f)
    log_save_stage("completed_marker replace start", episode=next_episode, path=marker_path)
    safe_replace_file(tmp_path, marker_path)
    log_save_stage("completed_marker done", episode=next_episode, path=marker_path)


def resume_state_already_completed(config: RuntimeConfig, state_path: Path) -> bool:
    if not state_path.exists():
        return False
    try:
        with open(state_path, "rb") as f:
            if f.read(len(TRAINING_STATE_PICKLE_MAGIC)) == TRAINING_STATE_PICKLE_MAGIC:
                payload = pickle.load(f)
            else:
                try:
                    payload = torch.load(state_path, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(state_path, map_location="cpu")
    except Exception:
        return False
    if not environment_matches_config(config, payload):
        return False
    if payload.get("rp_variant") != config.rp_variant:
        return False
    if not dijkstra_guidance_settings_match(config, payload):
        return False
    if not training_schedule_settings_match(config, payload):
        return False
    return int(payload.get("next_episode", 0)) >= int(config.schedule.num_episodes)


def completed_marker_matches_config(config: RuntimeConfig, output_dir: Path) -> bool:
    marker_path = training_completed_marker_path(output_dir)
    if not marker_path.exists():
        return False
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[warn] could not read completed training marker {marker_path}: {exc}")
        return False
    if not environment_matches_config(config, payload):
        return False
    if payload.get("rp_variant") != config.rp_variant:
        return False
    if not dijkstra_guidance_settings_match(config, payload):
        return False
    if not training_schedule_settings_match(config, payload):
        return False
    return bool(payload.get("completed")) and int(payload.get("next_episode", 0)) >= int(
        config.schedule.num_episodes
    )


def make_autosave_safe(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy().copy()
        return {TENSOR_NDARRAY_MARKER: True, "array": array}
    if isinstance(value, np.ndarray):
        copied = value.copy()
        if copied.dtype == object:
            for index in np.ndindex(copied.shape):
                copied[index] = make_autosave_safe(copied[index], _seen)
            return copied
        if copied.dtype.fields:
            for field_name, (field_dtype, _) in copied.dtype.fields.items():
                if field_dtype == object:
                    for index in np.ndindex(copied.shape):
                        copied[field_name][index] = make_autosave_safe(copied[field_name][index], _seen)
            return copied
        return copied
    if isinstance(value, dict):
        return {key: make_autosave_safe(item, _seen) for key, item in value.items()}
    if isinstance(value, list):
        return [make_autosave_safe(item, _seen) for item in value]
    if isinstance(value, tuple):
        items = [make_autosave_safe(item, _seen) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, deque):
        return deque((make_autosave_safe(item, _seen) for item in value), maxlen=value.maxlen)
    if hasattr(value, "to") and value.__class__.__module__.startswith("torch_geometric"):
        try:
            value = value.to("cpu")
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        object_id = id(value)
        if object_id in _seen:
            return value
        _seen.add(object_id)
        try:
            copied_obj = copy.copy(value)
        except Exception:
            _seen.remove(object_id)
            return value
        for attr_name, attr_value in vars(value).items():
            try:
                setattr(copied_obj, attr_name, make_autosave_safe(attr_value, _seen))
            except Exception:
                pass
        _seen.remove(object_id)
        return copied_obj
    return value


def restore_autosave_tensors(value: Any, _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    if isinstance(value, dict):
        if value.get(TENSOR_NDARRAY_MARKER) is True and "array" in value:
            return torch.from_numpy(value["array"])
        return {key: restore_autosave_tensors(item, _seen) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        copied = value.copy()
        if copied.dtype == object:
            for index in np.ndindex(copied.shape):
                copied[index] = restore_autosave_tensors(copied[index], _seen)
            return copied
        if copied.dtype.fields:
            for field_name, (field_dtype, _) in copied.dtype.fields.items():
                if field_dtype == object:
                    for index in np.ndindex(copied.shape):
                        copied[field_name][index] = restore_autosave_tensors(copied[field_name][index], _seen)
            return copied
        return copied
    if isinstance(value, list):
        return [restore_autosave_tensors(item, _seen) for item in value]
    if isinstance(value, tuple):
        items = [restore_autosave_tensors(item, _seen) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, deque):
        return deque((restore_autosave_tensors(item, _seen) for item in value), maxlen=value.maxlen)
    if hasattr(value, "__dict__"):
        object_id = id(value)
        if object_id in _seen:
            return value
        _seen.add(object_id)
        try:
            copied_obj = copy.copy(value)
        except Exception:
            _seen.remove(object_id)
            return value
        for attr_name, attr_value in vars(value).items():
            try:
                setattr(copied_obj, attr_name, restore_autosave_tensors(attr_value, _seen))
            except Exception:
                pass
        _seen.remove(object_id)
        return copied_obj
    return value


def replay_memory_for_autosave(agent: Any) -> Any:
    memory = getattr(agent, "memory", None)
    if memory is None:
        return None
    if getattr(agent, "use_gnn", False):
        return memory
    replay_array = getattr(memory, "memory", None)
    if not isinstance(replay_array, np.ndarray) or not replay_array.dtype.names:
        return memory
    try:
        copied_memory = copy.copy(memory)
        copied_memory.memory = replay_array.copy()
        for field_name in ("data_t", "next_data_t", "bag", "next_bag"):
            if field_name in copied_memory.memory.dtype.names:
                copied_memory.memory[field_name] = None
        return copied_memory
    except Exception:
        return memory


def _agent_training_state(agent: Any) -> dict[str, Any]:
    if hasattr(agent, "get_training_state"):
        return agent.get_training_state()
    networks = getattr(agent, "networks", None)
    state: dict[str, Any] = {
        "optimizer": agent.optimizer.state_dict() if hasattr(agent, "optimizer") else None,
        "memory": replay_memory_for_autosave(agent),
        "learn_step_counter": getattr(agent, "learn_step_counter", 0),
        "epsilon": getattr(agent, "EPSILON", None),
    }
    if networks is None:
        return state
    if hasattr(networks, "eval_q_net"):
        state["eval_q_net"] = networks.eval_q_net.state_dict()
    if hasattr(networks, "target_q_net"):
        state["target_q_net"] = networks.target_q_net.state_dict()
    if hasattr(networks, "eval_encoder"):
        state["eval_encoder"] = networks.eval_encoder.state_dict()
    if hasattr(networks, "target_encoder"):
        state["target_encoder"] = networks.target_encoder.state_dict()
    return state


def move_optimizer_state_to_device(optimizer: Any, device: Any) -> None:
    if device is None:
        return
    for optimizer_state in optimizer.state.values():
        for key, value in list(optimizer_state.items()):
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)


def _load_agent_training_state(agent: Any, state: dict[str, Any]) -> None:
    if hasattr(agent, "load_training_state"):
        agent.load_training_state(state)
        return
    networks = getattr(agent, "networks", None)
    if networks is not None:
        if "eval_q_net" in state and hasattr(networks, "eval_q_net"):
            networks.eval_q_net.load_state_dict(state["eval_q_net"])
        if "target_q_net" in state and hasattr(networks, "target_q_net"):
            networks.target_q_net.load_state_dict(state["target_q_net"])
        if "eval_encoder" in state and hasattr(networks, "eval_encoder"):
            networks.eval_encoder.load_state_dict(state["eval_encoder"])
        if "target_encoder" in state and hasattr(networks, "target_encoder"):
            networks.target_encoder.load_state_dict(state["target_encoder"])
    if state.get("optimizer") is not None and hasattr(agent, "optimizer"):
        agent.optimizer.load_state_dict(state["optimizer"])
        move_optimizer_state_to_device(agent.optimizer, getattr(agent, "device", None))
    if state.get("memory") is not None:
        agent.memory = state["memory"]
    if "learn_step_counter" in state:
        agent.learn_step_counter = int(state["learn_step_counter"])
    if state.get("epsilon") is not None:
        agent.EPSILON = float(state["epsilon"])


def fsync_file_object(file_obj: Any) -> None:
    file_obj.flush()
    os.fsync(file_obj.fileno())


def safe_replace_file(tmp_path: Path, output_path: Path) -> None:
    tmp_path.replace(output_path)


def backup_generation_path(path: Path, generation: int) -> Path:
    if generation <= 1:
        return path.with_suffix(path.suffix + ".bak")
    return path.with_suffix(path.suffix + f".bak{generation}")


def backup_generation_paths(path: Path) -> list[Path]:
    return [backup_generation_path(path, generation) for generation in range(1, BACKUP_GENERATIONS + 1)]


def rotate_backup_generations(path: Path) -> None:
    backup_paths = backup_generation_paths(path)
    for src, dst in reversed(list(zip(backup_paths[:-1], backup_paths[1:]))):
        if src.exists():
            try:
                src.replace(dst)
            except OSError as exc:
                print(f"[warn] could not rotate backup {src} -> {dst}: {exc}")


def copy_file_fsynced(src_path: Path, dst_tmp_path: Path) -> None:
    with open(src_path, "rb") as src, open(dst_tmp_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        fsync_file_object(dst)


def json_is_readable(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def save_training_state(
    *,
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
    progress: TrainingProgress,
    scenario_schedule: list[ScenarioKey],
    scenario_schedule_seed: int,
    next_episode: int,
    output_path: Path,
    verify_after_save: bool = False,
) -> None:
    log_save_stage("training_state payload build start", episode=next_episode - 1, path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "environment_name": bundle.config.environment.name,
        "next_episode": int(next_episode),
        "rp_variant": bundle.rp_variant,
        "agent_type": bundle.agent_type,
        "edge_feature_mode": bundle.edge_feature_mode,
        "dijkstra_guidance": dijkstra_guidance_settings(bundle.config),
        "training_schedule": training_schedule_settings(bundle.config),
        "validation_schedule": validation_schedule_settings(bundle.config),
        "scenario_schedule_seed": scenario_schedule_seed,
        "scenario_schedule": scenario_schedule,
        "artifacts": artifacts,
        "progress": progress,
        "agent_state": _agent_training_state(bundle.agent),
        "rng_state": capture_rng_state(),
        "reproducibility": {"training_seed": bundle.config.training_seed,
                            "deterministic": bundle.config.deterministic,
                            "max_decisions_per_ev": bundle.config.max_decisions_per_ev,
                            "max_simulation_steps": bundle.config.max_simulation_steps},
        "saved_at": time.strftime("%Y%m%d-%H%M%S"),
    }
    log_save_stage("training_state payload build done", episode=next_episode - 1, path=output_path)
    log_save_stage("training_state cpu copy start", episode=next_episode - 1, path=output_path)
    payload = make_autosave_safe(payload)
    log_save_stage("training_state cpu copy done", episode=next_episode - 1, path=output_path)
    tmp_path = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        log_save_stage("training_state pickle.dump start", episode=next_episode - 1, path=tmp_path)
        f.write(TRAINING_STATE_PICKLE_MAGIC)
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        fsync_file_object(f)
        log_save_stage("training_state pickle.dump done", episode=next_episode - 1, path=tmp_path)
    if verify_after_save:
        log_save_stage("training_state verify tmp start", episode=next_episode - 1, path=tmp_path)
        load_training_state_payload(bundle.agent, tmp_path)
        log_save_stage("training_state verify tmp done", episode=next_episode - 1, path=tmp_path)
    if output_path.exists():
        if verify_after_save:
            try:
                log_save_stage("training_state verify current start", episode=next_episode - 1, path=output_path)
                load_training_state_payload(bundle.agent, output_path)
                log_save_stage("training_state verify current done", episode=next_episode - 1, path=output_path)
            except Exception as exc:
                quarantine_bad_training_state(output_path, exc)
            else:
                backup_path = training_state_backup_path(output_path)
                backup_tmp_path = backup_path.with_name(f"{backup_path.name}.{os.getpid()}.tmp")
                copy_file_fsynced(output_path, backup_tmp_path)
                log_save_stage("training_state verify backup start", episode=next_episode - 1, path=backup_tmp_path)
                load_training_state_payload(bundle.agent, backup_tmp_path)
                log_save_stage("training_state verify backup done", episode=next_episode - 1, path=backup_tmp_path)
                rotate_backup_generations(output_path)
                safe_replace_file(backup_tmp_path, backup_path)
        else:
            backup_path = training_state_backup_path(output_path)
            rotate_backup_generations(output_path)
            try:
                output_path.replace(backup_path)
            except OSError as exc:
                print(f"[warn] could not move previous training state to backup {backup_path}: {exc}")
    log_save_stage("training_state replace start", episode=next_episode - 1, path=output_path)
    safe_replace_file(tmp_path, output_path)
    log_save_stage("training_state done", episode=next_episode - 1, path=output_path)


def training_state_backup_path(state_path: Path) -> Path:
    return backup_generation_path(state_path, 1)


def quarantine_bad_training_state(state_path: Path, exc: BaseException) -> Path | None:
    if not state_path.exists():
        return None
    corrupt_path = state_path.with_name(f"{state_path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        state_path.replace(corrupt_path)
    except OSError as move_exc:
        print(f"[warn] could not quarantine unreadable training state {state_path}: {move_exc}")
        return None
    print(f"[warn] quarantined unreadable training state {state_path} -> {corrupt_path}: {exc}")
    return corrupt_path


def load_training_state_payload(agent: Any, state_path: Path) -> dict[str, Any]:
    with open(state_path, "rb") as f:
        if f.read(len(TRAINING_STATE_PICKLE_MAGIC)) == TRAINING_STATE_PICKLE_MAGIC:
            return restore_autosave_tensors(pickle.load(f))
    try:
        return torch.load(state_path, map_location=getattr(agent, "device", None), weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location=getattr(agent, "device", None))

def load_training_state(bundle: RefactorRuntimeBundle, state_path: Path) -> dict[str, Any]:
    try:
        payload = load_training_state_payload(bundle.agent, state_path)
    except Exception as exc:
        print(f"[warn] failed to load {state_path}: {exc}")
        quarantine_bad_training_state(state_path, exc)
        backup_errors: list[tuple[Path, BaseException]] = []
        for backup_path in backup_generation_paths(state_path):
            if not backup_path.exists():
                continue
            print(f"[warn] trying backup training state: {backup_path}")
            try:
                payload = load_training_state_payload(bundle.agent, backup_path)
            except Exception as backup_exc:
                backup_errors.append((backup_path, backup_exc))
                quarantine_bad_training_state(backup_path, backup_exc)
                continue
            break
        else:
            attempted = [str(path) for path, _ in backup_errors]
            raise TrainingStateResumeUnavailable(
                f"Resume state is unreadable and no readable backup exists: {state_path}; attempted={attempted}"
            ) from exc
    if not environment_matches_config(bundle.config, payload):
        raise ValueError("Resume state environment does not match --env. Start a new run for a different environment.")
    if payload.get("rp_variant") != bundle.rp_variant:
        raise ValueError(
            f"Resume state variant mismatch: state={payload.get('rp_variant')} current={bundle.rp_variant}"
        )
    if not dijkstra_guidance_settings_match(bundle.config, payload):
        raise ValueError(
            "Resume state Dijkstra-guidance settings do not match the current command. "
            "Use the same guidance flags as the original run."
        )
    if not training_schedule_settings_match(bundle.config, payload):
        raise ValueError(
            "Resume state training schedule does not match the current command. "
            "Use the same total episodes and epsilon-boundary flags as the original run."
        )
    if "reproducibility" in payload:
        for key, saved in payload["reproducibility"].items():
            if getattr(bundle.config, key) != saved:
                raise ValueError(f"Resume reproducibility setting mismatch: {key}")
    _load_agent_training_state(bundle.agent, payload["agent_state"])
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])
    elif bundle.config.training_seed is not None:
        print("[warn] legacy state has no RNG snapshot; exact resumed randomness is unavailable")
    return payload


def run_training_episode(
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
    epsilon_scheduler: EpsilonScheduler,
    progress: TrainingProgress,
    episode: int,
    scenario_key: ScenarioKey | None = None,
) -> TrainingEpisodeResult:
    progress.last_episode = episode
    epsilon = (
        0.0
        if bundle.agent_type in {"ppo_rllib", "ppo_torch", "ppo_route"}
        else epsilon_scheduler.value(episode)
    )
    bundle.agent.EPSILON = epsilon
    if hasattr(bundle.agent, "set_training_episode"):
        bundle.agent.set_training_episode(episode)

    key, scenario = resolve_training_scenario(bundle.config, scenario_key)
    progress.record_scenario_hit(key.scenario_index)

    load_args = build_load_args(bundle.config, scenario)
    memory_before = len(bundle.agent.memory)
    load_retry_count = max(0, bundle.config.sumo_load_retry_count)
    for attempt in range(load_retry_count + 1):
        try:
            traci.load(load_args)
            break
        except Exception as exc:
            if PRINT_RECOVERABLE_EPISODE_FAILURES:
                print(f"traci.load failed at episode={episode} scenario={key}")
                print(load_args)
                print(f"sumo error log: {sumo_log_root(bundle.config)}")
            if attempt >= load_retry_count:
                raise RecoverableEpisodeError(
                    "traci.load failed after retries",
                    episode=episode,
                    scenario_key=key,
                    load_args=load_args,
                    original_error=exc,
                ) from exc
            restart_sumo_session(
                bundle.config,
                reason=(
                    f"retrying traci.load for episode={episode} "
                    f"scenario={key} attempt={attempt + 1}/{load_retry_count}"
                ),
            )

    runner = build_episode_runner(
        traci,
        bundle.net,
        bundle.static_pack,
        bundle.agent,
        bundle.config,
        logger=bundle.logger,
    )
    # PPO has no replay-buffer warmup: collect on-policy rollouts immediately
    # and update as soon as its rollout batch reaches train_batch_size.
    train_enabled = (
        bundle.agent_type in {"ppo_rllib", "ppo_torch", "ppo_route"}
        or (episode + 1) >= bundle.config.schedule.warmup_end
    )
    runner_input = build_runner_input(
        scenario,
        evaluate=False,
        train=train_enabled,
        dijkstra_guidance_probability=dijkstra_guidance_probability(bundle.config, episode),
    )
    # TODO: align remaining runner behavior with the original `run_episode_multi_dqngnn`.

    try:
        total_costs = runner.run(runner_input)
    except AgentLearningError:
        raise
    except Exception as exc:
        if PRINT_RECOVERABLE_EPISODE_FAILURES:
            print(f"runner.run failed at episode={episode} scenario={key}")
            print(load_args)
            print(f"sumo error log: {sumo_log_root(bundle.config)}")
        raise RecoverableEpisodeError(
            "runner.run failed",
            episode=episode,
            scenario_key=key,
            load_args=load_args,
            original_error=exc,
        ) from exc
    run_stats = getattr(runner, "last_run_stats", {})
    memory_after = len(bundle.agent.memory)
    episode_cost_minutes = float(run_stats.get("episode_cost_minutes", 0.0))
    ev_costs_minutes = [float(value) for value in run_stats.get("ev_costs_minutes", [])]
    update_training_artifacts(artifacts, total_costs, episode_cost_minutes)

    return TrainingEpisodeResult(
        episode=episode,
        scenario_key=key,
        scenario=scenario,
        load_args=load_args,
        epsilon=epsilon,
        dijkstra_guidance_probability=float(runner_input.dijkstra_guidance_probability),
        exploration_decision_count=int(run_stats.get("exploration_decision_count", 0)),
        dijkstra_guidance_available_count=int(run_stats.get("dijkstra_guidance_available_count", 0)),
        dijkstra_guided_decision_count=int(run_stats.get("dijkstra_guided_decision_count", 0)),
        train_enabled=train_enabled,
        memory_before=memory_before,
        memory_after=memory_after,
        total_costs=total_costs,
        episode_cost_minutes=episode_cost_minutes,
        ev_costs_minutes=ev_costs_minutes,
        penalized_episode_cost_minutes=float(run_stats.get("penalized_episode_cost_minutes", episode_cost_minutes)),
        penalized_ev_costs_minutes=[float(value) for value in run_stats.get("penalized_ev_costs_minutes", ev_costs_minutes)],
        success_count=int(run_stats.get("success_count", 0)),
        failure_count=int(run_stats.get("failure_count", 0)),
        success_rate=float(run_stats.get("success_rate", 0.0)),
        failure_reasons=dict(run_stats.get("failure_reasons", {})),
        simulation_steps=int(run_stats.get("simulation_steps", 0)),
        transition_steps=int(run_stats.get("transition_steps", 0)),
        truncated_by_max_steps=bool(run_stats.get("truncated_by_max_steps", False)),
        update_attempts=int(run_stats.get("update_attempts", 0)),
        learn_calls=int(run_stats.get("learn_calls", 0)),
        gradient_steps=int(run_stats.get("gradient_steps", 0)),
        update_metrics={
            str(key): float(value)
            for key, value in dict(run_stats.get("update_metrics", {})).items()
        },
        last_loss=run_stats.get("last_loss"),
        mean_loss=run_stats.get("mean_loss"),
        teleport_start_count=int(run_stats.get("teleport_start_count", 0)),
        teleport_end_count=int(run_stats.get("teleport_end_count", 0)),
        teleport_started_ids=list(run_stats.get("teleport_started_ids", [])),
        teleport_ended_ids=list(run_stats.get("teleport_ended_ids", [])),
        ev_teleport_started_ids=list(run_stats.get("ev_teleport_started_ids", [])),
        ev_teleport_ended_ids=list(run_stats.get("ev_teleport_ended_ids", [])),
        bottleneck_enabled=bool(run_stats.get("bottleneck_enabled", False)),
        bottleneck_triggered=bool(run_stats.get("bottleneck_triggered", False)),
        bottleneck_active_steps=int(run_stats.get("bottleneck_active_steps", 0)),
        bottleneck_events=list(run_stats.get("bottleneck_events", [])),
    )


def update_training_artifacts(
    artifacts: TrainingArtifacts,
    total_costs: list[float],
    episode_cost_minutes: float,
) -> None:
    if not total_costs:
        return
    reward_sum = sum(total_costs)
    artifacts.rewards.append(reward_sum)
    artifacts.reward_history.append(reward_sum)
    artifacts.average_rewards.append(float(np.mean(artifacts.reward_history)))
    artifacts.costs.append(float(episode_cost_minutes))
    artifacts.cost_history.append(float(episode_cost_minutes))
    artifacts.average_costs.append(float(np.mean(artifacts.cost_history)))


def update_loss_artifacts(artifacts: TrainingArtifacts, mean_loss: float | None) -> None:
    artifacts.losses.append(mean_loss)
    if mean_loss is None:
        last_avg = artifacts.average_losses[-1] if artifacts.average_losses else None
        if last_avg is not None:
            artifacts.average_losses.append(last_avg)
        return
    artifacts.loss_history.append(mean_loss)
    artifacts.average_losses.append(float(np.mean(artifacts.loss_history)))


def record_training_reward(artifacts: TrainingArtifacts, episode_result: TrainingEpisodeResult) -> None:
    if not episode_result.total_costs:
        return
    artifacts.reward_records.append(
        {
            "episode": episode_result.episode,
            "variant": artifacts.rp_variant,
            "reward": episode_result.reward_sum,
            "average_reward": artifacts.average_rewards[-1] if artifacts.average_rewards else None,
            "cost": episode_result.cost_sum,
            "cost_minutes": episode_result.cost_sum,
            "average_cost": artifacts.average_costs[-1] if artifacts.average_costs else None,
            "ev_costs_minutes": episode_result.ev_costs_minutes,
            "penalized_cost_minutes": episode_result.penalized_episode_cost_minutes,
            "penalized_ev_costs_minutes": episode_result.penalized_ev_costs_minutes,
            "success_count": episode_result.success_count,
            "failure_count": episode_result.failure_count,
            "success_rate": episode_result.success_rate,
            "failure_reasons": episode_result.failure_reasons,
            "scenario_index": episode_result.scenario_key.scenario_index,
            "probability_index": episode_result.scenario_key.probability_index,
            "seed": episode_result.scenario_key.seed,
            "epsilon": episode_result.epsilon,
            "dijkstra_guidance_probability": episode_result.dijkstra_guidance_probability,
            "exploration_decision_count": episode_result.exploration_decision_count,
            "dijkstra_guidance_available_count": episode_result.dijkstra_guidance_available_count,
            "dijkstra_guided_decision_count": episode_result.dijkstra_guided_decision_count,
            "simulation_steps": episode_result.simulation_steps,
            "transition_steps": episode_result.transition_steps,
            "truncated_by_max_steps": episode_result.truncated_by_max_steps,
            "update_attempts": episode_result.update_attempts,
            "learn_calls": episode_result.learn_calls,
            "gradient_steps": episode_result.gradient_steps,
            "update_metrics": episode_result.update_metrics,
            "last_loss": episode_result.last_loss,
            "mean_loss": episode_result.mean_loss,
            "teleport_start_count": episode_result.teleport_start_count,
            "teleport_end_count": episode_result.teleport_end_count,
            "teleport_started_ids": episode_result.teleport_started_ids,
            "teleport_ended_ids": episode_result.teleport_ended_ids,
            "ev_teleport_started_ids": episode_result.ev_teleport_started_ids,
            "ev_teleport_ended_ids": episode_result.ev_teleport_ended_ids,
            "bottleneck_enabled": episode_result.bottleneck_enabled,
            "bottleneck_triggered": episode_result.bottleneck_triggered,
            "bottleneck_active_steps": episode_result.bottleneck_active_steps,
            "bottleneck_events": episode_result.bottleneck_events,
        }
    )


def record_failed_episode(
    artifacts: TrainingArtifacts,
    *,
    episode: int,
    scenario_key: ScenarioKey,
    error: BaseException,
    load_args: list[str] | None = None,
) -> None:
    artifacts.failed_episode_records.append(
        {
            "episode": int(episode),
            "variant": artifacts.rp_variant,
            "scenario_index": scenario_key.scenario_index,
            "probability_index": scenario_key.probability_index,
            "seed": scenario_key.seed,
            "error_type": type(error).__name__,
            "error": str(error),
            "load_args": load_args or [],
        }
    )


def save_training_rewards_json(artifacts: TrainingArtifacts, output_path: Path) -> None:
    log_save_stage("rewards_json start", path=output_path)
    artifacts = normalize_training_artifacts(artifacts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rewards": artifacts.rewards,
        "average_rewards": artifacts.average_rewards,
        "costs": artifacts.costs,
        "average_costs": artifacts.average_costs,
        "cost_unit": "minutes",
        "losses": artifacts.losses,
        "average_losses": artifacts.average_losses,
        "records": artifacts.reward_records,
        "failed_episodes": artifacts.failed_episode_records,
        "scenario_schedule_seed": artifacts.scenario_schedule_seed,
        "rp_variant": artifacts.rp_variant,
        "agent_type": artifacts.agent_type,
        "edge_feature_mode": artifacts.edge_feature_mode,
        "best_validation_reward": artifacts.best_validation_reward,
        "best_better_ratio": artifacts.best_better_ratio,
        "best_better_ratio_cost": artifacts.best_better_ratio_cost,
        "best_arrival_rate": artifacts.best_arrival_rate,
        "best_arrival_cost": artifacts.best_arrival_cost,
        "best_arrival_episode": artifacts.best_arrival_episode,
        "bad_validation_count": artifacts.bad_validation_count,
        "arrival_gate_required": 1.0,
        "validation_records": artifacts.validation_records,
        "warm_start_checkpoint": artifacts.warm_start_checkpoint,
        "warm_start_checkpoint_episode": artifacts.warm_start_checkpoint_episode,
    }
    tmp_path = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            log_save_stage("rewards_json dump start", path=tmp_path)
            json.dump(payload, f, ensure_ascii=False, indent=2)
            fsync_file_object(f)
            log_save_stage("rewards_json dump done", path=tmp_path)
        if not json_is_readable(tmp_path):
            raise RuntimeError(f"temporary reward json is unreadable: {tmp_path}")
        if output_path.exists():
            if json_is_readable(output_path):
                backup_path = backup_generation_path(output_path, 1)
                backup_tmp_path = backup_path.with_name(f"{backup_path.name}.{os.getpid()}.tmp")
                copy_file_fsynced(output_path, backup_tmp_path)
                if not json_is_readable(backup_tmp_path):
                    raise RuntimeError(f"temporary reward json backup is unreadable: {backup_tmp_path}")
                rotate_backup_generations(output_path)
                safe_replace_file(backup_tmp_path, backup_path)
            else:
                corrupt_path = output_path.with_name(f"{output_path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
                output_path.replace(corrupt_path)
                print(f"[warn] quarantined unreadable reward json {output_path} -> {corrupt_path}")
        log_save_stage("rewards_json replace start", path=output_path)
        safe_replace_file(tmp_path, output_path)
        log_save_stage("rewards_json done", path=output_path)
    except PermissionError as exc:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fallback_path = output_path.with_name(f"{output_path.stem}_{stamp}.json")
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            fsync_file_object(f)
        print(f"[warn] could not replace locked reward json: {exc}; saved fallback: {fallback_path}")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except PermissionError:
            pass


def save_agent_checkpoint_atomic(agent: Any, episode: int, output_path: Path) -> None:
    log_save_stage("agent_checkpoint start", episode=episode, path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp")
    try:
        log_save_stage("agent_checkpoint torch.save start", episode=episode, path=tmp_path)
        agent.save_tmp_file(episode, filename=str(tmp_path))
        log_save_stage("agent_checkpoint replace start", episode=episode, path=output_path)
        safe_replace_file(tmp_path, output_path)
        log_save_stage("agent_checkpoint done", episode=episode, path=output_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except PermissionError:
            pass


def maybe_save_training_outputs(
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
    episode: int,
    output_dir: Path,
    save_interval: int = 100,
) -> None:
    if (episode + 1) % save_interval != 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_save_stage("training_outputs checkpoint start", episode=episode, path=output_dir / "tmp_file.pt")
    save_agent_checkpoint_atomic(bundle.agent, episode, output_dir / "tmp_file.pt")
    log_save_stage("training_outputs checkpoint done", episode=episode, path=output_dir / "tmp_file.pt")
    log_save_stage("training_outputs reward_plot start", episode=episode, path=output_dir / "Result.png")
    simple_plot(artifacts.rewards, xlabel="Episode", ylabel="Reward", filename=str(output_dir / "Result.png"))
    log_save_stage("training_outputs reward_plot done", episode=episode, path=output_dir / "Result.png")
    log_save_stage("training_outputs average_plot start", episode=episode, path=output_dir / "Average Result.png")
    simple_plot(
        artifacts.average_rewards,
        xlabel="Episode",
        ylabel="Reward",
        filename=str(output_dir / "Average Result.png"),
    )
    log_save_stage("training_outputs average_plot done", episode=episode, path=output_dir / "Average Result.png")


def run_with_agent_eval_mode(bundle: RefactorRuntimeBundle, fn) -> Any:
    rng_state = capture_rng_state() if bundle.config.training_seed is not None else None
    if hasattr(bundle.agent, "encoder"):
        bundle.agent.encoder.eval()
    bundle.agent.eval_net.eval()

    try:
        return fn()
    finally:
        if hasattr(bundle.agent, "encoder"):
            bundle.agent.encoder.train()
        bundle.agent.eval_net.train()
        if rng_state is not None:
            restore_rng_state(rng_state)

def run_validation_for_scenario_pairs(
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
    scenario_pairs: list[tuple[ScenarioKey, ScenarioSpec]],
    desc: str = "Validation",
    scenario_source: str = "validation",
) -> dict[str, Any]:
    total_time = 0
    total_penalized_time = 0
    total_success_count = 0
    total_ev_count = 0
    total_dijkstra_time = 0
    better_count = 0
    total_count = 0

    agent_cost_list = []
    penalized_agent_cost_list = []
    dijkstra_cost_list = []
    agent_path = []
    testing_paths = []
    agent_teleport_start_count = 0
    agent_teleport_end_count = 0
    dijkstra_teleport_start_count = 0
    dijkstra_teleport_end_count = 0
    agent_ev_teleport_started_ids: set[str] = set()
    agent_ev_teleport_ended_ids: set[str] = set()
    dijkstra_ev_teleport_started_ids: set[str] = set()
    dijkstra_ev_teleport_ended_ids: set[str] = set()
    agent_teleport_scenarios: list[dict[str, Any]] = []
    dijkstra_teleport_scenarios: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []


    for key, scenario in tqdm(scenario_pairs, desc=desc, unit="scenario"):
    # key, scenario = testing_pairs[0]
        traci.load(build_load_args(bundle.config, scenario))
        runner = build_episode_runner(
            traci,
            bundle.net,
            bundle.static_pack,
            bundle.agent,
            bundle.config,
            logger=bundle.logger,
        )

        runner_input = build_runner_input(
        scenario,
        evaluate=True,
        train=False,
        agent_path=agent_path,
        )

        costs = runner.run(runner_input)
        runner_stats = getattr(runner, "last_run_stats", {})
        # print(runner_input.agent_path)
        # print(costs)
        if not costs:
            print(f"error {key.scenario_index}, {key.probability_index}, {key.seed}")
            continue
        val_time = float(runner_stats.get("episode_cost_minutes", 3.0 * len(costs) - sum(costs)))
        penalized_val_time = float(runner_stats.get("penalized_episode_cost_minutes", val_time))
        total_success_count += int(runner_stats.get("success_count", len(costs)))
        total_ev_count += len(costs)

        # ================= dijkstra part
        if scenario_source == "training":
            result = run_dijkstra_baseline_for_training_key(
                traci=traci,
                net=bundle.net,
                static_pack=bundle.static_pack,
                runtime_config=bundle.config,
                scenario_index=key.scenario_index,
                probability_index=key.probability_index,
                seed=key.seed,
                training_paths=testing_paths,
                traffic_jam=False,
            )
        elif scenario_source == "testing":
            result = run_dijkstra_baseline_for_testing_key(
                traci=traci,
                net=bundle.net,
                static_pack=bundle.static_pack,
                runtime_config=bundle.config,
                scenario_index=key.scenario_index,
                probability_index=key.probability_index,
                seed=key.seed,
                testing_paths=testing_paths,
                traffic_jam=False,
            )
        elif scenario_source == "validation":
            result = run_dijkstra_baseline_for_validation_key(
                traci=traci,
                net=bundle.net,
                static_pack=bundle.static_pack,
                runtime_config=bundle.config,
                scenario_index=key.scenario_index,
                probability_index=key.probability_index,
                seed=key.seed,
                validation_paths=testing_paths,
                traffic_jam=False,
            )
        else:
            raise ValueError(f"Unknown scenario_source: {scenario_source}")

        dijk_cost = result.total_time_spend
        # print('-'*20, 'dijkstra')
        # print(dijk_cost)
        # print(result.ev_routes)
        # print(result.ev_times)

        total_time += val_time
        total_penalized_time += penalized_val_time
        total_dijkstra_time += dijk_cost
        agent_cost_list.append(val_time)
        penalized_agent_cost_list.append(penalized_val_time)
        dijkstra_cost_list.append(dijk_cost)
        if scenario_source == "training":
            flow_name = bundle.config.environment.training_flows[key.scenario_index]
        elif scenario_source == "testing":
            flow_name = bundle.config.environment.testing_flows[key.scenario_index]
        elif scenario_source == "validation":
            flow_name = bundle.config.environment.validation_flows[key.scenario_index]
        else:
            flow_name = str(key.scenario_index)
        case_records.append(
            {
                "variant": bundle.rp_variant,
                "scenario_source": scenario_source,
                "scenario_index": key.scenario_index,
                "demand_setting": flow_name,
                "probability_index": key.probability_index,
                "seed": key.seed,
                "cost": val_time,
                "cost_minutes": val_time,
                "penalized_cost_minutes": penalized_val_time,
                "dijkstra_cost_minutes": float(dijk_cost),
                "ev_count": len(costs),
                "ev_costs_minutes": list(runner_stats.get("ev_costs_minutes", [])),
                "penalized_ev_costs_minutes": list(runner_stats.get("penalized_ev_costs_minutes", [])),
                "success_count": int(runner_stats.get("success_count", 0)),
                "failure_count": int(runner_stats.get("failure_count", 0)),
                "success_rate": float(runner_stats.get("success_rate", 0.0)),
                "failure_reasons": dict(runner_stats.get("failure_reasons", {})),
            }
        )
        agent_teleport_start_count += int(runner_stats.get("teleport_start_count", 0))
        agent_teleport_end_count += int(runner_stats.get("teleport_end_count", 0))
        dijkstra_teleport_start_count += int(result.teleport_start_count)
        dijkstra_teleport_end_count += int(result.teleport_end_count)
        agent_ev_teleport_started_ids.update(runner_stats.get("ev_teleport_started_ids", []))
        agent_ev_teleport_ended_ids.update(runner_stats.get("ev_teleport_ended_ids", []))
        dijkstra_ev_teleport_started_ids.update(result.ev_teleport_started_ids)
        dijkstra_ev_teleport_ended_ids.update(result.ev_teleport_ended_ids)
        if int(runner_stats.get("teleport_start_count", 0)) > 0 or int(runner_stats.get("teleport_end_count", 0)) > 0:
            agent_teleport_scenarios.append(
                {
                    "scenario_index": key.scenario_index,
                    "probability_index": key.probability_index,
                    "seed": key.seed,
                    "teleport_start_count": int(runner_stats.get("teleport_start_count", 0)),
                    "teleport_end_count": int(runner_stats.get("teleport_end_count", 0)),
                    "teleport_started_ids": list(runner_stats.get("teleport_started_ids", [])),
                    "teleport_ended_ids": list(runner_stats.get("teleport_ended_ids", [])),
                    "ev_teleport_started_ids": list(runner_stats.get("ev_teleport_started_ids", [])),
                    "ev_teleport_ended_ids": list(runner_stats.get("ev_teleport_ended_ids", [])),
                }
            )
        if int(result.teleport_start_count) > 0 or int(result.teleport_end_count) > 0:
            dijkstra_teleport_scenarios.append(
                {
                    "scenario_index": key.scenario_index,
                    "probability_index": key.probability_index,
                    "seed": key.seed,
                    "teleport_start_count": int(result.teleport_start_count),
                    "teleport_end_count": int(result.teleport_end_count),
                    "teleport_started_ids": list(result.teleport_started_ids),
                    "teleport_ended_ids": list(result.teleport_ended_ids),
                    "ev_teleport_started_ids": list(result.ev_teleport_started_ids),
                    "ev_teleport_ended_ids": list(result.ev_teleport_ended_ids),
                }
            )

        if dijk_cost > 0 and penalized_val_time / dijk_cost <= 1.0:
            better_count += 1

        total_count += 1
    


    return {
        "variant": bundle.rp_variant,
        "agent_type": bundle.agent_type,
        "edge_feature_mode": bundle.edge_feature_mode,
        "scenario_source": scenario_source,
        "mean_cost": float(np.mean(agent_cost_list)) if agent_cost_list else None,
        "std_cost": float(np.std(agent_cost_list)) if agent_cost_list else None,
        "mean_penalized_cost": float(np.mean(penalized_agent_cost_list)) if penalized_agent_cost_list else None,
        "std_penalized_cost": float(np.std(penalized_agent_cost_list)) if penalized_agent_cost_list else None,
        "success_rate": total_success_count / total_ev_count if total_ev_count else 0.0,
        "success_count": total_success_count,
        "ev_count": total_ev_count,
        "raw_per_case_cost": agent_cost_list,
        "raw_per_case_penalized_cost": penalized_agent_cost_list,
        "cases": case_records,
        "total_time": total_time,
        "total_penalized_time": total_penalized_time,
        "selection_time": total_penalized_time,
        "total_dijkstra_time": total_dijkstra_time,
        "better_count": better_count,
        "total_count": total_count,
        "better_ratio": better_count / total_count if total_count else 0,
        "agent_cost_list": agent_cost_list,
        "dijkstra_cost_list": dijkstra_cost_list,
        "agent_path": agent_path,
        "agent_teleport_start_count": agent_teleport_start_count,
        "agent_teleport_end_count": agent_teleport_end_count,
        "dijkstra_teleport_start_count": dijkstra_teleport_start_count,
        "dijkstra_teleport_end_count": dijkstra_teleport_end_count,
        "agent_ev_teleport_started_ids": sorted(agent_ev_teleport_started_ids),
        "agent_ev_teleport_ended_ids": sorted(agent_ev_teleport_ended_ids),
        "dijkstra_ev_teleport_started_ids": sorted(dijkstra_ev_teleport_started_ids),
        "dijkstra_ev_teleport_ended_ids": sorted(dijkstra_ev_teleport_ended_ids),
        "agent_teleport_scenarios": agent_teleport_scenarios,
        "dijkstra_teleport_scenarios": dijkstra_teleport_scenarios,
    }


def run_validation_phase(
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
) -> dict[str, Any]:
    return run_validation_for_scenario_pairs(
        bundle,
        artifacts,
        resolve_validation_scenarios(bundle.config),
        desc="Validation",
        scenario_source="validation",
    )


def run_testing_data_validation_phase(
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
) -> dict[str, Any]:
    return run_validation_for_scenario_pairs(
        bundle,
        artifacts,
        resolve_testing_scenarios(bundle.config),
        desc="Test Validation",
        scenario_source="testing",
    )


def run_training_data_validation_phase(
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
) -> dict[str, Any]:
    return run_validation_for_scenario_pairs(
        bundle,
        artifacts,
        resolve_training_scenarios(bundle.config),
        desc="Train Validation",
        scenario_source="training",
    )


def _validation_json_safe(value: Any) -> Any:
    """Preserve complete evaluation data in portable, strict JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _validation_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _validation_json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _validation_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_validation_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_validation_result_atomic(result: dict[str, Any], output_path: Path) -> None:
    """Write the full validation report only after its checkpoint is saved."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as stream:
            json.dump(_validation_json_safe(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
            fsync_file_object(stream)
        safe_replace_file(tmp_path, output_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except PermissionError:
            pass


def maybe_run_validation_and_checkpoint(
    bundle: RefactorRuntimeBundle,
    artifacts: TrainingArtifacts,
    episode: int,
    output_dir: Path,
    close_sumo_before_save_enabled: bool = False,
) -> dict[str, Any] | None:
    if not training_validation_due(bundle.config, episode):
        return None

    try:
        result = run_with_agent_eval_mode(bundle, lambda: run_validation_phase(bundle, artifacts))
    except BaseException as exc:
        print(f"[warn] validation failed at episode={episode}: {type(exc).__name__}: {exc}")
        restart_sumo_session(
            bundle.config,
            reason=f"recovering after failed validation at episode={episode}",
        )
        return None
    print(
        f"better ratio {result['better_ratio']}, "
        f"agent raw {result['total_time']}, "
        f"agent penalized {result['selection_time']}, "
        f"dijkstra {result['total_dijkstra_time']}, "
        f"arrival {result['success_count']}/{result['ev_count']} "
        f"({result['success_rate']:.2%})"
    )
    artifacts = normalize_training_artifacts(artifacts)
    validation_time = float(result.get("selection_time", float("inf")))
    better_ratio = float(result.get("better_ratio", float("-inf")))
    success_count = int(result.get("success_count", 0))
    ev_count = int(result.get("ev_count", 0))
    valid_arrival_counts = ev_count > 0 and 0 <= success_count <= ev_count
    arrival_rate = success_count / ev_count if valid_arrival_counts else float("-inf")
    meets_arrival_gate = valid_arrival_counts and success_count == ev_count
    should_save_best_validation = (
        meets_arrival_gate
        and np.isfinite(validation_time)
        and validation_time < artifacts.best_validation_reward
    )
    should_save_best_better_ratio = (
        meets_arrival_gate
        and np.isfinite(better_ratio)
        and (
            better_ratio > artifacts.best_better_ratio
            or (
                better_ratio == artifacts.best_better_ratio
                and validation_time < artifacts.best_better_ratio_cost
            )
        )
    )
    should_save_best_arrival = (
        valid_arrival_counts
        and np.isfinite(validation_time)
        and (
            arrival_rate > artifacts.best_arrival_rate
            or (
                arrival_rate == artifacts.best_arrival_rate
                and validation_time < artifacts.best_arrival_cost
            )
        )
    )
    checkpoint_prefix = bundle.agent_type or bundle.rp_variant
    lowest_cost_path = output_dir / f"{checkpoint_prefix}_best_100pct_arrival_lowest_cost.pt"
    better_ratio_path = output_dir / f"{checkpoint_prefix}_best_100pct_arrival_better_ratio.pt"
    best_arrival_path = output_dir / f"{checkpoint_prefix}_best_arrival.pt"
    best_arrival_result_path = best_arrival_path.with_suffix(".json")
    checkpoint_path = output_dir / "validation" / f"episode_{episode + 1:06d}.pt"
    result_path = checkpoint_path.with_suffix(".json")
    result.update(
        episode=int(episode),
        episode_number=int(episode + 1),
        validation_schedule=validation_schedule_settings(bundle.config),
        variant=bundle.rp_variant,
        agent_type=bundle.agent_type,
        checkpoint=str(checkpoint_path),
        validation_result_file=str(result_path),
        arrival_gate_required=1.0,
        arrival_gate_passed=meets_arrival_gate,
        saved_lowest_cost_checkpoint=bool(should_save_best_validation),
        saved_better_ratio_checkpoint=bool(should_save_best_better_ratio),
        saved_best_arrival_checkpoint=bool(should_save_best_arrival),
        lowest_cost_checkpoint=str(lowest_cost_path),
        better_ratio_checkpoint=str(better_ratio_path),
        best_arrival_checkpoint=str(best_arrival_path),
        best_arrival_result_file=str(best_arrival_result_path),
        best_arrival_selection_policy="highest arrival rate; ties use lowest total penalized cost",
    )
    closed_for_save = close_sumo_before_save(
        close_sumo_before_save_enabled,
        reason=f"before saving validation checkpoints after episode={episode}",
    )
    try:
        # Every completed validation keeps an independently loadable model,
        # even if the strict 100% arrival gate rejects the best-model aliases.
        save_agent_checkpoint_atomic(bundle.agent, episode, checkpoint_path)
        if should_save_best_validation:
            save_agent_checkpoint_atomic(bundle.agent, episode, lowest_cost_path)
        if should_save_best_better_ratio:
            save_agent_checkpoint_atomic(bundle.agent, episode, better_ratio_path)
        if should_save_best_arrival:
            save_agent_checkpoint_atomic(bundle.agent, episode, best_arrival_path)
        save_validation_result_atomic(result, result_path)
        if should_save_best_arrival:
            best_result = dict(result)
            best_result["source_checkpoint"] = str(checkpoint_path)
            best_result["checkpoint"] = str(best_arrival_path)
            best_result["validation_result_file"] = str(best_arrival_result_path)
            save_validation_result_atomic(best_result, best_arrival_result_path)
    finally:
        restart_sumo_after_save(
            bundle.config,
            closed_for_save,
            reason=f"after saving validation checkpoints for episode={episode}",
        )
    # Advance selection state only after all model/result writes succeed.
    if should_save_best_validation:
        artifacts.best_validation_reward = validation_time
    if should_save_best_better_ratio:
        artifacts.best_better_ratio = better_ratio
        artifacts.best_better_ratio_cost = validation_time
    if should_save_best_arrival:
        artifacts.best_arrival_rate = arrival_rate
        artifacts.best_arrival_cost = validation_time
        artifacts.best_arrival_episode = int(episode)
    if not should_save_best_validation and not should_save_best_better_ratio:
        artifacts.bad_validation_count += 1
    artifacts.validation_records.append(
        {
            "episode": int(episode),
            "episode_number": int(episode + 1),
            "success_count": success_count,
            "ev_count": ev_count,
            "success_rate": float(result.get("success_rate", 0.0)),
            "arrival_gate_passed": meets_arrival_gate,
            "selection_time": validation_time,
            "better_ratio": better_ratio,
            "saved_lowest_cost_checkpoint": bool(should_save_best_validation),
            "saved_better_ratio_checkpoint": bool(should_save_best_better_ratio),
            "saved_best_arrival_checkpoint": bool(should_save_best_arrival),
            "lowest_cost_checkpoint": str(lowest_cost_path),
            "better_ratio_checkpoint": str(better_ratio_path),
            "best_arrival_checkpoint": str(best_arrival_path),
            "checkpoint": str(checkpoint_path),
            "validation_result_file": str(result_path),
        }
    )
    if not meets_arrival_gate:
        print(
            f"[checkpoint] 100%-arrival best models skipped "
            f"({success_count}/{ev_count}); validation snapshot saved: {checkpoint_path}"
        )
    return result


def run_training_loop(
    bundle: RefactorRuntimeBundle,
    start_episode: int = 0,
    checkpoint_path: Path | None = None,
    warm_start_checkpoint_path: Path | None = None,
    resume_state_path: Path | None = None,
    max_episodes: int | None = None,
    scenario_schedule_seed: int = 42,
    scenario_schedule: list[ScenarioKey] | None = None,
    output_dir: Path | None = None,
    autosave_state_path: Path | None = None,
    autosave_every: int = 10,
    rewards_save_every: int = 10,
    verify_autosave_state: bool = False,
    close_sumo_before_save_enabled: bool = False,
    training_validation_enabled: bool = True,
) -> TrainingArtifacts:

    epsilon_scheduler = EpsilonScheduler(bundle.config.schedule, min_epsilon=bundle.agent.min_epsilon)
    output_dir = resolve_variant_output_dir(bundle.config, output_dir)
    autosave_state_path = autosave_state_path or default_training_state_path(bundle.config, output_dir)
    initialization_sources = [
        checkpoint_path is not None,
        warm_start_checkpoint_path is not None,
        resume_state_path is not None,
    ]
    if sum(initialization_sources) > 1:
        raise ValueError(
            "Use only one of checkpoint_path, warm_start_checkpoint_path, or resume_state_path."
        )

    if resume_state_path is not None:
        try:
            state_payload = load_training_state(bundle, resume_state_path)
        except TrainingStateResumeUnavailable as exc:
            print(f"[warn] {exc}")
            print("[warn] starting a new training run because no readable resume state is available.")
            resume_state_path = None
        else:
            artifacts = normalize_training_artifacts(state_payload.get("artifacts") or create_training_artifacts())
            progress = state_payload.get("progress") or TrainingProgress(last_episode=int(state_payload["next_episode"]))
            scenario_schedule_seed = int(state_payload.get("scenario_schedule_seed", scenario_schedule_seed))
            scenario_schedule = state_payload.get("scenario_schedule") or scenario_schedule
            resumed_episode = int(state_payload["next_episode"])

    if resume_state_path is None:
        artifacts = create_training_artifacts()
        artifacts.scenario_schedule_seed = scenario_schedule_seed
        artifacts.rp_variant = bundle.rp_variant
        artifacts.agent_type = bundle.agent_type
        artifacts.edge_feature_mode = bundle.edge_feature_mode
        progress = TrainingProgress(last_episode=start_episode)
        if warm_start_checkpoint_path is not None:
            source_episode = maybe_load_checkpoint(bundle.agent, warm_start_checkpoint_path)
            artifacts.warm_start_checkpoint = str(warm_start_checkpoint_path.resolve())
            artifacts.warm_start_checkpoint_episode = int(source_episode)
            resumed_episode = start_episode
            print(
                f"[fine-tune] warm-started from {warm_start_checkpoint_path} "
                f"(source episode={source_episode}); new run starts at episode={start_episode}"
            )
        else:
            resumed_episode = maybe_load_checkpoint(bundle.agent, checkpoint_path) if checkpoint_path else start_episode

    artifacts.scenario_schedule_seed = scenario_schedule_seed
    artifacts.rp_variant = bundle.rp_variant
    artifacts.agent_type = bundle.agent_type
    artifacts.edge_feature_mode = bundle.edge_feature_mode
    end_episode = bundle.config.schedule.num_episodes
    if max_episodes is not None:
        end_episode = min(end_episode, resumed_episode + max_episodes)
    training_reached_config_end = end_episode >= bundle.config.schedule.num_episodes

    if resumed_episode >= bundle.config.schedule.num_episodes:
        close_sumo_before_save(
            close_sumo_before_save_enabled,
            reason="before saving completed training marker",
        )
        save_training_rewards_json(artifacts, output_dir / "training_rewards.json")
        save_training_completed_marker(
            output_dir=output_dir,
            config=bundle.config,
            artifacts=artifacts,
            next_episode=resumed_episode,
        )
        print(
            f"[info] training already completed: next_episode={resumed_episode}, "
            f"num_episodes={bundle.config.schedule.num_episodes}"
        )
        return artifacts

    if scenario_schedule is None:
        scenario_schedule = build_training_scenario_schedule(
            bundle.config,
            num_episodes=bundle.config.schedule.num_episodes,
            seed=scenario_schedule_seed,
        )
    elif len(scenario_schedule) < end_episode:
        original_schedule_len = len(scenario_schedule)
        extended_schedule = build_training_scenario_schedule(
            bundle.config,
            num_episodes=bundle.config.schedule.num_episodes,
            seed=scenario_schedule_seed,
        )
        scenario_schedule = list(scenario_schedule) + list(extended_schedule[original_schedule_len:])
        print(
            f"[info] extended scenario_schedule from {original_schedule_len} "
            f"to {len(scenario_schedule)} using seed={scenario_schedule_seed}"
        )

    episode_iter = tqdm(
        range(resumed_episode, end_episode),
        desc="Training",
        unit="episode",
    )
    
    for episode in episode_iter:
        restart_interval = bundle.config.sumo_restart_interval_episodes
        if (
            restart_interval is not None
            and restart_interval > 0
            and episode > resumed_episode
            and (episode - resumed_episode) % restart_interval == 0
        ):
            restart_sumo_session(
                bundle.config,
                reason=f"periodic restart before episode={episode}",
            )
        if bundle.logger:
            bundle.logger.start_episode(episode)

        try:
            episode_result = run_training_episode(
                bundle,
                artifacts,
                epsilon_scheduler,
                progress,
                episode,
                scenario_key=scenario_schedule[episode],
            )
        except RecoverableEpisodeError as exc:
            if PRINT_RECOVERABLE_EPISODE_FAILURES:
                print(
                    f"[warn] skipping failed episode={episode} "
                    f"scenario={exc.scenario_key}: {type(exc.original_error).__name__}: {exc.original_error}"
                )
            record_failed_episode(
                artifacts,
                episode=episode,
                scenario_key=exc.scenario_key,
                error=exc.original_error,
                load_args=exc.load_args,
            )
            close_sumo_before_save(
                close_sumo_before_save_enabled,
                reason=f"before saving recovery data after episode={episode}",
            )
            save_training_rewards_json(artifacts, output_dir / "training_rewards.json")
            if autosave_every > 0:
                save_training_state(
                    bundle=bundle,
                    artifacts=artifacts,
                    progress=progress,
                    scenario_schedule=scenario_schedule,
                    scenario_schedule_seed=scenario_schedule_seed,
                    next_episode=episode + 1,
                    output_path=autosave_state_path,
                    verify_after_save=verify_autosave_state,
                )
            restart_sumo_session(
                bundle.config,
                reason=f"recovering after failed episode={episode}",
            )
            continue
        episode_iter.set_postfix(
            epsilon=f"{episode_result.epsilon:.3f}",
            reward=f"{episode_result.reward_sum:.3f}",
            cost=f"{episode_result.cost_sum:.3f}",
            loss="-" if episode_result.mean_loss is None else f"{episode_result.mean_loss:.4f}",
            memory=episode_result.memory_after,
            train=episode_result.train_enabled,
        )

        if not episode_result.total_costs:
            record_failed_episode(
                artifacts,
                episode=episode,
                scenario_key=episode_result.scenario_key,
                error=RuntimeError("episode produced no total_costs"),
                load_args=episode_result.load_args,
            )
            close_sumo_before_save(
                close_sumo_before_save_enabled,
                reason=f"before saving recovery data after episode={episode}",
            )
            save_training_rewards_json(artifacts, output_dir / "training_rewards.json")
            if autosave_every > 0:
                save_training_state(
                    bundle=bundle,
                    artifacts=artifacts,
                    progress=progress,
                    scenario_schedule=scenario_schedule,
                    scenario_schedule_seed=scenario_schedule_seed,
                    next_episode=episode + 1,
                    output_path=autosave_state_path,
                    verify_after_save=verify_autosave_state,
                )
            restart_sumo_session(
                bundle.config,
                reason=f"recovering after empty episode={episode}",
            )
            continue
        update_loss_artifacts(artifacts, episode_result.mean_loss)
        record_training_reward(artifacts, episode_result)
        is_last_episode = episode + 1 >= end_episode
        should_save_rewards = rewards_save_every > 0 and ((episode + 1) % rewards_save_every == 0 or is_last_episode)
        should_save_state = autosave_every > 0 and ((episode + 1) % autosave_every == 0 or is_last_episode)
        should_save_outputs = (episode + 1) % 100 == 0
        closed_for_save = close_sumo_before_save(
            close_sumo_before_save_enabled and (should_save_rewards or should_save_state or should_save_outputs),
            reason=f"before saving outputs after episode={episode}",
        )
        try:
            if should_save_rewards:
                save_training_rewards_json(artifacts, output_dir / "training_rewards.json")
            if should_save_state:
                save_training_state(
                    bundle=bundle,
                    artifacts=artifacts,
                    progress=progress,
                    scenario_schedule=scenario_schedule,
                    scenario_schedule_seed=scenario_schedule_seed,
                    next_episode=episode + 1,
                    output_path=autosave_state_path,
                    verify_after_save=verify_autosave_state,
                )
            if should_save_outputs:
                maybe_save_training_outputs(bundle, artifacts, episode, output_dir)
        finally:
            restart_sumo_after_save(
                bundle.config,
                closed_for_save,
                reason=f"after saving outputs for episode={episode}",
            )
        if training_validation_enabled:
            maybe_run_validation_and_checkpoint(
                bundle,
                artifacts,
                episode,
                output_dir,
                close_sumo_before_save_enabled=close_sumo_before_save_enabled,
            )
    if training_reached_config_end:
        close_sumo_before_save(
            close_sumo_before_save_enabled,
            reason="before saving final training outputs",
        )
        save_training_rewards_json(artifacts, output_dir / "training_rewards.json")
        save_training_completed_marker(
            output_dir=output_dir,
            config=bundle.config,
            artifacts=artifacts,
            next_episode=end_episode,
        )
    return artifacts


def run_validate_only_flow(
    bundle: RefactorRuntimeBundle,
    checkpoint_path: Path,
    use_training_scenarios: bool = False,
    use_testing_scenarios: bool = False,
) -> dict[str, Any]:
    maybe_load_checkpoint(bundle.agent, checkpoint_path, validate_only=True)
    if use_training_scenarios and use_testing_scenarios:
        raise ValueError("Only one validation scenario source can be selected.")
    if use_training_scenarios:
        validation_fn = run_training_data_validation_phase
    elif use_testing_scenarios:
        validation_fn = run_testing_data_validation_phase
    else:
        validation_fn = run_validation_phase
    return run_with_agent_eval_mode(bundle, lambda: validation_fn(bundle, create_training_artifacts()))

def configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def add_bottleneck_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-bottleneck",
        action="store_true",
        help="Enable a fixed lane-speed bottleneck during each episode.",
    )
    parser.add_argument(
        "--bottleneck-edges",
        type=str,
        default=None,
        help="Comma-separated SUMO edge IDs whose lanes will be slowed when bottleneck is enabled.",
    )
    parser.add_argument(
        "--bottleneck-start",
        type=float,
        default=120.0,
        help="Bottleneck start time in simulation seconds.",
    )
    parser.add_argument(
        "--bottleneck-end",
        type=float,
        default=900.0,
        help="Bottleneck end time in simulation seconds.",
    )
    parser.add_argument(
        "--bottleneck-speed-factor",
        type=float,
        default=0.5,
        help="Lane speed multiplier during bottleneck, e.g. 0.5 moderate or 0.3 severe.",
    )


def apply_bottleneck_args(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    if not getattr(args, "enable_bottleneck", False):
        return config
    raw_edges = getattr(args, "bottleneck_edges", None)
    edges = tuple(edge.strip() for edge in raw_edges.split(",") if edge.strip()) if raw_edges else ()
    if not edges:
        raise ValueError("--enable-bottleneck requires --bottleneck-edges EDGE_ID[,EDGE_ID...]")
    return replace(
        config,
        enable_bottleneck=True,
        bottleneck_candidate_edges=edges,
        bottleneck_start_time_sec=float(args.bottleneck_start),
        bottleneck_end_time_sec=float(args.bottleneck_end),
        bottleneck_speed_factor=float(args.bottleneck_speed_factor),
    )


def apply_edqn_dijkstra_guidance_args(
    config: RuntimeConfig,
    args: argparse.Namespace,
) -> RuntimeConfig:
    if not getattr(args, "edqn_dijkstra_guided_exploration", False):
        return config
    if config.rp_variant != "edqn_baseline":
        raise ValueError("--edqn-dijkstra-guided-exploration requires --rp_variant edqn_baseline.")
    probability = float(args.edqn_dijkstra_guide_probability)
    hold_episodes = int(args.edqn_dijkstra_guide_hold_episodes)
    decay_episodes = int(args.edqn_dijkstra_guide_decay_episodes)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("--edqn-dijkstra-guide-probability must be between 0 and 1.")
    if hold_episodes < 0:
        raise ValueError("--edqn-dijkstra-guide-hold-episodes must be non-negative.")
    if decay_episodes <= 0:
        raise ValueError("--edqn-dijkstra-guide-decay-episodes must be positive.")
    return replace(
        config,
        edqn_dijkstra_guidance_enabled=True,
        edqn_dijkstra_guide_probability=probability,
        edqn_dijkstra_guide_hold_episodes=hold_episodes,
        edqn_dijkstra_guide_decay_episodes=decay_episodes,
    )


def apply_training_schedule_args(
    config: RuntimeConfig,
    args: argparse.Namespace,
) -> RuntimeConfig:
    requested_num_episodes = getattr(args, "num_episodes", None)
    requested_warmup = getattr(args, "epsilon_warmup_episodes", None)
    requested_decay_end = getattr(args, "epsilon_decay_end_episode", None)
    if requested_num_episodes is None and requested_warmup is None and requested_decay_end is None:
        return config

    num_episodes = (
        int(requested_num_episodes)
        if requested_num_episodes is not None
        else int(config.schedule.num_episodes)
    )
    # Extending a run should add a low-epsilon stabilization tail by default,
    # rather than stretching the already-tested 225/1200 exploration phases.
    warmup_episodes = (
        int(requested_warmup)
        if requested_warmup is not None
        else int(config.schedule.warmup_end)
    )
    decay_end_episode = (
        int(requested_decay_end)
        if requested_decay_end is not None
        else int(config.schedule.training_end)
    )
    if num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    if warmup_episodes < 0:
        raise ValueError("--epsilon-warmup-episodes must be non-negative.")
    if decay_end_episode < warmup_episodes:
        raise ValueError("--epsilon-decay-end-episode must not precede epsilon warmup.")
    if decay_end_episode > num_episodes:
        raise ValueError("--epsilon-decay-end-episode must not exceed --num-episodes.")
    return replace(
        config,
        schedule=replace(
            config.schedule,
            num_episodes=num_episodes,
            warmup_episodes=warmup_episodes,
            epsilon_decay_end_episode=decay_end_episode,
        ),
    )


def apply_episode_limit_args(
    config: RuntimeConfig, args: argparse.Namespace,
) -> RuntimeConfig:
    """Override common budgets without editing per-machine configuration."""
    overrides = {}
    for field, flag in (
        ("max_decisions_per_ev", "--max-decisions-per-ev"),
        ("max_simulation_steps", "--max-simulation-steps"),
    ):
        value = getattr(args, field, None)
        if value is not None:
            if value <= 0:
                raise ValueError(f"{flag} must be positive.")
            overrides[field] = int(value)
    return replace(config, **overrides) if overrides else config


def apply_validation_schedule_args(
    config: RuntimeConfig, args: argparse.Namespace,
) -> RuntimeConfig:
    start = getattr(args, "validation_start_episode", None)
    interval = getattr(args, "validation_interval", None)
    if start is None and interval is None:
        return config
    updated = replace(
        config,
        validation_start_episode=start if start is not None else config.validation_start_episode,
        validation_interval=interval if interval is not None else config.validation_interval,
    )
    first, _ = resolve_training_validation_schedule(updated)
    if first > config.schedule.num_episodes:
        raise ValueError("The first validation episode exceeds --num-episodes.")
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or validate route-planning ablation variants.")
    parser.add_argument(
        "--max-decisions-per-ev", type=int, default=None,
        help="Override the per-EV decision budget for this run and periodic validation.",
    )
    parser.add_argument(
        "--max-simulation-steps", type=int, default=None,
        help="Override the per-scenario simulation-step budget.",
    )
    parser.add_argument("--env", choices=ENV_CHOICES, default="env",
                        help="Dataset environment: env (original) or new_env (Berlin manifest split).")
    parser.add_argument(
        "--rp_variant",
        choices=RP_VARIANT_CHOICES,
        default="proposed",
        help="Route-planning ablation variant.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Legacy checkpoint continuation using its stored episode number. "
            "Use --fine-tune-checkpoint for a fresh fine-tuning run."
        ),
    )
    parser.add_argument(
        "--fine-tune-checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start a new fine-tuning run from model/target/optimizer weights. "
            "Unlike --resume-state, episode counters, metrics, and replay memory start fresh."
        ),
    )
    parser.add_argument(
        "--fine-tune-episodes",
        type=int,
        default=500,
        help="Number of episodes in a --fine-tune-checkpoint run (default: 500).",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        default=None,
        help="Resume a full autosaved training state, including model, optimizer, replay memory, artifacts, and schedule.",
    )
    parser.add_argument(
        "--autosave-state",
        type=Path,
        default=None,
        help="Path for the latest full training state. Defaults to <output-dir>/training_state_latest.pt.",
    )
    parser.add_argument(
        "--autosave-every",
        type=int,
        default=10,
        help="Save full training state every N completed episodes. Use 0 to disable.",
    )
    parser.add_argument(
        "--rewards-save-every",
        type=int,
        default=10,
        help="Save training_rewards.json every N completed episodes. Use 0 to save only failed/final episodes.",
    )
    parser.add_argument(
        "--verify-autosave-state",
        action="store_true",
        help="After each autosave, reload the full state file to verify it. Slower, but stricter.",
    )
    parser.add_argument(
        "--close-sumo-before-save",
        action="store_true",
        help="Close the SUMO/TraCI session before persistent saves, then restart it after saving.",
    )
    parser.add_argument(
        "--disable-training-validation",
        action="store_true",
        help="Disable periodic validation/checkpointing during training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Defaults to outputs/<rp_variant>/ for env, "
            "outputs/new_env/<rp_variant>/ for new_env, or an environment-specific "
            "outputs/edqn_finetune_revisit/ for --fine-tune-checkpoint."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help=(
            "Total episodes for a new run. Extending this value preserves the existing "
            "epsilon warmup/decay boundaries unless they are explicitly overridden."
        ),
    )
    parser.add_argument(
        "--epsilon-warmup-episodes",
        type=int,
        default=None,
        help="Episodes held at epsilon=1.0 (default boundary remains 225).",
    )
    parser.add_argument(
        "--epsilon-decay-end-episode",
        type=int,
        default=None,
        help="Episode where epsilon reaches its minimum 0.05 (default boundary remains 1200).",
    )
    parser.add_argument(
        "--validation-start-episode", type=int, default=None,
        help="First validation episode (one-based). Omit to preserve the original schedule.",
    )
    parser.add_argument(
        "--validation-interval", type=int, default=None,
        help="Episodes between validations; independent of epsilon decay. Omit for num_episodes // 20.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional short-run cap; this cannot increase the configured total episodes.",
    )
    parser.add_argument(
        "--scenario-schedule-seed",
        type=int,
        default=42,
        help="Random seed used to build the training scenario schedule for a new run.",
    )
    parser.add_argument(
        "--edqn-dijkstra-guided-exploration",
        action="store_true",
        help=(
            "During eDQN epsilon exploration, choose the static-distance Dijkstra "
            "next hop with a decaying probability. Validation/testing remain pure eDQN."
        ),
    )
    parser.add_argument(
        "--edqn-dijkstra-guide-probability",
        type=float,
        default=0.7,
        help="Initial Dijkstra teacher probability within an eDQN exploration action (default: 0.7).",
    )
    parser.add_argument(
        "--edqn-dijkstra-guide-hold-episodes",
        type=int,
        default=500,
        help="Episodes that keep the initial teacher probability unchanged (default: 500).",
    )
    parser.add_argument(
        "--edqn-dijkstra-guide-decay-episodes",
        type=int,
        default=1000,
        help="Episodes used to linearly decay teacher probability to zero (default: 1000).",
    )
    parser.add_argument(
        "--use-gui",
        action="store_true",
        help="Use sumo-gui instead of sumo.",
    )
    add_bottleneck_args(parser)
    add_reproducibility_args(parser)
    return parser.parse_args()


def main() -> None:
    configure_console_encoding()
    args = parse_args()

    initialization_sources = [
        args.checkpoint is not None,
        args.fine_tune_checkpoint is not None,
        args.resume_state is not None,
    ]
    if sum(initialization_sources) > 1:
        raise ValueError(
            "Use only one of --checkpoint, --fine-tune-checkpoint, or --resume-state."
        )
    if args.fine_tune_episodes <= 0:
        raise ValueError("--fine-tune-episodes must be positive.")

    project_root = THIS_DIR
    # config = build_small_env_config(repo_root) 
    config = build_selected_env_config(project_root, env_name=args.env)
    config = replace(config, rp_variant=args.rp_variant, use_gui=args.use_gui)
    if args.fine_tune_checkpoint is not None:
        if any(
            value is not None
            for value in (
                args.num_episodes,
                args.epsilon_warmup_episodes,
                args.epsilon_decay_end_episode,
            )
        ):
            raise ValueError(
                "Use --fine-tune-episodes for fine-tuning; do not combine it with the new-run schedule flags."
            )
        if args.rp_variant != "edqn_baseline":
            raise ValueError("This fine-tune workflow requires --rp_variant edqn_baseline.")
        config = replace(
            config,
            schedule=TrainingSchedule(
                num_episodes=args.fine_tune_episodes,
                warmup_ratio=0.0,
                training_ratio=0.0,
            ),
        )
    else:
        config = apply_training_schedule_args(config, args)
    config = apply_episode_limit_args(config, args)
    config = apply_reproducibility_args(config, args)
    seed_everything(config.training_seed, deterministic=config.deterministic)
    config = apply_validation_schedule_args(config, args)
    config = apply_bottleneck_args(config, args)
    config = apply_edqn_dijkstra_guidance_args(config, args)
    mode = build_execution_mode(read_tmp_file=False)
    requested_output_dir = args.output_dir
    if args.fine_tune_checkpoint is not None and requested_output_dir is None:
        requested_output_dir = resolve_variant_output_dir(config).parent / "edqn_finetune_revisit"
    output_dir = resolve_variant_output_dir(config, requested_output_dir)

    if args.resume_state is not None:
        marker_path = training_completed_marker_path(output_dir)
        if completed_marker_matches_config(config, output_dir):
            print(f"[info] completed training marker exists: {marker_path}")
            print("[info] nothing to resume; exiting normally.")
            return
        try:
            already_completed = resume_state_already_completed(config, args.resume_state)
        except Exception as exc:
            print(f"[warn] could not inspect resume state before SUMO startup: {exc}")
            already_completed = False
        if already_completed:
            print(
                f"[info] resume state already completed all episodes: "
                f"{args.resume_state}"
            )
            print("[info] nothing to resume; exiting normally.")
            return

    validate_environment_inputs(config)
    bootstrap_sumo_environment(config)
    session_info = start_sumo_session(config)

    try:

        print(describe_runtime(config))
        print(summarize_static_graph(config))

        # train_example = build_training_scenario(config, scenario_index=0, probability_index=1, seed=100)
        # test_example = build_testing_scenario(config, scenario_index=0, probability_index=1, seed=1)
        # sampled_key = choose_training_scenario_key(config)
        # resolved_key, resolved_train = resolve_training_scenario(config, sampled_key)
        # testing_pairs = resolve_testing_scenarios(config)
        # resolved_ids = resolve_ev_ids(None, train_example.ev_ids)
        # validation_preview = compare_episode_costs([10.0, 12.0], [10.0, 15.0], tolerance=1.0)
        # epsilon_scheduler = EpsilonScheduler(config.schedule, min_epsilon=0.05)
        # progress = TrainingProgress()
        # progress.record_scenario_hit(resolved_key.scenario_index)
        # execution_preview = preview_episode_execution(config, resolved_train)

        # print(f"training example: {train_example.ev_route_file}")
        # print(f"testing example: {test_example.ev_route_file}")
        # print(f"execution mode: {mode.value}")
        # print(f"sampled training key: {sampled_key}")
        # print(f"resolved training route: {resolved_train.ev_route_file}")
        # print(f"testing scenario count: {len(testing_pairs)}")
        # print(f"epsilon@0: {epsilon_scheduler.value(0):.2f}")
        # print(f"epsilon@warmup_end: {epsilon_scheduler.value(config.schedule.warmup_end):.2f}")
        # print(f"resolved ev count: {len(resolved_ids)}")
        # print(f"sample testing key count: {len(iter_testing_scenario_keys(config))}")
        # print(f"scenario hit map: {progress.scenario_hits}")
        # print(f"validation better ratio: {validation_preview.better_ratio:.2f}")
        # print(f"sumo command: {session_info['cmd']}")
        # print(f"sumo log dir: {session_info['log_dir']}")
        # print(f"load args preview: {execution_preview['load_args']}")
        # print(f"runner preview ev count: {execution_preview['ev_count']}")
        # print(f"runner preview goal count: {execution_preview['goal_count']}")
        # print(f"runner preview max_neighbors: {execution_preview['max_neighbors']}")

        log_dir = sumo_log_root(config)
        logger = EpisodeLogger(log_dir / "rollout_debug.txt")
        logger.start_episode(0)
        logger.close()


        bundle = load_runtime_bundle(config, logger=None)
        write_run_manifest(bundle, args, output_dir, session_info)
        print(f"agent type: {type(bundle.agent).__name__}")
        # print(f"training artifact keys: {list(vars(artifacts).keys())}")

        # =============  validation ==================
        validation = False
        if validation:
            artifacts = create_training_artifacts()
            episode = 999
            output_dir = bundle.config.repo_root / "outputs"
            checkpoint_path = bundle.config.repo_root / "models" / "using_all" / "dqngnn_with_EMA.pt"
            maybe_load_checkpoint(bundle.agent, checkpoint_path, validate_only=True)
            result = maybe_run_validation_and_checkpoint(bundle, artifacts, episode, output_dir)
            # result = run_validate_only_flow(bundle, checkpoint_path=checkpoint_path)
            print(f"better ratio {result['better_ratio']}")
            return

        # ============= training ===================
        artifacts = run_training_loop(
            bundle=bundle,
            start_episode=0,
            checkpoint_path=args.checkpoint,
            warm_start_checkpoint_path=args.fine_tune_checkpoint,
            resume_state_path=args.resume_state,
            max_episodes=args.max_episodes,
            scenario_schedule_seed=args.scenario_schedule_seed,
            output_dir=output_dir,
            autosave_state_path=args.autosave_state,
            autosave_every=args.autosave_every,
            rewards_save_every=args.rewards_save_every,
            verify_autosave_state=args.verify_autosave_state,
            close_sumo_before_save_enabled=args.close_sumo_before_save,
            training_validation_enabled=not args.disable_training_validation,
        )

        print("num rewards:", len(artifacts.rewards))
        print("last avg reward:", artifacts.average_rewards[-1] if artifacts.average_rewards else None)
        return
        

        print("training loop scaffold is ready: run_training_loop(bundle, checkpoint_path=...)")
        print("validate-only scaffold is ready: run_validate_only_flow(bundle, checkpoint_path=...)")
    finally:
        stop_sumo_session()


if __name__ == "__main__":
    main()

