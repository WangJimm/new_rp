from __future__ import annotations

import os
import sys
import sysconfig
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from .scenario_catalog import CatalogScenario, load_scenario_catalog

ENV_CHOICES = ("env", "new_env")


PROJECT_DIR = Path(__file__).resolve().parents[2]


def resolve_sumo_home() -> Path:
    candidates: list[Path] = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]).expanduser())
    candidates.extend([
        Path(sysconfig.get_paths()["purelib"]) / "sumo",
        Path(sys.prefix),
        Path("/usr/share/sumo"), Path("/usr/local/share/sumo"),
        Path(r"C:\Program Files\Eclipse\Sumo"),
        Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
    ])
    binary = shutil.which("sumo")
    if binary:
        candidates.append(Path(binary).resolve().parent.parent)
    executable = "sumo.exe" if os.name == "nt" else "sumo"
    for candidate in candidates:
        if any((folder / executable).is_file()
               for folder in (candidate / "bin", candidate / "Scripts", candidate)):
            return candidate.resolve()
    raise FileNotFoundError("SUMO executable not found. Install eclipse-sumo==1.24.0 or set SUMO_HOME.")


def resolve_sumo_config(run_dir: Path) -> Path:
    for filename in ("osm.sumocfg", "osm.sumocfg.xml"):
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return run_dir / "osm.sumocfg"


@dataclass(frozen=True)
class RPVariantSpec:
    name: str
    agent_type: str
    edge_feature_mode: str | None
    description: str


RP_VARIANT_SPECS: dict[str, RPVariantSpec] = {
    "dqn_baseline": RPVariantSpec(
        name="dqn_baseline",
        agent_type="dqn",
        edge_feature_mode=None,
        description="Non-GNN DQN baseline; Double target, dueling QNetwork, and PER.",
    ),
    "edqn_baseline": RPVariantSpec(
        name="edqn_baseline",
        agent_type="edqn",
        edge_feature_mode=None,
        description="Non-GNN eDQN baseline following HEDQN: Double target, competitive Q network, uniform replay, and Huber loss.",
    ),
    "ppo_baseline": RPVariantSpec(
        name="ppo_baseline",
        agent_type="ppo_rllib",
        edge_feature_mode=None,
        description="Non-GNN PPO baseline implemented with Ray RLlib; masked outgoing-neighbor policy with current/target/time state.",
    ),
    "ppo_torch_baseline": RPVariantSpec(
        name="ppo_torch_baseline",
        agent_type="ppo_torch",
        edge_feature_mode=None,
        description="Non-GNN PPO baseline implemented with pure PyTorch; masked outgoing-neighbor policy with current/target/time state.",
    ),
    "sacd_baseline": RPVariantSpec(
        name="sacd_baseline",
        agent_type="sacd",
        edge_feature_mode=None,
        description="Non-GNN discrete SAC baseline; replay-buffer actor-critic with masked outgoing-neighbor policy and current/target/time state.",
    ),
    "gine_no_edge_features": RPVariantSpec(
        name="gine_no_edge_features",
        agent_type="gine_action",
        edge_feature_mode="none",
        description="GINE graph encoder with action-centric Q; edge_attr is zeroed.",
    ),
    "gine_fixed_action_head": RPVariantSpec(
        name="gine_fixed_action_head",
        agent_type="gine",
        edge_feature_mode="static_occupancy",
        description="GINE graph encoder with fixed max-neighbor Q head; no action-centric Q network.",
    ),
    "proposed": RPVariantSpec(
        name="proposed",
        agent_type="gine_action",
        edge_feature_mode="static_occupancy",
        description="Current full RP agent: GINE encoder, static+occupancy edge features, action-centric Q.",
    ),
}
RP_VARIANT_CHOICES = tuple(RP_VARIANT_SPECS.keys())


def resolve_rp_variant_spec(rp_variant: str) -> RPVariantSpec:
    try:
        return RP_VARIANT_SPECS[rp_variant]
    except KeyError as exc:
        raise ValueError(f"Unknown rp_variant: {rp_variant}. Valid variants: {RP_VARIANT_CHOICES}") from exc


@dataclass(frozen=True)
class SumoPaths:
    sumo_home: Path
    run_dir: Path
    config_file: Path
    net_file: Path


@dataclass(frozen=True)
class TrainingSchedule:
    num_episodes: int = 1500
    warmup_ratio: float = 0.15
    training_ratio: float = 0.8
    warmup_episodes: int | None = None
    epsilon_decay_end_episode: int | None = None

    @property
    def warmup_end(self) -> int:
        if self.warmup_episodes is not None:
            return int(self.warmup_episodes)
        return int(self.num_episodes * self.warmup_ratio)

    @property
    def training_end(self) -> int:
        if self.epsilon_decay_end_episode is not None:
            return int(self.epsilon_decay_end_episode)
        return int(self.num_episodes * self.training_ratio)


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    cs_locations: Sequence[str]
    general_vehicle_flows: Sequence[str]
    training_vehicle_flows: Sequence[str] | None = None
    testing_vehicle_flows: Sequence[str] | None = None
    validation_vehicle_flows: Sequence[str] | None = None
    training_seed_start: int = 100
    training_seed_end: int = 150
    testing_seed_start: int = 1
    testing_seed_end: int = 10
    validation_seed_start: int | None = None
    validation_seed_end: int | None = None
    scenario_catalog: tuple[CatalogScenario, ...] = ()
    scenario_manifest: Path | None = None

    @property
    def training_flows(self) -> Sequence[str]:
        return self.training_vehicle_flows or self.general_vehicle_flows

    @property
    def testing_flows(self) -> Sequence[str]:
        return self.testing_vehicle_flows or self.general_vehicle_flows

    @property
    def validation_flows(self) -> Sequence[str]:
        return self.validation_vehicle_flows or self.testing_flows

@dataclass(frozen=True)
class RuntimeConfig:
    repo_root: Path
    paths: SumoPaths
    environment: EnvironmentConfig
    schedule: TrainingSchedule = field(default_factory=TrainingSchedule)
    max_decisions_per_ev: int = 40
    max_simulation_steps: int = 6000
    memory_learning_threshold: int = 100
    memory_push_learning_threshold: int = 100
    sumo_restart_interval_episodes: int | None = 200
    sumo_load_retry_count: int = 1
    enable_bottleneck: bool = False
    bottleneck_start_time_sec: float = 120.0
    bottleneck_end_time_sec: float = 900.0
    bottleneck_speed_factor: float = 0.5
    bottleneck_candidate_edges: Sequence[str] = field(default_factory=tuple)
    step_length: float = 0.5
    decision_distance_threshold: float = 100.0
    goal_arrive_distance: float = 2.0
    success_bonus: float = 3.0
    uturn_penalty: float = 1.0
    revisit_penalty_enabled: bool = True
    revisit_penalty: float = 1.0
    dead_end_penalty: float = 20.0
    potential_shaping_enabled: bool = False
    potential_shaping_gamma: float = 0.99
    potential_distance_scale: float = 1000.0
    edqn_dijkstra_guidance_enabled: bool = False
    edqn_dijkstra_guide_probability: float = 0.7
    edqn_dijkstra_guide_hold_episodes: int = 500
    edqn_dijkstra_guide_decay_episodes: int = 1000
    use_gui: bool = False
    use_gnn: bool = True
    # Start SACD from the same current/target/time state as eDQN. Enable this
    # only as a separate state ablation after the clean baseline is stable.
    sacd_use_action_features: bool = False
    sacd_replay_alpha: float = 0.0
    sacd_target_entropy_ratio: float = 0.30
    sacd_use_epsilon_exploration: bool = False
    rp_variant: str = "proposed"
    validation_start_episode: int | None = None
    validation_interval: int | None = None


def build_small_env_config(repo_root: Path, use_gnn = True, use_gui = False) -> RuntimeConfig:
    route_root = PROJECT_DIR
    sumo_home = resolve_sumo_home()
    run_dir = route_root / "small_env" 


    return RuntimeConfig(
        repo_root=repo_root,
        paths=SumoPaths(
            sumo_home=sumo_home,
            run_dir=run_dir,
            config_file=resolve_sumo_config(run_dir),
            net_file=run_dir / "small_env.net.xml",
        ),
        environment=EnvironmentConfig(
            name="small_env",
            cs_locations=[
                "42496394",
                "cluster_1275778430_42504174_9744131154",
                "42907440",
            ],
            general_vehicle_flows=[
                "free_0900",
                "norm_1500",
                "busy_2100",
                "heavy_2400",
            ],
            training_vehicle_flows=[
                "free_0900",
                "norm_1500",
                "busy_2100",
            ],
            testing_vehicle_flows=[
                "free_0900",
                "norm_1500",
                "busy_2100",
                "heavy_2400",
            ],
        ),
        use_gui=use_gui,
        use_gnn=use_gnn,
    )

def build_env_config(repo_root: Path, use_gnn = True, use_gui = False) -> RuntimeConfig:
    route_root = PROJECT_DIR
    sumo_home = resolve_sumo_home()
    run_dir = route_root / "env" 


    return RuntimeConfig(
        repo_root=repo_root,
        paths=SumoPaths(
            sumo_home=sumo_home,
            run_dir=run_dir,
            config_file=resolve_sumo_config(run_dir),
            net_file=run_dir / "env.net.xml",
        ),
        environment=EnvironmentConfig(
            name="env",
            cs_locations=[
                "42496394",
                "42490303",
                "42474000",
            ],
            general_vehicle_flows=[
                "1800",
                # "2100",
                # "2400",
            ],
            training_vehicle_flows=[
                "1800",
                # "2100",
                # "2400",
            ],
            testing_vehicle_flows=[
                "1800",
                "2100",
                "2400",
            ],
            training_seed_start = 1,
            training_seed_end=20,
            testing_seed_start = 51,
            testing_seed_end = 60,
        ),
        use_gui=use_gui,
        use_gnn=use_gnn,
    )

def build_static_env_config(repo_root: Path, use_gnn = True, use_gui = False) -> RuntimeConfig:
    route_root = Path(repo_root)
    sumo_home = resolve_sumo_home()
    run_dir = route_root / "env" # 要使用哪種環境，指定到env


    config = RuntimeConfig(
        repo_root=repo_root,
        paths=SumoPaths(
            sumo_home=sumo_home,
            run_dir=run_dir,
            config_file=resolve_sumo_config(run_dir),
            net_file=run_dir / "env_static_tls.net.xml",
        ),
        environment=EnvironmentConfig(
            name="env",
            cs_locations=[
                "42496394",
                "42490303",
                "42474000",
            ],
            general_vehicle_flows=[
                "1800",
                # "2100",
                # "2400",
            ],
            training_vehicle_flows=[
                "1800",
                # "2100",
                # "2400",
            ],
            testing_vehicle_flows=[
                "1500",
                "1800",
                "2100",
            ],
            validation_vehicle_flows = [
                "1500",
                "1800",
                "2100",
            ],
            training_seed_start = 1, # 1
            training_seed_end=20,# 20
            testing_seed_start = 51, # 51
            testing_seed_end = 60, # 60
            validation_seed_start = 201, # 201
            validation_seed_end = 210, # 210
        ),
        use_gui=use_gui,
        use_gnn=use_gnn,
    )

    # New partitions can be deployed without hardcoding sparse generator seeds.
    # Keep the legacy ranges available when no manifest has been installed.
    manifest = run_dir / "scenario_manifest.csv"
    if manifest.is_file():
        catalog = load_scenario_catalog(manifest)
        flows = {
            split: tuple(sorted({r.flow_name for r in catalog if r.split == split}, key=int))
            for split in ("training", "validation", "testing")
        }
        config = replace(config, environment=replace(
            config.environment,
            general_vehicle_flows=tuple(sorted({r.flow_name for r in catalog}, key=int)),
            training_vehicle_flows=flows["training"],
            validation_vehicle_flows=flows["validation"],
            testing_vehicle_flows=flows["testing"],
            scenario_catalog=catalog,
            scenario_manifest=manifest,
        ))
    return config


def build_new_env_config(repo_root: Path, use_gnn=True, use_gui=False) -> RuntimeConfig:
    run_dir = Path(repo_root) / "new_env"
    manifest = run_dir / "scenario_manifest.csv"
    catalog = load_scenario_catalog(manifest)
    flows = {
        split: tuple(sorted({r.flow_name for r in catalog if r.split == split}, key=int))
        for split in ("training", "validation", "testing")
    }
    return RuntimeConfig(
        repo_root=repo_root,
        paths=SumoPaths(
            sumo_home=resolve_sumo_home(), run_dir=run_dir,
            config_file=resolve_sumo_config(run_dir),
            net_file=run_dir / "env_static_tls.net.xml",
        ),
        environment=EnvironmentConfig(
            name="new_env",
            cs_locations=("1419061109", "446290091", "5984003368"),
            general_vehicle_flows=tuple(sorted({r.flow_name for r in catalog}, key=int)),
            training_vehicle_flows=flows["training"], testing_vehicle_flows=flows["testing"],
            validation_vehicle_flows=flows["validation"],
            scenario_catalog=catalog, scenario_manifest=manifest,
        ),
        use_gui=use_gui, use_gnn=use_gnn,
    )


def build_selected_env_config(repo_root: Path, env_name="env", use_gnn=True, use_gui=False) -> RuntimeConfig:
    if env_name == "env":
        return build_static_env_config(repo_root, use_gnn=use_gnn, use_gui=use_gui)
    if env_name == "new_env":
        return build_new_env_config(repo_root, use_gnn=use_gnn, use_gui=use_gui)
    raise ValueError(f"Unknown environment: {env_name}; choose from {ENV_CHOICES}")


def validate_environment_inputs(config: RuntimeConfig) -> None:
    """Check manifest-backed environments before starting a SUMO process."""
    if config.environment.name != "new_env" and not config.environment.scenario_catalog:
        return
    import xml.etree.ElementTree as ET
    from .scenario_loader import build_training_scenario, build_validation_scenario, build_testing_scenario
    for path in (config.paths.net_file, config.paths.config_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {config.environment.name} input: {path}")
    net_root = ET.parse(config.paths.net_file).getroot()
    nodes = {node.get("id") for node in net_root.findall("junction")}
    missing = set(config.environment.cs_locations) - nodes
    if missing:
        raise ValueError(f"Charging-station nodes absent from the network: {sorted(missing)}")
    sumo_root = ET.parse(config.paths.config_file).getroot()
    net_setting = sumo_root.find("input/net-file")
    if net_setting is None or (config.paths.config_file.parent / net_setting.get("value", "")).resolve() != config.paths.net_file.resolve():
        raise ValueError("SUMO config and RuntimeConfig must reference the same network")
    builders = {"training": build_training_scenario, "validation": build_validation_scenario, "testing": build_testing_scenario}
    for record in config.environment.scenario_catalog:
        flows = getattr(config.environment, f"{record.split}_flows")
        spec = builders[record.split](config, flows.index(record.flow_name), record.probability_index, record.seed)
        invalid = set(spec.goal_nodes.values()) - set(config.environment.cs_locations)
        if invalid:
            raise ValueError(f"Unexpected goals in {spec.goal_file}: {sorted(invalid)}")


__all__ = [
    "PROJECT_DIR",
    "ENV_CHOICES",
    "build_new_env_config",
    "build_selected_env_config",
    "validate_environment_inputs",
    "resolve_sumo_home",
    "resolve_sumo_config",
    "RPVariantSpec",
    "RP_VARIANT_SPECS",
    "RP_VARIANT_CHOICES",
    "resolve_rp_variant_spec",
    "SumoPaths",
    "TrainingSchedule",
    "EnvironmentConfig",
    "RuntimeConfig",
    "build_small_env_config",
    "build_env_config",
    "build_static_env_config",
]
