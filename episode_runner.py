"""Compatibility shim; new code should import from :mod:`rp.experiment`."""

from rp.experiment.episode_runner import (
    AgentDecisionSnapshot,
    AgentLearningError,
    BottleneckController,
    BottleneckEvent,
    DQNGNNEpisodeRunner,
    EpisodeConfig,
    EpisodeRunInput,
    EpisodeRuntime,
    TransitionPayload,
    build_agent_snapshot,
    build_candidate_action_features,
    build_tls_state,
    candidate_lane_metrics,
    collect_candidate_debug_rows,
    dist_to_lane_end,
    handle_arrival,
    learn_if_needed,
    make_step_transition,
    make_terminal_transition,
    push_agent_transition,
    resolve_ev_ids,
    resolve_movement_out_lanes,
    should_make_decision,
    summarize_episode,
    summarize_episode_time_costs,
)

__all__ = [
    "AgentLearningError",
    "TransitionPayload",
    "EpisodeConfig",
    "EpisodeRunInput",
    "DQNGNNEpisodeRunner",
    "resolve_ev_ids",
]
