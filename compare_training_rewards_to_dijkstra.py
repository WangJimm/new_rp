"""Compatibility entry point for the training-vs-Dijkstra comparison CLI."""

from scripts.compare_training_rewards_to_dijkstra import (
    build_episode_result,
    compute_agent_cost,
    load_records,
    main,
    parse_args,
    restart_runtime,
    slice_records,
    summarize,
)

__all__ = [
    "parse_args",
    "load_records",
    "slice_records",
    "compute_agent_cost",
    "build_episode_result",
    "summarize",
    "restart_runtime",
    "main",
]

if __name__ == "__main__":
    main()
