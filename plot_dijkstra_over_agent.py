"""Compatibility entry point for :mod:`scripts.plot_dijkstra_over_agent`."""

from scripts.plot_dijkstra_over_agent import (
    build_output_paths,
    load_episode_rows,
    main,
    moving_average,
    parse_args,
    plot_raw,
    plot_with_average,
)

__all__ = [
    "parse_args",
    "load_episode_rows",
    "moving_average",
    "build_output_paths",
    "plot_raw",
    "plot_with_average",
    "main",
]

if __name__ == "__main__":
    main()
