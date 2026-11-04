"""Plotting helpers kept separate from SUMO and routing logic."""

from __future__ import annotations

from typing import Any


def simple_plot(
    values: Any,
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    filename: str = "tmp123.png",
    *,
    pyplot_api: Any = None,
) -> None:
    """Save a one-series line plot using the legacy styling."""
    if pyplot_api is None:
        import matplotlib.pyplot as pyplot_api

    pyplot_api.figure()
    pyplot_api.plot(values)
    pyplot_api.xlabel(xlabel)
    pyplot_api.ylabel(ylabel)
    pyplot_api.savefig(filename)
    pyplot_api.close()


__all__ = ["simple_plot"]
