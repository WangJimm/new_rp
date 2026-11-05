"""File-only plotting helpers, independent of GUI backends and SUMO."""
from __future__ import annotations
from threading import RLock
from typing import Any

# Matplotlib shares caches even between independent figures.
_PLOT_LOCK = RLock()


def simple_plot(
    values: Any,
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    filename: str = "tmp123.png",
    *,
    pyplot_api: Any = None,
) -> None:
    """Write a PNG with an explicit Agg canvas, without importing pyplot/Tk."""
    with _PLOT_LOCK:
        if pyplot_api is not None:
            # Preserve the explicitly injected adapter used by compatibility callers.
            pyplot_api.figure()
            try:
                pyplot_api.plot(values)
                pyplot_api.xlabel(xlabel)
                pyplot_api.ylabel(ylabel)
                pyplot_api.savefig(filename)
            finally:
                pyplot_api.close()
            return

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure()
        FigureCanvasAgg(figure)
        try:
            axes = figure.subplots()
            axes.plot(values)
            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            figure.savefig(filename)
        finally:
            figure.clear()


__all__ = ["simple_plot"]
