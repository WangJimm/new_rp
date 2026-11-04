"""Compatibility shim; new code should import from :mod:`rp.routing`."""

from rp.routing.algorithms import cost_by_length, dijkstra_next_hop, dijkstra_path

__all__ = ["cost_by_length", "dijkstra_next_hop", "dijkstra_path"]
