"""Compatibility shim; new code should import from :mod:`rp.reporting`."""

from rp.reporting import simple_plot

__all__ = ["simple_plot"]
