"""Compatibility shim; new code should import from :mod:`rp.reporting`."""

from rp.reporting import inspect_link_api, save_memory_to_txt

__all__ = ["save_memory_to_txt", "inspect_link_api"]
