"""Diagnostic reporting utilities with no SUMO dependency."""

from __future__ import annotations

import os
from typing import Any, Iterable


def save_memory_to_txt(buffer: Iterable[Any], filename: str = "memory_buffer.txt") -> str:
    """Write replay-buffer items to the current directory in legacy format."""
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, "w", encoding="utf-8") as file:
        for index, item in enumerate(buffer):
            file.write(f"{index}: {str(item)}\n")

    print(f"[INFO] buffer 已儲存到 {filepath}")
    return filepath


def inspect_link_api(link: Any) -> None:
    """Print public callable methods and non-callable attributes."""
    methods = [
        name
        for name in dir(link)
        if callable(getattr(link, name)) and not name.startswith("_")
    ]
    attrs = [
        name
        for name in dir(link)
        if not callable(getattr(link, name)) and not name.startswith("_")
    ]
    print("METHODS:", methods)
    print("ATTRS:", attrs)


__all__ = ["save_memory_to_txt", "inspect_link_api"]
