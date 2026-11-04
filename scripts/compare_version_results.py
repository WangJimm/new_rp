from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two version snapshots.")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def compare_values(
    old: Any,
    new: Any,
    *,
    path: str,
    tolerance: float,
    differences: list[dict[str, Any]],
) -> None:
    if isinstance(old, bool) or isinstance(new, bool):
        if old != new:
            differences.append({"path": path, "old": old, "new": new})
        return
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if not math.isclose(float(old), float(new), rel_tol=0.0, abs_tol=tolerance):
            differences.append({"path": path, "old": old, "new": new})
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            child = f"{path}.{key}" if path else str(key)
            if key not in old:
                differences.append({"path": child, "old": "<missing>", "new": new[key]})
            elif key not in new:
                differences.append({"path": child, "old": old[key], "new": "<missing>"})
            else:
                compare_values(
                    old[key],
                    new[key],
                    path=child,
                    tolerance=tolerance,
                    differences=differences,
                )
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            differences.append(
                {"path": f"{path}.length", "old": len(old), "new": len(new)}
            )
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            compare_values(
                old_item,
                new_item,
                path=f"{path}[{index}]",
                tolerance=tolerance,
                differences=differences,
            )
        return
    if old != new:
        differences.append({"path": path, "old": old, "new": new})


def main() -> None:
    args = parse_args()
    old = json.loads(args.old.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    metadata_differences: list[dict[str, Any]] = []
    for key in ("variant", "source", "scenario_key", "random_seed", "device"):
        compare_values(
            old.get(key),
            new.get(key),
            path=key,
            tolerance=args.float_tolerance,
            differences=metadata_differences,
        )
    for key in (
        "checkpoint_sha256",
        "assets",
        "scenario_semantics_sha256",
        "topology",
    ):
        compare_values(
            old.get(key),
            new.get(key),
            path=key,
            tolerance=args.float_tolerance,
            differences=metadata_differences,
        )

    result_differences: list[dict[str, Any]] = []
    compare_values(
        old["result"],
        new["result"],
        path="result",
        tolerance=args.float_tolerance,
        differences=result_differences,
    )
    report = {
        "equivalent": not metadata_differences and not result_differences,
        "metadata_equivalent": not metadata_differences,
        "result_equivalent": not result_differences,
        "float_tolerance": args.float_tolerance,
        "metadata_difference_count": len(metadata_differences),
        "result_difference_count": len(result_differences),
        "metadata_differences": metadata_differences,
        "result_differences": result_differences,
        "old_elapsed_seconds": old.get("elapsed_seconds"),
        "new_elapsed_seconds": new.get("elapsed_seconds"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
