from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Segoe UI", "Calibri", "DejaVu Sans"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot dijkstra_over_agent ratios from training-vs-dijkstra JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to compare_training_rewards_to_dijkstra output JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving-average window size.",
    )
    return parser.parse_args()


def load_episode_rows(input_path: Path) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("Input JSON does not contain a valid 'episodes' list.")
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("--window must be > 0")
    result: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append(running_sum / len(queue))
    return result


def build_output_paths(input_path: Path, output_dir: Path | None) -> tuple[Path, Path]:
    base_dir = output_dir or input_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    raw_path = base_dir / f"{stem}_dijkstra_over_agent.png"
    avg_path = base_dir / f"{stem}_dijkstra_over_agent_avg.png"
    return raw_path, avg_path


def plot_raw(episodes: list[int], ratios: list[float], output_path: Path) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(episodes, ratios, linewidth=1.0, color="tab:blue")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, color="tab:red")
    plt.xlim(left=0, right=max(episodes) if episodes else 0)
    plt.xlabel("Episode", fontsize=27)
    plt.ylabel("Dijkstra / Agent", fontsize=27)
    plt.title("Per-Episode Dijkstra / Agent Ratio", fontsize=39)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.margins(x=0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_with_average(
    episodes: list[int],
    ratios: list[float],
    average_ratios: list[float],
    output_path: Path,
    window: int,
) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(episodes, ratios, linewidth=0.8, alpha=0.25, color="tab:blue", label="Per episode")
    plt.plot(episodes, average_ratios, linewidth=2.2, color="tab:orange", label=f"Moving average ({window})")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, color="tab:red", label="Parity")
    plt.xlim(left=0, right=max(episodes) if episodes else 0)
    plt.xlabel("Episode", fontsize=27)
    plt.ylabel("Dijkstra / Agent", fontsize=27)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.margins(x=0)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = load_episode_rows(args.input)

    filtered_rows = [
        row for row in rows
        if row.get("dijkstra_over_agent") is not None
    ]
    if not filtered_rows:
        raise ValueError("No valid dijkstra_over_agent values found in input JSON.")

    raw_episodes = [int(row["episode"]) for row in filtered_rows]
    episode_offset = raw_episodes[0]
    episodes = [ep - episode_offset for ep in raw_episodes]
    ratios = [float(row["dijkstra_over_agent"]) for row in filtered_rows]
    average_ratios = moving_average(ratios, args.window)
    raw_path, avg_path = build_output_paths(args.input, args.output_dir)

    plot_raw(episodes, ratios, raw_path)
    plot_with_average(episodes, ratios, average_ratios, avg_path, args.window)

    print(f"saved raw plot: {raw_path}")
    print(f"saved average plot: {avg_path}")


if __name__ == "__main__":
    main()

    # """
    # 使用方式:
    # C:\Users\ntpum\anaconda3\envs\osm\python.exe c:/Users/ntpum/OneDrive/桌面/Wang/route_planning/plot_dijkstra_over_agent.py `
    #     --input ".\outputs\gine_action_static_only\training_vs_dijkstra_sample.json" `
    #     --window 50 `
    #     --output-dir ".\outputs\gine_action_static_only\plots"

    # """
