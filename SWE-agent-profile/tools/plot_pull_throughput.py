#!/usr/bin/env python3
"""Plot docker pull throughput using measurement data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE_GB = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot docker pull throughput from measure_pull_throughput.py results.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="JSON file containing measurement results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pull_throughput.png"),
        help="Output path for the throughput bar chart (default: %(default)s).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of batch sizes to plot (default: all in results).",
    )
    return parser.parse_args()


def compute_throughput(num_images: int, elapsed_seconds: float) -> float:
    if num_images <= 0 or elapsed_seconds <= 0:
        return 0.0
    total_gb = num_images * IMAGE_SIZE_GB
    return total_gb / (elapsed_seconds / 3600.0)


def plot_throughput(batch_sizes: list[int], throughputs: list[float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(batch_sizes))
    ax.bar(positions, throughputs, width=0.6, color="#4C72B0", edgecolor="white", linewidth=0.8)

    y_offset = max(throughputs) * 0.02 if throughputs else 0.05
    for idx, value in enumerate(throughputs):
        ax.text(
            positions[idx],
            value + y_offset,
            f"{value:.1f} GB/h",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#20303c",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(batch_sizes)
    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel("Throughput (GB/h)", fontsize=14)
    ax.set_title("Docker Pull Throughput by Batch Size", fontsize=16, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Results file must contain a list of objects")
    return data


def filter_and_prepare(data: list[dict[str, Any]], subset: set[int] | None) -> tuple[list[int], list[float]]:
    batches: list[int] = []
    throughputs: list[float] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        batch_size = entry.get("batch_size")
        num_images = entry.get("num_images")
        elapsed = entry.get("elapsed_seconds")
        if not isinstance(batch_size, int) or not isinstance(num_images, int) or not isinstance(elapsed, (int, float)):
            continue
        if subset and batch_size not in subset:
            continue
        batches.append(batch_size)
        throughputs.append(compute_throughput(num_images, float(elapsed)))
    if not batches:
        raise ValueError("No matching entries found in results.")
    combined = sorted(zip(batches, throughputs), key=lambda x: x[0])
    sorted_batches, sorted_throughputs = map(list, zip(*combined))
    return sorted_batches, sorted_throughputs


def main() -> None:
    args = parse_args()

    if not args.results.exists():
        raise SystemExit(f"Results file {args.results} not found.")

    data = load_results(args.results)
    subset = set(args.batch_sizes) if args.batch_sizes else None
    batches, throughputs = filter_and_prepare(data, subset)
    plot_throughput(batches, throughputs, args.output)
    print(f"Saved throughput chart to {args.output}")


if __name__ == "__main__":
    main()
