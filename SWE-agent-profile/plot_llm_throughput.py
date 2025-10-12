#!/usr/bin/env python3
"""Plot LLM throughput by batch size as a bar chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_data(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(batch): value for batch, value in raw.items()}


def format_value(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M tokens/s"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k tokens/s"
    return f"{value:.0f} tokens/s"


def plot_throughput(data: dict[int, float], output: Path) -> None:
    batches = sorted(data.keys())
    throughputs = [data[b] for b in batches]
    positions = list(range(len(batches)))

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(
        positions,
        throughputs,
        width=0.6,
        color="#3b82f6",
        edgecolor="#1d4ed8",
    )

    ax.set_title("LLM Throughput by Batch Size", fontsize=16, fontweight="bold")
    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=14)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(b) for b in batches], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value in zip(bars, throughputs):
        ax.annotate(
            format_value(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1d3557",
        )

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data",
        nargs="?",
        default=Path("llm_throughput_by_batch.json"),
        type=Path,
        help="JSON file containing batch size to throughput mapping.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=Path("llm_throughput_by_batch.png"),
        type=Path,
        help="Output image file (PNG).",
    )
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")

    data = load_data(args.data)
    plot_throughput(data, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
