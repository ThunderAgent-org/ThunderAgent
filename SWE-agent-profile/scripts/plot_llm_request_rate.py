#!/usr/bin/env python3
"""Compute and plot LLM request throughput (req/s) by batch size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

from compute_llm_throughput import (
    STAGE_FILE,
    chunk_instances,
    extract_batch_size,
    load_total_tokens,
    parse_stage_timings,
)


def compute_group_metrics(
    instances: Iterable[str],
    tokens_map: Dict[str, List[int]],
    timings_map: Dict[str, Dict[int, Tuple[float, float]]],
) -> Tuple[int, float]:
    """Return (request_count, total_time) for a concurrent group."""
    names = [name for name in instances if name in tokens_map and name in timings_map]
    if not names:
        return 0, 0.0

    max_step = 0
    for name in names:
        seq = tokens_map.get(name)
        timing = timings_map.get(name)
        if not seq or not timing:
            continue
        timing_steps = timing.keys()
        if not timing_steps:
            continue
        max_step = max(max_step, min(len(seq), max(timing_steps)))

    total_requests = 0
    total_time = 0.0

    for step in range(1, max_step + 1):
        enters: List[float] = []
        exits: List[float] = []
        active_names: List[str] = []
        for name in names:
            seq = tokens_map.get(name)
            timing = timings_map.get(name)
            if not seq or not timing:
                continue
            if step > len(seq) or step not in timing:
                continue
            enter_ts, exit_ts = timing[step]
            enters.append(enter_ts)
            exits.append(exit_ts)
            active_names.append(name)
        if not active_names or not enters or not exits:
            continue
        step_time = max(exits) - min(enters)
        if step_time <= 0:
            continue
        total_requests += len(active_names)
        total_time += step_time

    return total_requests, total_time


def compute_request_rates(
    root: Path,
    tokens_data: Dict[str, Dict[str, List[int]]],
) -> Dict[str, float]:
    results: Dict[str, float] = {}

    for batch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        batch_name = batch_dir.name
        try:
            batch_size = extract_batch_size(batch_name)
        except ValueError:
            continue

        tokens_map = tokens_data.get(batch_name)
        if not tokens_map:
            continue

        instances = sorted(
            p.name for p in batch_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        if not instances:
            continue

        timings_map: Dict[str, Dict[int, Tuple[float, float]]] = {}
        for name in instances:
            stage_path = batch_dir / name / STAGE_FILE
            if not stage_path.exists():
                continue
            timing = parse_stage_timings(stage_path)
            if timing:
                timings_map[name] = timing

        groups = chunk_instances(instances, batch_size)

        total_requests = 0
        total_time = 0.0
        for group in groups:
            group_requests, group_time = compute_group_metrics(group, tokens_map, timings_map)
            total_requests += group_requests
            total_time += group_time

        if total_time > 0 and total_requests > 0:
            results[str(batch_size)] = total_requests / total_time

    return results


def format_req(value: float) -> str:
    if value >= 1_000:
        return f"{value / 1_000:.2f}k req/s"
    return f"{value:.2f} req/s"


def plot_rates(rates: Dict[str, float], output_path: Path) -> None:
    if not rates:
        raise SystemExit("No request-rate data found to plot.")

    ordered = sorted((int(k), v) for k, v in rates.items())
    labels = [str(k) for k, _ in ordered]
    values = [v for _, v in ordered]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(positions, values, width=0.6, color="#2563eb", edgecolor="#1d4ed8")

    ax.set_title("LLM Request Throughput by Batch Size", fontsize=20, pad=16)
    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel("Requests per Second (req/s)", fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            format_req(value),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1f2933",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("test_data/with_cache"),
        help="Directory containing batch-size subdirectories.",
    )
    parser.add_argument(
        "--total-tokens-json",
        type=Path,
        default=Path("with_cache_total_tokens.json"),
        help="JSON file with per-step total tokens.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("llm_request_rate_by_batch.json"),
        help="Where to write the computed request rates.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("llm_request_rate_by_batch.png"),
        help="Output PNG file for the bar chart.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output (default: 2).",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root directory not found: {args.root}")
    if not args.total_tokens_json.exists():
        raise SystemExit(f"Total tokens JSON not found: {args.total_tokens_json}")

    tokens_data = load_total_tokens(args.total_tokens_json)
    rates = compute_request_rates(args.root, tokens_data)

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(rates, handle, indent=args.indent)
        handle.write("\n")

    plot_rates(rates, args.figure)
    print(
        f"Wrote request rates for {len(rates)} batch sizes to {args.output_json} "
        f"and chart to {args.figure}."
    )


if __name__ == "__main__":
    main()
