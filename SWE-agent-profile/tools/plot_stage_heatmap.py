#!/usr/bin/env python3
"""Plot stage duration heatmap versus batch size for local non-blocking runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from aggregate_stage_timings import (  # type: ignore[import-untyped]
    STAGES,
    compute_instance_spans,
    find_run_directory,
    gather_instances,
    load_stage_events,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate stage durations for each batch size and plot a heatmap.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing run outputs (e.g., trajectories/<user>).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[4, 8, 16, 32, 64],
        help="Batch sizes to include when searching for run directories (default: %(default)s).",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="vllm_local",
        help="Substring that must be contained in run directories (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stage_duration_heatmap.png"),
        help="Path to save the heatmap image (default: %(default)s).",
    )
    parser.add_argument(
        "--stat",
        choices=["average", "sum"],
        default="average",
        help="Whether to aggregate durations per stage using an average or sum across instances (default: %(default)s).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of instances to skip after sorting by start time (default: %(default)s).",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Number of instances to include after skipping. If omitted, use all available instances.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information while processing runs.",
    )
    return parser.parse_args()


def _stage_durations_for_instance(inst_dir: Path) -> dict[str, float]:
    events = load_stage_events(inst_dir / "stage_timings.jsonl")
    spans = compute_instance_spans(events)
    durations = {stage: 0.0 for stage in STAGES}
    for stage, steps in spans.items():
        duration_sum = 0.0
        for enter_time, exit_time in steps.values():
            duration = max(0.0, exit_time - enter_time)
            duration_sum += duration
        durations[stage] = duration_sum
    return durations


def _aggregate_rows(
    instances: Iterable[Path],
    *,
    stat: str,
) -> list[float]:
    per_stage_values: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for inst_dir in instances:
        durations = _stage_durations_for_instance(inst_dir)
        for stage in STAGES:
            per_stage_values[stage].append(durations.get(stage, 0.0))
    row: list[float] = []
    for stage in STAGES:
        values = per_stage_values[stage]
        if not values:
            row.append(0.0)
            continue
        if stat == "sum":
            row.append(float(sum(values)))
        else:
            row.append(float(sum(values) / len(values)))
    return row


def _select_instances(instances: list[Path], *, skip: int, take: int | None) -> list[Path]:
    if not instances:
        return []
    sorted_instances = instances  # already sorted upstream
    start_idx = min(skip, len(sorted_instances))
    end_idx = len(sorted_instances) if take is None else min(start_idx + take, len(sorted_instances))
    return sorted_instances[start_idx:end_idx]


def sort_instances_by_start(instances: list[Path]) -> list[Path]:
    from aggregate_stage_timings import sort_instances_by_start as _sort  # type: ignore[import-untyped]

    return _sort(instances)


def plot_heatmap(matrix: np.ndarray, batch_sizes: list[int], output: Path, *, stat: str) -> None:
    fig, ax = plt.subplots(figsize=(1.5 * len(STAGES) + 2, 0.6 * len(batch_sizes) + 2))
    cmap = plt.get_cmap("YlOrRd")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(STAGES)))
    ax.set_xticklabels([stage.replace("_", " ").title() for stage in STAGES], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_yticklabels([f"bs{size}" for size in batch_sizes])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Batch Size")
    title_stat = "Average" if stat == "average" else "Total"
    ax.set_title(f"{title_stat} Stage Duration vs Batch Size")

    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
            color="white" if value > matrix.max() * 0.6 else "black",
            fontsize=10,
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Duration (s)" if stat == "sum" else "Average Duration (s)")

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    batch_sizes_found: list[int] = []
    rows: list[list[float]] = []

    for batch_size in args.batch_sizes:
        run_dir = find_run_directory(args.root, batch_size, args.run_prefix)
        if run_dir is None:
            if args.verbose:
                print(f"Warning: No run directory found for batch size {batch_size}")
            continue
        instances = gather_instances(run_dir)
        if not instances:
            if args.verbose:
                print(f"Warning: No instance data found in {run_dir}")
            continue
        instances = sort_instances_by_start(instances)
        selected = _select_instances(instances, skip=args.skip, take=args.take)
        if not selected:
            if args.verbose:
                print(f"Warning: No instances selected for batch size {batch_size}")
            continue
        if args.verbose:
            print(f"Batch size {batch_size}: using {len(selected)} instances from {run_dir}")
        row = _aggregate_rows(selected, stat=args.stat)
        rows.append(row)
        batch_sizes_found.append(batch_size)

    if not rows:
        raise SystemExit("No data available to plot heatmap. Check the root path and batch sizes.")

    matrix = np.array(rows, dtype=float)
    plot_heatmap(matrix, batch_sizes_found, args.output, stat=args.stat)
    print(f"Saved heatmap to {args.output}")


if __name__ == "__main__":
    main()
