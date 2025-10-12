"""Aggregate stage timing logs for different batch sizes and plot stacked shares."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

STAGES = ["env_prepare", "llm_prefill", "llm_decode", "tool_execution", "observation_packaging"]
LEGACY_STAGE_ALIASES = {"llm_reasoning": "llm_decode"}
COLORS = {
    "env_prepare": "#4C72B0",
    "llm_prefill": "#55A868",
    "llm_decode": "#73AF48",
    "tool_execution": "#C44E52",
    "observation_packaging": "#8172B2",
}
COLORS.setdefault("llm_reasoning", COLORS["llm_decode"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute shared stage durations for different batch sizes and plot stacked shares.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Parent directory containing run outputs (each run directory includes stage_timings.jsonl files).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[4, 8, 16, 32],
        help="Batch sizes to analyse. Directories whose names end with 'bs{size}' will be used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("stage_time_shares.png"),
        help="Path to save the resulting stacked bar chart.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="vllm_local",
        help="Only consider run directories whose name contains this prefix (default: 'vllm_local').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed warnings for missing or malformed data.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of instances to skip from the beginning after sorting by start time.",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Number of instances to analyze after skipping. If omitted, use all remaining instances.",
    )
    return parser.parse_args()


def find_run_directory(root: Path, batch_size: int, prefix: str | None) -> Path | None:
    pattern = f"bs{batch_size}"
    candidates = [p for p in root.rglob(f"*{pattern}") if p.is_dir()]
    if prefix:
        candidates = [p for p in candidates if prefix in p.name]
    if not candidates:
        return None
    # pick the most recent directory (in case multiple matches exist)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_stage_events(stage_file: Path) -> dict[str, dict[int, dict[str, list[float]]]]:
    events: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: {"enter": [], "exit": []}))
    try:
        with stage_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage_raw = record.get("stage")
                stage = LEGACY_STAGE_ALIASES.get(stage_raw, stage_raw)
                phase = record.get("phase")
                if stage not in STAGES or phase not in {"enter", "exit"}:
                    continue
                step = record.get("step")
                timestamp = record.get("timestamp")
                if not isinstance(step, int) or timestamp is None:
                    continue
                try:
                    ts = float(timestamp)
                except (TypeError, ValueError):
                    continue
                events[stage][step][phase].append(ts)
    except FileNotFoundError:
        return {}
    return events


def compute_instance_spans(events: dict[str, dict[int, dict[str, list[float]]]]) -> dict[str, dict[int, tuple[float, float]]]:
    result: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for stage, steps in events.items():
        for step, phases in steps.items():
            enters = phases.get("enter", [])
            exits = phases.get("exit", [])
            if not enters or not exits:
                continue
            enter_time = min(enters)
            exit_time = max(exits)
            if exit_time < enter_time:
                continue
            result[stage][step] = (enter_time, exit_time)
    return result


def aggregate_shared_times(instances: list[Path], expected_size: int | None = None, verbose: bool = False) -> dict[str, float]:
    # stage -> step -> list of (enter, exit)
    grouped: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for inst_dir in instances:
        stage_file = inst_dir / "stage_timings.jsonl"
        events = load_stage_events(stage_file)
        if not events and verbose:
            print(f"Warning: No stage events for {inst_dir}")
        spans = compute_instance_spans(events)
        for stage, steps in spans.items():
            for step, span in steps.items():
                grouped[stage][step].append(span)

    totals = {stage: 0.0 for stage in STAGES}
    for stage in STAGES:
        steps = grouped.get(stage, {})
        for step, spans in steps.items():
            enters = [span[0] for span in spans]
            exits = [span[1] for span in spans]
            if not enters or not exits:
                continue
            earliest = min(enters)
            latest = max(exits)
            if expected_size and len(spans) < expected_size and verbose:
                print(
                    f"Warning: stage {stage} step {step} only has {len(spans)} spans (expected {expected_size})."
                )
            shared_window = latest - earliest if latest >= earliest else 0.0
            duration_sum = sum(max(0.0, exit_time - enter_time) for enter_time, exit_time in spans)
            totals[stage] += min(shared_window, duration_sum)
    return totals


def gather_instances(run_dir: Path) -> list[Path]:
    return [p for p in run_dir.iterdir() if p.is_dir() and (p / "stage_timings.jsonl").exists()]


def sort_instances_by_start(instances: list[Path]) -> list[Path]:
    def first_timestamp(inst_dir: Path) -> float:
        stage_file = inst_dir / "stage_timings.jsonl"
        try:
            with stage_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = record.get("timestamp")
                    if ts is None:
                        continue
                    try:
                        return float(ts)
                    except (TypeError, ValueError):
                        continue
        except FileNotFoundError:
            pass
        return float("inf")

    return sorted(instances, key=first_timestamp)


def normalize_totals(totals: dict[str, float]) -> dict[str, float]:
    total_time = sum(totals.values())
    if total_time == 0:
        return {stage: 0.0 for stage in STAGES}
    return {stage: totals[stage] / total_time * 100.0 for stage in STAGES}


def plot_shares(batch_sizes: list[int], shares: list[dict[str, float]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    positions = np.arange(len(batch_sizes))
    bottoms = np.zeros(len(batch_sizes))
    bar_width = 0.65

    for stage in STAGES:
        values = np.array([shares[idx].get(stage, 0.0) for idx in range(len(batch_sizes))], dtype=float)
        ax.bar(
            positions,
            values,
            bar_width,
            bottom=bottoms,
            label=stage.replace("_", " ").title(),
            color=COLORS.get(stage, None),
            edgecolor="white",
            linewidth=0.8,
        )

        for idx, value in enumerate(values):
            if value <= 0:
                continue
            y = bottoms[idx] + value / 2
            ax.text(
                positions[idx] - bar_width / 2 - 0.08,
                y,
                f"{value:.1f}%",
                ha="right",
                va="center",
                fontsize=11,
                color="#1f2d3a",
            )
        bottoms += values

    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel("Stage Share (%)", fontsize=14)
    ax.set_title("Shared Stage Time Distribution", fontsize=16, pad=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(batch_sizes)
    ax.set_ylim(0, 101)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    shares_per_batch: list[dict[str, float]] = []
    available_batches: list[int] = []

    for batch_size in args.batch_sizes:
        run_dir = find_run_directory(args.root, batch_size, args.run_prefix)
        if run_dir is None:
            if args.verbose:
                print(f"Warning: No run directory found for batch size {batch_size}")
            continue
        all_instances = sort_instances_by_start(gather_instances(run_dir))
        if not all_instances:
            if args.verbose:
                print(f"Warning: No instance directories found for batch size {batch_size}")
            continue
        start_idx = min(args.skip, len(all_instances))
        end_idx = len(all_instances) if args.take is None else min(start_idx + args.take, len(all_instances))
        instances = all_instances[start_idx:end_idx]
        if args.verbose:
            print(
                f"Batch size {batch_size}: using instances[{start_idx}:{end_idx}]/{len(all_instances)}"
            )
        if len(instances) < batch_size and args.verbose:
            print(
                f"Warning: Not enough instances ({len(instances)}) for batch size {batch_size}; using all available instead"
            )

        group_step = batch_size if len(instances) >= batch_size else max(1, len(instances))
        totals = {stage: 0.0 for stage in STAGES}
        for idx in range(0, len(instances), group_step):
            group = instances[idx : idx + group_step]
            if not group:
                continue
            group_totals = aggregate_shared_times(
                group,
                expected_size=len(group),
                verbose=args.verbose,
            )
            for stage in STAGES:
                totals[stage] += group_totals.get(stage, 0.0)
        shares = normalize_totals(totals)
        shares_per_batch.append(shares)
        available_batches.append(batch_size)
        if args.verbose:
            print(f"Batch size {batch_size}: totals={totals}")

    if not shares_per_batch:
        raise SystemExit("No batch size data found. Check --root path and directory naming.")

    plot_shares(available_batches, shares_per_batch, args.output)
    print(f"Saved stage share chart to {args.output}")


if __name__ == "__main__":
    main()
