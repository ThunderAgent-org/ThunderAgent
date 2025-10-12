#!/usr/bin/env python3
"""Visualize worker timelines within a selected time window."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt  # type: ignore[import-not-found]
from matplotlib.lines import Line2D  # type: ignore[import-not-found]

from aggregate_stage_timings import (  # type: ignore[import-untyped]
    COLORS,
    LEGACY_STAGE_ALIASES,
    STAGES,
    find_run_directory,
    gather_instances,
)

MERGED_STAGE = "tool"
MERGED_SOURCE_STAGES = {"tool_execution", "observation_packaging"}
STAGE_ALIAS = {
    stage: (MERGED_STAGE if stage in MERGED_SOURCE_STAGES else stage)
    for stage in STAGES + list(LEGACY_STAGE_ALIASES.keys())
}

DISPLAY_STAGES: list[str] = []
for stage in STAGES:
    alias = STAGE_ALIAS.get(stage, stage)
    if alias not in DISPLAY_STAGES:
        DISPLAY_STAGES.append(alias)
if MERGED_STAGE not in DISPLAY_STAGES:
    DISPLAY_STAGES.append(MERGED_STAGE)

DISPLAY_COLORS = {}
for stage in DISPLAY_STAGES:
    if stage == MERGED_STAGE:
        DISPLAY_COLORS[stage] = COLORS.get("tool_execution", "#C44E52")
    else:
        DISPLAY_COLORS[stage] = COLORS.get(stage, "#999999")

COLOR_OVERRIDES = {
    "env_prepare": "#6A4C93",  # purple
    "llm_prefill": "#FFD166",  # bright golden
    "llm_decode": "#26547C",   # deep blue
    MERGED_STAGE: "#C81D25",     # vivid crimson
    "observation_packaging": "#1B998B",  # teal
}

for stage, color in COLOR_OVERRIDES.items():
    if stage in DISPLAY_COLORS:
        DISPLAY_COLORS[stage] = color


class RawSegment(NamedTuple):
    worker: str
    stage: str
    start: float
    end: float
    instance: str
    step: int
    attempt: int


class Segment(NamedTuple):
    start: float
    duration: float
    stage: str
    instance: str
    step: int
    attempt: int


@dataclass
class WorkerRow:
    label: str
    worker: str
    segments: list[Segment]


_THREAD_PATTERN = re.compile(r".*?(?:ThreadPoolExecutor-)?(\d+)(?:_(\d+))?")


def _worker_sort_tuple(worker: str) -> tuple[int, int, str]:
    match = _THREAD_PATTERN.match(worker)
    if match:
        pool = int(match.group(1))
        thread = int(match.group(2)) if match.group(2) is not None else -1
        return (pool, thread, worker)
    return (1 << 30, 1 << 30, worker)


def _worker_display_index(worker: str, default: int) -> int:
    match = _THREAD_PATTERN.match(worker)
    if match:
        if match.group(2) is not None:
            return int(match.group(2))
        return int(match.group(1))
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot worker activity timelines within a chosen time window.",
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
        default=[4, 8, 16, 32],
        help="Batch sizes to analyse. Defaults to 4, 8, 16, 32.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="",
        help=(
            "Substring that must appear in the run directory name."
            " Leave empty to accept any directory (default: '%(default)s')."
        ),
    )
    parser.add_argument(
        "--time-start",
        type=float,
        default=0.0,
        help="Start of the time window (seconds from run start, default: %(default)s).",
    )
    parser.add_argument(
        "--time-end",
        type=float,
        default=None,
        help="End of the time window (seconds from run start). If omitted, use the full duration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("worker_timeline.png"),
        help="Path to save the timeline heatmap image (default: %(default)s).",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=28.0,
        help="Figure width in inches (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="Output DPI for the saved figure (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information while processing runs.",
    )
    return parser.parse_args()


def _canonical_stage(stage: str) -> str:
    stage = LEGACY_STAGE_ALIASES.get(stage, stage)
    return STAGE_ALIAS.get(stage, stage)


def _load_raw_segments(inst_dir: Path, verbose: bool = False) -> list[RawSegment]:
    stage_file = inst_dir / "stage_timings.jsonl"
    pending: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    raw_segments: list[RawSegment] = []
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
                phase = record.get("phase")
                if not stage_raw or phase not in {"enter", "exit"}:
                    continue
                stage = _canonical_stage(str(stage_raw))
                if stage not in DISPLAY_STAGES:
                    continue
                thread = record.get("thread") or "unknown"
                try:
                    timestamp = float(record["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue
                step = int(record.get("step", 0) or 0)
                attempt = int(record.get("attempt", 0) or 0)
                instance_id = record.get("instance_id") or inst_dir.name
                key = (thread, stage, step, attempt)
                if phase == "enter":
                    pending[key].append(timestamp)
                else:
                    enters = pending.get(key)
                    if not enters:
                        if verbose:
                            print(f"[warn] Exit without matching enter: {inst_dir} {key}")
                        continue
                    enter_time = enters.pop()
                    if not enters:
                        pending.pop(key, None)
                    end_time = timestamp
                    if end_time <= enter_time:
                        continue
                    raw_segments.append(
                        RawSegment(
                            worker=str(thread),
                            stage=stage,
                            start=enter_time,
                            end=end_time,
                            instance=instance_id,
                            step=step,
                            attempt=attempt,
                        )
                    )
    except FileNotFoundError:
        if verbose:
            print(f"[warn] Missing stage timings: {stage_file}")
    return raw_segments


def _load_step_ratios(inst_dir: Path, verbose: bool = False) -> dict[int, tuple[float, float]]:
    path = inst_dir / "prefix_cache_metrics.jsonl"
    ratios: dict[int, tuple[float, float]] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("sample") == "baseline":
                    continue
                try:
                    step = int(record.get("step", 0) or 0)
                except (TypeError, ValueError):
                    continue
                prefill_delta = float(record.get("prefill_time_sum_delta", 0.0))
                decode_delta = float(record.get("decode_time_sum_delta", 0.0))
                total_delta = prefill_delta + decode_delta
                if total_delta > 0:
                    prefill_share = max(0.0, min(1.0, prefill_delta / total_delta))
                    decode_share = max(0.0, min(1.0, decode_delta / total_delta))
                else:
                    prefill_share = 0.0
                    decode_share = 1.0
                ratios[step] = (prefill_share, decode_share)
    except FileNotFoundError:
        if verbose:
            print(f"[warn] Missing prefix cache metrics: {path}")
    return ratios


def _group_segments_by_worker(
    raw_segments: Iterable[RawSegment],
    ratio_lookup: dict[tuple[str, int], tuple[float, float]],
    time_start: float,
    time_end: float | None,
) -> tuple[dict[str, list[Segment]], float]:
    segments = list(raw_segments)
    if not segments:
        return {}, 0.0
    reference = min(seg.start for seg in segments)
    run_end = max(seg.end for seg in segments) - reference
    window_start = max(0.0, time_start)
    window_end = time_end if time_end is not None else run_end
    window_end = max(window_start, min(window_end, run_end))

    grouped: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        start_rel = seg.start - reference
        end_rel = seg.end - reference
        if end_rel <= window_start or start_rel >= window_end:
            continue
        clip_start = max(start_rel, window_start)
        clip_end = min(end_rel, window_end)
        duration = clip_end - clip_start
        if duration <= 0:
            continue
        grouped[seg.worker].append(
            Segment(
                start=clip_start,
                duration=duration,
                stage=seg.stage,
                instance=seg.instance,
                step=seg.step,
                attempt=seg.attempt,
            )
        )

    if not grouped:
        return {}, window_end - window_start

    ordered: dict[str, list[Segment]] = {}
    for worker, segs in grouped.items():
        segs.sort(key=lambda s: s.start)
        ordered[worker] = segs
    ordered = dict(
        sorted(
            ordered.items(),
            key=lambda item: min(seg.start for seg in item[1]),
        )
    )

    def split_segments(segs: list[Segment]) -> list[Segment]:
        expanded: list[Segment] = []
        for seg in segs:
            if seg.stage == "llm_decode":
                pref_share, dec_share = ratio_lookup.get((seg.instance, seg.step), (0.0, 1.0))
                pref_share = max(0.0, min(1.0, pref_share))
                dec_share = max(0.0, min(1.0, dec_share))
                total_share = pref_share + dec_share
                if total_share > 0:
                    pref_share /= total_share
                    dec_share /= total_share
                else:
                    pref_share = 0.0
                    dec_share = 1.0
                prefill_duration = seg.duration * pref_share
                decode_duration = seg.duration - prefill_duration
                current_start = seg.start
                if prefill_duration > 1e-9:
                    expanded.append(
                        Segment(
                            start=current_start,
                            duration=prefill_duration,
                            stage="llm_prefill",
                            instance=seg.instance,
                            step=seg.step,
                            attempt=seg.attempt,
                        )
                    )
                    current_start += prefill_duration
                if decode_duration > 1e-9:
                    expanded.append(
                        Segment(
                            start=current_start,
                            duration=decode_duration,
                            stage="llm_decode",
                            instance=seg.instance,
                            step=seg.step,
                            attempt=seg.attempt,
                        )
                    )
            else:
                expanded.append(seg)
        return expanded

    for worker, segs in ordered.items():
        ordered[worker] = split_segments(segs)

    max_time = 0.0
    for segs in ordered.values():
        for seg in segs:
            max_time = max(max_time, seg.start + seg.duration)
    return ordered, max_time


def _collect_worker_rows(
    root: Path,
    batch_sizes: Iterable[int],
    run_prefix: str | None,
    time_start: float,
    time_end: float | None,
    verbose: bool = False,
) -> tuple[list[WorkerRow], float]:
    rows: list[WorkerRow] = []
    global_max_time = 0.0
    for batch_size in batch_sizes:
        run_dir = find_run_directory(root, batch_size, run_prefix)
        if run_dir is None:
            if verbose:
                print(f"[warn] No run directory found for batch size {batch_size}")
            continue
        raw_segments: list[RawSegment] = []
        ratio_lookup: dict[tuple[str, int], tuple[float, float]] = {}
        for inst_dir in gather_instances(run_dir):
            raw_segments.extend(_load_raw_segments(inst_dir, verbose=verbose))
            ratios = _load_step_ratios(inst_dir, verbose=verbose)
            for step, shares in ratios.items():
                ratio_lookup[(inst_dir.name, step)] = shares
        if not raw_segments:
            if verbose:
                print(f"[warn] No stage segments found in {run_dir}")
            continue
        worker_segments, run_max = _group_segments_by_worker(raw_segments, ratio_lookup, time_start, time_end)
        if not worker_segments:
            if verbose:
                print(f"[warn] Time window produced no data for batch size {batch_size}")
            continue
        sorted_workers = sorted(worker_segments.items(), key=lambda item: _worker_sort_tuple(item[0]))
        for seq_idx, (worker, segments) in enumerate(sorted_workers):
            display_idx = _worker_display_index(worker, seq_idx)
            label = f"bs{batch_size}/Worker {display_idx} ({worker})"
            rows.append(WorkerRow(label=label, worker=worker, segments=segments))
            for seg in segments:
                global_max_time = max(global_max_time, seg.start + seg.duration)
    return rows, global_max_time


def plot_worker_timelines(
    rows: list[WorkerRow],
    output: Path,
    fig_width: float,
    dpi: int,
) -> None:
    if not rows:
        raise SystemExit("No worker timelines available to plot.")

    stage_palette = {stage: DISPLAY_COLORS.get(stage, "#999999") for stage in DISPLAY_STAGES}
    min_time = min((seg.start for row in rows for seg in row.segments), default=0.0)
    max_time = max((seg.start + seg.duration) for row in rows for seg in row.segments)

    fig_height = max(6, 0.6 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    y_positions = list(range(len(rows)))
    for idx, row in enumerate(rows):
        for seg in row.segments:
            ax.broken_barh(
                [(seg.start, seg.duration)],
                (idx - 0.4, 0.8),
                facecolors=stage_palette.get(seg.stage, "#999999"),
                edgecolors="none",
                linewidth=0.0,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row.label for row in rows])
    ax.set_xlabel("Time since run start (s)", fontsize=12)
    ax.set_ylabel("Workers grouped by batch size", fontsize=12)
    left = min_time
    right = max_time * 1.02 if max_time else 1
    if right <= left:
        right = left + 1
    ax.set_xlim(left, right)
    ax.set_title("Worker Stage Timelines", fontsize=14, pad=15)

    legend_handles = [Line2D([0], [0], color=stage_palette.get(stage, "#999999"), lw=6) for stage in DISPLAY_STAGES]
    legend_labels = [stage.replace("_", " ").title() for stage in DISPLAY_STAGES]
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Saved worker timeline heatmap to {output}")


def main() -> None:
    args = parse_args()
    rows, _ = _collect_worker_rows(
        root=args.root,
        batch_sizes=args.batch_sizes,
        run_prefix=args.run_prefix,
        time_start=args.time_start,
        time_end=args.time_end,
        verbose=args.verbose,
    )
    if not rows:
        raise SystemExit("No workers found for the requested configuration/time window.")
    plot_worker_timelines(rows, args.output, args.fig_width, args.dpi)


if __name__ == "__main__":
    main()
