#!/usr/bin/env python3
"""Plot per-step average prefill/decode times across all instances in a run."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore[import-not-found]

PREFILL_COLOR = "#FFD166"  # bright golden
DECODE_COLOR = "#26547C"   # deep blue


def load_instance_prefix_metrics(prefix_file: Path) -> Dict[int, Tuple[float, float]]:
    per_step: Dict[int, Tuple[float, float]] = {}
    with prefix_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = int(record.get("step", 0))
            if step <= 0:
                continue
            prefill = float(record.get("prefill_time", record.get("request_prefill_time", 0.0)))
            decode = float(record.get("decode_time", record.get("request_decode_time", 0.0)))
            per_step[step] = (prefill, decode)
    return per_step


def collect_step_averages(base_dir: Path) -> Dict[int, Dict[str, float]]:
    totals: Dict[int, Dict[str, float]] = defaultdict(lambda: {"prefill_sum": 0.0, "decode_sum": 0.0, "count": 0})
    instance_count = 0

    for instance_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        prefix_file = instance_dir / "prefix_cache_metrics.jsonl"
        if not prefix_file.exists():
            continue
        try:
            per_step = load_instance_prefix_metrics(prefix_file)
        except json.JSONDecodeError as exc:  # noqa: PERF203
            raise SystemExit(f"Failed to parse {prefix_file}: {exc}") from exc
        if not per_step:
            continue
        instance_count += 1
        for step, (prefill, decode) in per_step.items():
            totals[step]["prefill_sum"] += prefill
            totals[step]["decode_sum"] += decode
            totals[step]["count"] += 1

    if instance_count == 0:
        raise SystemExit(f"No instances with prefix_cache_metrics.jsonl found under {base_dir}")

    averages: Dict[int, Dict[str, float]] = {}
    for step, values in totals.items():
        count = int(values["count"])
        if count <= 0:
            continue
        averages[step] = {
            "prefill_avg": values["prefill_sum"] / count,
            "decode_avg": values["decode_sum"] / count,
            "count": count,
        }
    if not averages:
        raise SystemExit(f"No per-step prefix metrics found under {base_dir}")
    return dict(sorted(averages.items()))


def plot_step_breakdown(
    steps: List[int],
    prefill: List[float],
    decode: List[float],
    output_path: Path,
    *,
    title: str,
    dpi: int = 320,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 0.6), 6))
    ax.bar(steps, prefill, color=PREFILL_COLOR, label="Prefill", align="edge", width=0.8)
    ax.bar(steps, decode, bottom=prefill, color=DECODE_COLOR, label="Decode", align="edge", width=0.8)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Average Time (s)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlim(min(steps) - 0.1, max(steps) + 0.9)
    ax.set_xticks(steps)
    if len(steps) > 30:
        ax.set_xticks(steps[::max(1, len(steps) // 30)])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved step breakdown chart to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing instance subdirectories with prefix_cache_metrics.jsonl",
    )
    parser.add_argument("output", type=Path, help="Path to save the figure")
    parser.add_argument("--title", type=str, default="Average Prefill vs Decode per Step", help="Figure title")
    parser.add_argument("--dpi", type=int, default=320, help="Output DPI (default: %(default)s)")
    parser.add_argument("--max-steps", type=int, default=50, help="Limit to first N steps (default: 50)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"Run directory not found: {args.run_dir}")

    data = collect_step_averages(args.run_dir)
    if not data:
        raise SystemExit(f"No step data found in {args.run_dir}")

    steps = list(data.keys())
    if args.max_steps is not None:
        steps = steps[: args.max_steps]
    prefill: List[float] = []
    decode: List[float] = []
    for step in steps:
        record = data[step]
        prefill.append(float(record.get("prefill_avg", 0.0)))
        decode.append(float(record.get("decode_avg", 0.0)))

    plot_step_breakdown(
        steps,
        prefill,
        decode,
        args.output,
        title=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
