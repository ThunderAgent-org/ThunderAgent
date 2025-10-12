#!/usr/bin/env python3
"""Plot GPU SM utilization over time from collected gpu_metrics.json files."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore[import-not-found]


def load_gpu_metrics(root: Path, stride: int = 1) -> Dict[int, List[Tuple[float, float]]]:
    series: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for metrics_path in sorted(root.rglob("gpu_metrics.json")):
        try:
            data = json.loads(metrics_path.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            timestamp = entry.get("timestamp")
            gpus = entry.get("gpus")
            if timestamp is None or not isinstance(gpus, list):
                continue
            if stride > 1 and idx % stride != 0:
                continue
            try:
                ts = float(timestamp)
            except (TypeError, ValueError):
                continue
            for gpu in gpus:
                if not isinstance(gpu, dict):
                    continue
                idx = gpu.get("index")
                sm_util = gpu.get("sm_util")
                if idx is None or sm_util is None:
                    continue
                try:
                    idx_int = int(idx)
                    util = float(sm_util)
                except (TypeError, ValueError):
                    continue
                series[idx_int].append((ts, util))
    return series


def plot_gpu_utilization(
    series: Dict[int, List[Tuple[float, float]]],
    output: Path,
    *,
    dpi: int = 200,
) -> None:
    if not series:
        raise SystemExit("No GPU metrics found.")

    min_ts = min(ts for values in series.values() for ts, _ in values)

    plt.figure(figsize=(14, 6))
    color_cycle = plt.get_cmap("tab10")

    for idx, values in sorted(series.items()):
        values.sort(key=lambda item: item[0])
        xs = [(ts - min_ts) for ts, _ in values]
        ys = [util for _, util in values]
        plt.step(xs, ys, where="post", linewidth=1.0, alpha=0.9, label=f"GPU {idx}", color=color_cycle(idx % 10))

    plt.xlabel("Time since first measurement (s)")
    plt.ylabel("SM Utilization (%)")
    plt.title("GPU Utilization Over Time")
    plt.legend(ncol=2, frameon=False)
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()
    print(f"Saved GPU utilization plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("trajectories/ziyang/vllm_local__--data--models--GLM-4.5-FP8__t-0.00__p-1.00__c-0.00___swe_bench_lite_test__bs4"),
        help="Root directory containing gpu_metrics.json files (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gpu_utilization.png"),
        help="Path to save the output plot (default: %(default)s).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for the saved figure (default: %(default)s).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every N-th measurement to reduce density (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root.exists():
        raise SystemExit(f"Root directory not found: {args.root}")
    series = load_gpu_metrics(args.root, stride=max(1, args.stride))
    plot_gpu_utilization(series, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
