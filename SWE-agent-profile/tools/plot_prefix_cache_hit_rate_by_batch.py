#!/usr/bin/env python3
"""Plot average prefix cache hit rate per step for multiple batch sizes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt

PREFIX_CACHE_FILENAME = "prefix_cache_metrics.jsonl"
DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64]
FILL_STRATEGIES = ["carry-forward", "zero", "nan"]


@dataclass
class StepAccumulator:
    """Container for aggregating prefix cache metrics per step."""

    hits: float = 0.0
    queries: float = 0.0
    records: int = 0
    zero_query_records: int = 0
    fallback_rate_sum: float = 0.0
    fallback_rate_count: int = 0
    observed: bool = False
    instances: set[str] = field(default_factory=set)
    zero_query_instances: set[str] = field(default_factory=set)

    def add(
        self,
        hits: float,
        queries: float,
        fallback_rate: float | None,
        *,
        instance_id: str,
    ) -> None:
        self.records += 1
        self.instances.add(instance_id)
        if queries > 0:
            self.hits += hits
            self.queries += queries
            self.observed = True
            self.zero_query_instances.discard(instance_id)
        else:
            self.zero_query_records += 1
            if fallback_rate is not None:
                self.fallback_rate_sum += fallback_rate
                self.fallback_rate_count += 1
                self.observed = True
            self.zero_query_instances.add(instance_id)

    def hit_rate(self) -> tuple[float | None, bool]:
        if self.queries > 0:
            return self.hits / self.queries, True
        if self.fallback_rate_count > 0:
            return self.fallback_rate_sum / self.fallback_rate_count, True
        return None, self.observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate prefix cache metrics and plot the average hit rate per step "
            "for multiple batch sizes."
        ),
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing subdirectories for each batch size (e.g. bs4, bs8, ...). "
            "The script will search recursively for prefix_cache_metrics.jsonl files."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prefix_cache_hit_rate_by_batch.png"),
        help="Path to save the generated plot (default: %(default)s).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help=(
            "Optional path to save the aggregated data as CSV with columns: "
            "batch_size, step, hit_rate_percent, total_hits_delta, total_queries_delta, "
            "record_count, zero_query_records."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to include (default: %(default)s).",
    )
    parser.add_argument(
        "--fill-missing",
        choices=FILL_STRATEGIES,
        default="carry-forward",
        help=(
            "Strategy for steps without prefix cache queries: "
            "carry-forward uses the previous rate, zero forces 0%%, nan leaves gaps."
        ),
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower bound for the x-axis (step).",
    )
    parser.add_argument(
        "--report-zero",
        action="store_true",
        help="Print steps and sample instance IDs where the aggregated queries delta remained zero.",
    )
    parser.add_argument(
        "--report-zero-limit",
        type=int,
        default=5,
        help="Maximum number of zero-query steps to list per batch when reporting (default: %(default)s).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional upper bound for the x-axis (step).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=None,
        help="Optional lower bound for the y-axis (hit rate %%).",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper bound for the y-axis (hit rate %%).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing information.",
    )
    return parser.parse_args()


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _iter_metrics_files(batch_dirs: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for batch_dir in batch_dirs:
        for metrics_path in sorted(batch_dir.rglob(PREFIX_CACHE_FILENAME)):
            resolved = metrics_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield metrics_path


def _load_batch_directories(root_dir: Path, batch_size: int) -> list[Path]:
    direct = root_dir / f"bs{batch_size}"
    if direct.is_dir():
        return [direct]
    matches = [p for p in root_dir.glob(f"**/*bs{batch_size}*") if p.is_dir()]
    return matches


def _aggregate_metrics_from_file(
    metrics_file: Path,
    step_totals: defaultdict[int, StepAccumulator],
    *,
    instance_id: str,
    verbose: bool = False,
) -> int:
    processed = 0
    try:
        with metrics_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if verbose:
                        print(f"Skipping invalid JSON in {metrics_file}:{line_number}")
                    continue

                step_raw = record.get("step")
                try:
                    step_value = int(step_raw)
                except (TypeError, ValueError):
                    step_value = None
                if step_value is None:
                    if verbose:
                        print(f"Missing step in {metrics_file}:{line_number}")
                    continue

                hits_delta = _safe_float(record.get("prefix_cache_hits_delta"))
                queries_delta = _safe_float(record.get("prefix_cache_queries_delta"))
                fallback_rate = _safe_float(record.get("hit_rate_delta"))

                if queries_delta is None and fallback_rate is not None:
                    queries_delta = 0.0
                if queries_delta is None:
                    queries_delta = 0.0
                if hits_delta is None:
                    if fallback_rate is not None and queries_delta > 0:
                        hits_delta = fallback_rate * queries_delta
                    else:
                        hits_delta = 0.0

                hits_delta = max(hits_delta, 0.0)
                queries_delta = max(queries_delta, 0.0)

                accumulator = step_totals[step_value]
                accumulator.add(
                    hits_delta,
                    queries_delta,
                    fallback_rate,
                    instance_id=instance_id,
                )
                processed += 1
    except FileNotFoundError:
        if verbose:
            print(f"Metrics file not found: {metrics_file}")
    return processed


def aggregate_batch_metrics(
    batch_dirs: Iterable[Path],
    *,
    verbose: bool = False,
) -> tuple[dict[int, StepAccumulator], set[Path], int, int]:
    step_totals: defaultdict[int, StepAccumulator] = defaultdict(StepAccumulator)
    instance_dirs: set[Path] = set()
    record_count = 0
    file_count = 0

    for metrics_file in _iter_metrics_files(batch_dirs):
        file_count += 1
        instance_dirs.add(metrics_file.parent)
        record_count += _aggregate_metrics_from_file(
            metrics_file,
            step_totals,
            instance_id=metrics_file.parent.name,
            verbose=verbose,
        )

    return dict(step_totals), instance_dirs, record_count, file_count


def _apply_fill_strategy(values: list[float | None], strategy: str) -> list[float]:
    if strategy not in FILL_STRATEGIES:
        raise ValueError(f"Unsupported fill strategy: {strategy}")

    filled: list[float] = []
    last_valid: float | None = None

    for value in values:
        value_is_valid = value is not None and not math.isnan(value)

        if not value_is_valid:
            if strategy == "carry-forward":
                filled.append(last_valid if last_valid is not None else float("nan"))
            elif strategy == "zero":
                filled.append(0.0)
            elif strategy == "nan":
                filled.append(float("nan"))
        else:
            filled.append(value)
            last_valid = value

    if last_valid is None and strategy == "carry-forward":
        # No valid observations at all; fall back to zeros so the series is still visible.
        filled = [0.0 for _ in values]

    return filled


def build_series(
    accumulators: dict[int, StepAccumulator], *, fill_strategy: str
) -> dict[str, list]:
    steps = sorted(accumulators.keys())
    raw_hit_rates: list[float | None] = []
    hits_totals = []
    queries_totals = []
    record_counts = []
    zero_query_counts = []
    missing_counts = []

    for step in steps:
        stats = accumulators[step]
        rate_value, observed = stats.hit_rate()
        raw_hit_rates.append(None if rate_value is None else rate_value * 100)
        hits_totals.append(stats.hits)
        queries_totals.append(stats.queries)
        record_counts.append(stats.records)
        zero_query_counts.append(len(stats.zero_query_instances))
        missing_counts.append(0 if observed else 1)

    hit_rates = _apply_fill_strategy(raw_hit_rates, fill_strategy)

    return {
        "steps": steps,
        "hit_rates": hit_rates,
        "hits": hits_totals,
        "queries": queries_totals,
        "records": record_counts,
        "zero_queries": zero_query_counts,
        "missing": missing_counts,
    }


def _report_zero_query_steps(
    batch_size: int,
    accumulators: dict[int, StepAccumulator],
    *,
    limit: int,
) -> None:
    zero_entries: list[tuple[int, StepAccumulator]] = [
        (step, stats)
        for step, stats in sorted(accumulators.items())
        if stats.zero_query_instances
    ]

    if not zero_entries:
        print(f"Batch size {batch_size}: no zero-query steps detected.")
        return

    total_zero_steps = len(zero_entries)
    print(
        f"Batch size {batch_size}: {total_zero_steps} steps had zero aggregate prefix-cache queries."
    )

    max_steps_to_show = max(0, limit)
    if max_steps_to_show == 0:
        return

    for step, stats in zero_entries[:max_steps_to_show]:
        instance_names = sorted(stats.zero_query_instances)
        sample_limit = 5
        sample = ", ".join(instance_names[:sample_limit])
        extra = len(instance_names) - min(len(instance_names), sample_limit)
        if extra > 0:
            sample += f", +{extra} more"
        print(f"  step {step}: {len(instance_names)} instances (e.g. {sample})")

    if total_zero_steps > max_steps_to_show:
        remaining = total_zero_steps - max_steps_to_show
        print(f"  ... {remaining} additional steps omitted (increase --report-zero-limit to see more).")


def _compute_axis_limits(
    x_range: Tuple[float | None, float | None],
    y_range: Tuple[float | None, float | None],
) -> Tuple[Tuple[float | None, float | None], Tuple[float | None, float | None]]:
    x_min, x_max = x_range
    y_min, y_max = y_range
    if x_min is not None and x_max is not None and x_min > x_max:
        raise ValueError("x-min cannot be greater than x-max")
    if y_min is not None and y_max is not None and y_min > y_max:
        raise ValueError("y-min cannot be greater than y-max")
    return (x_min, x_max), (y_min, y_max)


def plot_batch_series(
    batch_series: dict[int, dict[str, list]],
    output: Path,
    *,
    x_limits: Tuple[float | None, float | None] = (None, None),
    y_limits: Tuple[float | None, float | None] = (None, None),
) -> None:
    if not batch_series:
        raise RuntimeError("No data to plot.")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    for index, (batch_size, series) in enumerate(sorted(batch_series.items())):
        steps = series["steps"]
        hit_rates = series["hit_rates"]
        if not steps:
            continue
        if all(math.isnan(rate) for rate in hit_rates):
            continue
        color = cmap(index % cmap.N)
        ax.plot(
            steps,
            hit_rates,
            label=f"bs={batch_size}",
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Average Prefix Cache Hit Rate (Δ %)", fontsize=12)
    ax.set_title("Average Prefix Cache Hit Rate by Batch Size", fontsize=14, pad=12)
    ax.grid(True, alpha=0.3)

    (x_min, x_max), (y_min, y_max) = _compute_axis_limits(x_limits, y_limits)
    if x_min is not None or x_max is not None:
        ax.set_xlim(x_min, x_max)
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    ax.legend(title="Batch Size", fontsize=9)

    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved plot to {output}")


def write_csv(batch_series: dict[int, dict[str, list]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "batch_size",
                "step",
                "hit_rate_percent",
                "total_hits_delta",
                "total_queries_delta",
                "record_count",
                "zero_query_records",
            ]
        )
        for batch_size, series in sorted(batch_series.items()):
            for idx, step in enumerate(series["steps"]):
                writer.writerow(
                    [
                        batch_size,
                        step,
                        f"{series['hit_rates'][idx]:.6f}",
                        f"{series['hits'][idx]:.6f}",
                        f"{series['queries'][idx]:.6f}",
                        series["records"][idx],
                        series["zero_queries"][idx],
                    ]
                )
    print(f"Saved aggregated data to {csv_path}")


def main() -> None:
    args = parse_args()

    if not args.root_dir.is_dir():
        raise SystemExit(f"Root directory {args.root_dir} does not exist or is not a directory.")

    batch_series: dict[int, dict[str, list]] = {}
    batch_accumulators: dict[int, dict[int, StepAccumulator]] = {}
    for batch_size in args.batch_sizes:
        batch_dirs = _load_batch_directories(args.root_dir, batch_size)
        if not batch_dirs:
            if args.verbose:
                print(f"No directories found for batch size {batch_size} under {args.root_dir}")
            continue

        accumulators, instances, record_count, file_count = aggregate_batch_metrics(
            batch_dirs,
            verbose=args.verbose,
        )

        if not accumulators:
            if args.verbose:
                print(f"No metrics found for batch size {batch_size} (checked {file_count} files)")
            continue

        if args.verbose:
            print(
                f"Batch size {batch_size}: {len(instances)} instance dirs, "
                f"{file_count} metric files, {record_count} records"
            )

        series = build_series(accumulators, fill_strategy=args.fill_missing)
        batch_series[batch_size] = series
        batch_accumulators[batch_size] = accumulators

        if args.verbose:
            hit_rates = series["hit_rates"]
            valid_rates = [r for r in hit_rates if not math.isnan(r)]
            if valid_rates:
                min_rate = min(valid_rates)
                max_rate = max(valid_rates)
                print(
                    f"  Steps: {len(series['steps'])}, valid rates: {len(valid_rates)}, "
                    f"min={min_rate:.2f}%, max={max_rate:.2f}%"
                )
            else:
                print(
                    "  No valid hit rate observations (all steps lacked prefix cache queries)."
                )

    if not batch_series:
        raise SystemExit("No prefix cache metrics found for the requested batch sizes.")

    plot_batch_series(
        batch_series,
        args.output,
        x_limits=(args.x_min, args.x_max),
        y_limits=(args.y_min, args.y_max),
    )

    if args.report_zero:
        for batch_size in sorted(batch_series.keys()):
            _report_zero_query_steps(
                batch_size,
                batch_accumulators[batch_size],
                limit=args.report_zero_limit,
            )

    if args.csv_output:
        write_csv(batch_series, args.csv_output)


if __name__ == "__main__":
    main()

