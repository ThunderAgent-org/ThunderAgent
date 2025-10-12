#!/usr/bin/env python3
"""Collect per-step input token counts from SWE-agent trace logs.

The script expects a directory that contains one subdirectory per batch size
(e.g. ``...__bs4``). Each of those directories contains many instance folders
with ``.trace.log`` files. For every trace log, we extract every occurrence of
``input_tokens=`` and emit the sequence for that instance.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

INPUT_PATTERN = re.compile(r"input_tokens=([\d,]+)")


def parse_trace(trace_path: Path) -> list[int]:
    """Return all input_tokens values (ordered) from a trace log."""
    tokens: list[int] = []
    with trace_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = INPUT_PATTERN.search(line)
            if match:
                value = int(match.group(1).replace(",", ""))
                tokens.append(value)
    return tokens


def to_step_deltas(tokens: list[int]) -> list[int]:
    """Convert cumulative token counts to per-step deltas."""
    if not tokens:
        return tokens
    deltas = [tokens[0]]
    for prev, current in zip(tokens, tokens[1:]):
        deltas.append(current - prev)
    return deltas


def _matches_batch(name: str, batches: set[int] | None) -> bool:
    if batches is None:
        return True
    return any(f"bs{size}" in name for size in batches)


def _ensure_length(buffer: list[float], length: int) -> None:
    if len(buffer) < length:
        buffer.extend([0.0] * (length - len(buffer)))


def collect(root: Path, batch_sizes: set[int] | None = None) -> dict[str, dict[str, list[float]]]:
    """Walk batch-size directories under ``root`` and gather aggregated token sequences."""
    per_batch_instances: dict[str, dict[str, list[int]]] = {}
    for batch_dir in sorted(p for p in root.iterdir() if p.is_dir() and _matches_batch(p.name, batch_sizes)):
        batch_key = batch_dir.name
        batch_entries: dict[str, list[int]] = {}
        trace_paths = sorted(batch_dir.rglob("*.trace.log"))
        for trace_path in trace_paths:
            relative_name = str(trace_path.relative_to(batch_dir))
            tokens = parse_trace(trace_path)
            batch_entries[relative_name] = to_step_deltas(tokens)
        per_batch_instances[batch_key] = batch_entries

    aggregated: dict[str, dict[str, list[float]]] = {}
    for batch_key, entries in per_batch_instances.items():
        totals: list[float] = []
        counts: list[float] = []
        for seq in entries.values():
            _ensure_length(totals, len(seq))
            _ensure_length(counts, len(seq))
            for idx, value in enumerate(seq):
                totals[idx] += value
                counts[idx] += 1.0
        averages = [
            (totals[idx] / counts[idx]) if counts[idx] else 0.0 for idx in range(len(totals))
        ]
        aggregated[batch_key] = {
            "sum": totals,
            "count": counts,
            "avg": averages,
        }

    return aggregated


def compute_multistep(per_batch: dict[str, dict[str, list[float]]]) -> list[float]:
    if not per_batch:
        return []
    max_steps = max(len(data["sum"]) for data in per_batch.values())
    if max_steps == 0:
        return []
    multistep: list[float] = []
    for idx in range(max_steps):
        total_sum = 0.0
        total_count = 0.0
        for data in per_batch.values():
            sums = data["sum"]
            counts = data["count"]
            if idx < len(sums):
                total_sum += sums[idx]
                total_count += counts[idx] if idx < len(counts) else 0.0
        multistep.append(total_sum / total_count if total_count else 0.0)
    return multistep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="trajectories/ziyang",
        type=Path,
        help="Directory containing batch size subdirectories.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("input_tokens_by_step.json"),
        help="Path of the JSON file to write.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output (default: 2).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Only include batch directories whose name contains 'bs{size}' for any of the provided sizes.",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    batch_filter = set(args.batch_sizes) if args.batch_sizes else None
    data = collect(root, batch_filter)
    multistep = compute_multistep(data)
    output_payload = {
        "batch_sizes": data,
        "multistep": multistep,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=args.indent)
        handle.write("\n")

    print(
        f"Wrote {args.output} with per-batch token stats (sum/count/avg) "
        f"for {len(data)} batch sizes plus multistep averages."
    )


if __name__ == "__main__":
    main()
