#!/usr/bin/env python3
"""Collect per-step decode token counts and aggregate across instances."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Dict, List

OUTPUT_PATTERN = re.compile(r"output_tokens=([\d,]+)")


def parse_tokens(trace_path: Path, pattern: re.Pattern[str]) -> List[int]:
    tokens: List[int] = []
    with trace_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                tokens.append(int(match.group(1).replace(",", "")))
    return tokens


def _matches_batch(name: str, batches: set[int] | None) -> bool:
    if batches is None:
        return True
    return any(f"bs{size}" in name for size in batches)


def _ensure_length(buffer: list[float], length: int) -> None:
    if len(buffer) < length:
        buffer.extend([0.0] * (length - len(buffer)))


def collect_decode_aggregates(root: Path, batch_sizes: set[int] | None = None) -> Dict[str, Dict[str, List[float]]]:
    aggregated: Dict[str, Dict[str, List[float]]] = {}
    for batch_dir in sorted(p for p in root.iterdir() if p.is_dir() and _matches_batch(p.name, batch_sizes)):
        sequences: List[List[int]] = []
        for trace_path in sorted(batch_dir.rglob("*.trace.log")):
            seq = parse_tokens(trace_path, OUTPUT_PATTERN)
            if seq:
                sequences.append(seq)
        if not sequences:
            aggregated[batch_dir.name] = {"sum": [], "count": [], "avg": []}
            continue
        totals: list[float] = []
        counts: list[float] = []
        for seq in sequences:
            _ensure_length(totals, len(seq))
            _ensure_length(counts, len(seq))
            for idx, value in enumerate(seq):
                totals[idx] += value
                counts[idx] += 1.0
        averages = [
            (totals[idx] / counts[idx]) if counts[idx] else 0.0 for idx in range(len(totals))
        ]
        aggregated[batch_dir.name] = {
            "sum": totals,
            "count": counts,
            "avg": averages,
        }
    return aggregated


def compute_multistep(per_batch: Dict[str, Dict[str, List[float]]]) -> List[float]:
    if not per_batch:
        return []
    max_steps = max(len(data["sum"]) for data in per_batch.values())
    if max_steps == 0:
        return []
    multistep: List[float] = []
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
        default=Path("trajectories/ziyang"),
        type=Path,
        help="Directory containing batch-size subdirectories.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("trajectories_output_tokens.json"),
        help="Where to write the aggregated per-step decode token counts.",
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
        help="Only include batch directories whose name contains 'bs{size}' for any provided size.",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    batch_filter = set(args.batch_sizes) if args.batch_sizes else None

    aggregated = collect_decode_aggregates(root, batch_filter)
    multistep = compute_multistep(aggregated)

    payload = {
        "batch_sizes": aggregated,
        "multistep": multistep,
    }

    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=args.indent)
        handle.write("\n")

    print(
        f"Wrote {args.output_json} with per-batch decode stats (sum/count/avg) "
        f"for {len(aggregated)} batch sizes plus multistep averages."
    )


if __name__ == "__main__":
    main()
