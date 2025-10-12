#!/usr/bin/env python3
"""Compute average LLM throughput per batch size from SWE-agent trajectories.

For each batch-size directory (e.g. ``...__bs4``) under the given root, the script
assumes there are 64 instance subdirectories. Instances are sorted lexicographically
and partitioned into consecutive groups of size equal to the batch size. Each group
corresponds to the set of instances that ran concurrently.

For every group and every step, we measure the duration between the earliest
``llm_reasoning`` enter timestamp and the latest exit timestamp recorded in
``stage_timings.jsonl``. We pair that with the per-step total token counts (input
+ output) loaded from ``trajectories_total_tokens.json``. Summing tokens and time
across all steps and groups yields the effective throughput for the batch size.

The script outputs a JSON file mapping each batch size to its average throughput
(tokens per second).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import math

STAGE_FILE = "stage_timings.jsonl"
TOTAL_TOKENS_JSON = "trajectories_total_tokens.json"

def load_total_tokens(path: Path) -> Dict[str, Dict[str, List[int]]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    converted: Dict[str, Dict[str, List[int]]] = {}
    for batch_key, instances in raw.items():
        inst_tokens: Dict[str, List[int]] = {}
        for rel_path, seq in instances.items():
            instance_name = rel_path.split("/", 1)[0]
            inst_tokens[instance_name] = seq
        converted[batch_key] = inst_tokens
    return converted


def parse_stage_timings(stage_path: Path) -> Dict[int, Tuple[float, float]]:
    """Return per-step (enter, exit) timestamps for llm_reasoning."""
    import json as _json

    step_times: Dict[int, Dict[str, List[float]]] = {}
    with stage_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if record.get("stage") != "llm_reasoning":
                continue
            step = int(record.get("step", 0))
            phase = record.get("phase")
            if phase not in {"enter", "exit"}:
                continue
            bucket = step_times.setdefault(step, {"enter": [], "exit": []})
            ts = float(record.get("timestamp"))
            bucket[phase].append(ts)
    result: Dict[int, Tuple[float, float]] = {}
    for step, phases in step_times.items():
        enters = phases.get("enter") or []
        exits = phases.get("exit") or []
        if enters and exits:
            result[step] = (min(enters), max(exits))
    return result


def chunk_instances(instances: Iterable[str], chunk_size: int) -> List[List[str]]:
    inst_list = list(instances)
    return [inst_list[i : i + chunk_size] for i in range(0, len(inst_list), chunk_size)]


def compute_group_totals(
    instances: List[str],
    tokens_map: Dict[str, List[int]],
    timings_map: Dict[str, Dict[int, Tuple[float, float]]],
) -> Tuple[float, float]:
    """Return (total_tokens, total_time) aggregated across steps for a group."""
    total_tokens = 0.0
    total_time = 0.0

    max_step = 0
    for name in instances:
        timing = timings_map.get(name)
        if not timing:
            continue
        max_step = max(max_step, max(timing))

    for step in range(1, max_step + 1):
        step_tokens = 0.0
        enters: List[float] = []
        exits: List[float] = []
        for name in instances:
            token_seq = tokens_map.get(name)
            timing = timings_map.get(name)
            if not token_seq or not timing or step not in timing:
                continue
            idx = step - 1
            if idx < len(token_seq):
                step_tokens += token_seq[idx]
            enter_ts, exit_ts = timing[step]
            enters.append(enter_ts)
            exits.append(exit_ts)
        if not enters or not exits:
            continue
        step_time = max(exits) - min(enters)
        if step_time <= 0:
            continue
        total_tokens += step_tokens
        total_time += step_time
    return total_tokens, total_time


def compute_throughput_for_batch(
    batch_dir: Path,
    batch_size: int,
    tokens_map: Dict[str, List[int]],
) -> float:
    timings_map: Dict[str, Dict[int, Tuple[float, float]]] = {}
    instances = sorted(
        [p.name for p in batch_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )

    for name in instances:
        stage_path = batch_dir / name / STAGE_FILE
        if not stage_path.exists():
            continue
        timings_map[name] = parse_stage_timings(stage_path)

    groups = chunk_instances(instances, batch_size)

    total_tokens = 0.0
    total_time = 0.0
    for group in groups:
        group_tokens, group_time = compute_group_totals(group, tokens_map, timings_map)
        total_tokens += group_tokens
        total_time += group_time

    if total_time <= 0:
        return 0.0
    return total_tokens / total_time


def extract_batch_size(directory_name: str) -> int:
    if "bs" not in directory_name:
        raise ValueError(f"Cannot infer batch size from directory name: {directory_name}")
    suffix = directory_name.rsplit("bs", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"Batch size suffix is not numeric in {directory_name}")
    return int(suffix)


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
        "--total-tokens-json",
        type=Path,
        default=Path(TOTAL_TOKENS_JSON),
        help="JSON file produced by collect_output_and_total_tokens.py",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("llm_throughput_by_batch.json"),
        help="Output JSON file for throughput results.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for JSON output (default: 2).",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    if not args.total_tokens_json.exists():
        raise SystemExit(f"Total tokens JSON not found: {args.total_tokens_json}")

    total_tokens_data = load_total_tokens(args.total_tokens_json)

    results: Dict[str, float] = {}

    for batch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        batch_name = batch_dir.name
        try:
            batch_size = extract_batch_size(batch_name)
        except ValueError:
            continue
        tokens_map = total_tokens_data.get(batch_name)
        if not tokens_map:
            continue
        throughput = compute_throughput_for_batch(batch_dir, batch_size, tokens_map)
        results[str(batch_size)] = throughput

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=args.indent)
        handle.write("\n")

    print(f"Wrote throughput results for {len(results)} batch sizes to {args.output}.")


if __name__ == "__main__":
    main()
