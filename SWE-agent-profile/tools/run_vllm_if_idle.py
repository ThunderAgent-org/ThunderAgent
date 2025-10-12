#!/usr/bin/env python3
"""Launch run_vllm_batch.py for selected runs when all GPUs are idle."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

DEFAULT_RUNS = ("bs64")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check GPU utilisation and trigger run_vllm_batch.py if GPUs are idle.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("config/vllm_batch.yaml"),
        help="Configuration file passed to tools/run_vllm_batch.py (default: %(default)s).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=list(DEFAULT_RUNS),
        help="Run labels to execute when GPUs are idle (default:bs64).",
    )
    parser.add_argument(
        "--memory-threshold",
        type=float,
        default=50.0,
        help="Maximum memory usage (MiB) per GPU to be considered idle (default: %(default)s).",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Seconds to wait between idle checks until GPUs become free (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed GPU utilisation information.",
    )
    return parser.parse_args()


def query_gpu_memory() -> list[float]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,nounits,noheader",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to query GPU utilisation with {' '.join(cmd)} (exit {completed.returncode}):\n"
            f"{completed.stderr.strip()}"
        )
    values = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def gpus_idle(threshold: float, verbose: bool) -> bool:
    mem_used = query_gpu_memory()
    if verbose:
        if mem_used:
            print("GPU memory used (MiB):", ", ".join(f"{v:.1f}" for v in mem_used))
        else:
            print("No GPUs detected or unable to parse nvidia-smi output.")
    if not mem_used:
        return False
    return all(value <= threshold for value in mem_used)


def run_batches(config_file: Path, runs: Sequence[str]) -> int:
    cmd = [
        sys.executable,
        "tools/run_vllm_batch.py",
        "--config-file",
        str(config_file),
        "--only-runs",
        *runs,
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd)
    return completed.returncode


def main() -> int:
    args = parse_args()

    if not args.config_file.exists():
        print(f"Config file {args.config_file} does not exist.", file=sys.stderr)
        return 1

    if not args.runs:
        print("No runs specified; nothing to do.")
        return 0

    try:
        while True:
            if gpus_idle(args.memory_threshold, args.verbose):
                break
            print(
                f"GPUs are not idle (threshold {args.memory_threshold} MiB). Checking again in {args.check_interval}s..."
            )
            time.sleep(max(1, args.check_interval))
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 1

    exit_code = run_batches(args.config_file, args.runs)
    if exit_code != 0:
        print(f"run_vllm_batch.py exited with code {exit_code}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
