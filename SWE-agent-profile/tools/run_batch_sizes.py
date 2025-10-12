#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_BATCH_SIZES = (4, 8, 16, 32)


def build_command(batch_size: int, config: str | None, extra_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sweagent.run.run",
        "run-batch",
        "--num_workers",
        str(batch_size),
        "--instances.type",
        "swe_bench",
        "--instances.subset",
        "lite",
        "--instances.split",
        "test",
        "--instances.slice",
        ":64",
        "--suffix",
        f"bs{batch_size}",
    ]
    if config:
        cmd.extend(["--config", config])
    cmd.extend(extra_args)
    return cmd


def run_batches(batch_sizes: list[int], config: str | None, extra_args: list[str], dry_run: bool) -> None:
    for batch_size in batch_sizes:
        cmd = build_command(batch_size, config, extra_args)
        joined_cmd = " ".join(cmd)
        print(f"\n[Batch size {batch_size}] Running: {joined_cmd}")
        if dry_run:
            continue
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run sweagent run-batch with multiple batch sizes over the first 64 SWE-bench lite test instances.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional sweagent config file to use for every run. If omitted, the CLI defaults apply "
            "(identical to invoking `sweagent run-batch` directly)."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=list(DEFAULT_BATCH_SIZES),
        help="Override the default batch sizes (4, 8, 16, 32).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    args, extra = parser.parse_known_args()
    return args, extra


def main() -> None:
    args, extra = parse_args()
    batch_sizes = sorted(set(args.batch_sizes))
    if not batch_sizes:
        raise SystemExit("At least one batch size must be specified")
    run_batches(batch_sizes=batch_sizes, config=args.config, extra_args=extra, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
