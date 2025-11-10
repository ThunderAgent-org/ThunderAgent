#!/usr/bin/env python3
"""Extract the last step input token count from each trace log and compute the average."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import re

INPUT_PATTERN = re.compile(r"input_tokens=([\d,]+)")

def extract_last_input_tokens(trace_path: Path) -> int | None:
    """Return the last input_tokens value in the given trace log, if any."""
    last_value: int | None = None
    try:
        with trace_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = INPUT_PATTERN.search(line)
                if match:
                    last_value = int(match.group(1).replace(",", ""))
    except OSError as exc:
        raise RuntimeError(f"Failed to read {trace_path}: {exc}") from exc
    return last_value

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="test_data/ziyang/vllm_local__--data--models--GLM-4.5-FP8__t-0.00__p-1.00__c-0.00___swe_bench_lite_test__bs4",
        type=Path,
        help="Directory containing instance subdirectories with .trace.log files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output and only print the average.",
    )
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    token_values: list[int] = []
    trace_paths = sorted(root.rglob("*.trace.log"))
    if not trace_paths:
        raise SystemExit(f"No .trace.log files found under {root}")

    for trace_path in trace_paths:
        last_tokens = extract_last_input_tokens(trace_path)
        if last_tokens is None:
            continue
        token_values.append(last_tokens)
        if not args.quiet:
            print(f"{trace_path.relative_to(root)}: {last_tokens}")

    if not token_values:
        raise SystemExit("Found trace logs but no input_tokens entries.")

    average = statistics.fmean(token_values)
    print(f"Processed {len(token_values)} trace logs.")
    print(f"Average last-step input_tokens: {average:.2f}")

if __name__ == "__main__":
    main()
