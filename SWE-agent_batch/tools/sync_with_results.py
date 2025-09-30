#!/usr/bin/env python3
"""Sync profiling outputs with evaluation results, removing error instances.

Usage:

    python tools/sync_with_results.py \
        --batch-dir trajectories/.../swe_bench_lite_test \
        --results-json <run_id>.lite_test_success_eval.json

The script performs the following actions:
    1. Load `results_json` to obtain `resolved_ids`, `unresolved_ids`, and `error_ids`.
    2. Delete instance directories for all IDs listed in `error_ids`.
    3. Update each `profiling.json` so that `status` reflects evaluation outcome
       (`success` for resolved, `failure` otherwise).
    4. Rebuild `profiling_summary.json` with the updated statuses, preserving
       the existing `configuration` section if present.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

from sweagent.utils.profiling import ProfilingAggregator


def _collect_ids(values: Iterable[str] | dict[str, object] | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, dict):
        return set(values.keys())
    return {str(v) for v in values}


def sync_profiling(batch_dir: Path, results_json: Path) -> None:
    results = json.loads(results_json.read_text())
    resolved = _collect_ids(results.get("resolved_ids"))
    unresolved = _collect_ids(results.get("unresolved_ids"))
    error_ids = _collect_ids(results.get("error_ids"))

    aggregator = ProfilingAggregator()

    removed = []
    for inst_id in error_ids:
        inst_dir = batch_dir / inst_id
        if inst_dir.exists():
            shutil.rmtree(inst_dir, ignore_errors=True)
            removed.append(inst_id)

    print(f"Removed {len(removed)} error-instance directories.")
    if removed:
        print("Removed IDs:", ", ".join(sorted(removed)))

    for profiling_path in batch_dir.glob("*/profiling.json"):
        data = json.loads(profiling_path.read_text())
        inst_id = data.get("problem_id") or profiling_path.parent.name
        if inst_id in resolved:
            status = "success"
        else:
            status = "failure"
        data["status"] = status
        metadata = data.setdefault("metadata", {})
        metadata["evaluation_status"] = status
        metadata["evaluation_source"] = str(results_json)
        profiling_path.write_text(json.dumps(data, indent=2))
        aggregator.add(data)

    summary = aggregator.summary()
    if summary is None:
        raise RuntimeError("No profiling data after syncing.")

    summary["evaluation"] = {
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "error_ids": sorted(error_ids),
        "results_json": str(results_json),
    }

    summary_path = batch_dir / "profiling_summary.json"
    if summary_path.exists():
        try:
            old_summary = json.loads(summary_path.read_text())
            if "configuration" in old_summary:
                summary["configuration"] = old_summary["configuration"]
        except Exception:
            pass

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Updated profiling_summary.json (success rate {summary['success_rate']['percentage']*100:.2f}%).")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sync profiling outputs with evaluation results.")
    parser.add_argument("--batch-dir", required=True, type=Path, help="Path to run-batch output directory")
    parser.add_argument("--results-json", required=True, type=Path, help="Path to evaluation results JSON")
    args = parser.parse_args()

    sync_profiling(args.batch_dir.resolve(), args.results_json.resolve())


if __name__ == "__main__":
    main()
