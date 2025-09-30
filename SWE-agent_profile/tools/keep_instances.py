#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional


def load_status(profiling_path: Path) -> Optional[str]:
    try:
        data = json.loads(profiling_path.read_text())
    except Exception:
        return None
    return data.get("status")


def main(batch_dir: Path) -> None:
    kept, removed = [], []

    for profiling in batch_dir.glob("*/profiling.json"):
        status = load_status(profiling)
        instance_dir = profiling.parent
        if status == "success":
            kept.append(instance_dir.name)
            continue
        shutil.rmtree(instance_dir, ignore_errors=True)
        removed.append(instance_dir.name)

    print(f"Kept {len(kept)} success instances.")
    print(f"Removed {len(removed)} non-success instances.")
    if removed:
        print("Removed:", ", ".join(sorted(removed)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Keep only success instances (status == 'success').")
    parser.add_argument(
        "--batch-dir",
        required=True,
        type=Path,
        help="Path to run-batch output directory (e.g., trajectories/.../swe_bench_lite_test)",
    )
    args = parser.parse_args()
    main(args.batch_dir.resolve())
