#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _parse_dir(name: str) -> Tuple[str, int]:
    m = re.search(r"hle_paper_(?P<method>[^_]+)_c(?P<c>\d+)_rep(?P<rep>\d+)_", name)
    if not m:
        return "", 0
    return m.group("method"), int(m.group("c"))


def _load_steps_per_sec(out_dir: Path) -> float | None:
    combined = out_dir / "combined_summary.json"
    if combined.exists():
        try:
            data = json.loads(combined.read_text(encoding="utf-8"))
            steps_block = data.get("steps") or {}
            # Newer format: steps["steps"]["steps_per_sec"]
            nested = (steps_block.get("steps") or {}).get("steps_per_sec")
            if isinstance(nested, (int, float)):
                return float(nested)
            # Legacy/flat format: steps["steps_per_sec"]
            flat = steps_block.get("steps_per_sec")
            if isinstance(flat, (int, float)):
                return float(flat)
        except Exception:
            pass
    return None


def _collect(outputs_dir: Path) -> Dict[str, Dict[int, float]]:
    rows: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for out_dir in outputs_dir.glob("hle_paper_*"):
        method, c = _parse_dir(out_dir.name)
        if not method or c <= 0:
            continue
        steps_per_sec = _load_steps_per_sec(out_dir)
        if steps_per_sec is None:
            continue
        # pick the newest run for each (method, c)
        mtime = out_dir.stat().st_mtime
        rows.setdefault(method, {})
        prev = rows[method].get(c)
        if prev is None or mtime > prev[0]:
            rows[method][c] = (mtime, steps_per_sec)
    out: Dict[str, Dict[int, float]] = {}
    for method, by_c in rows.items():
        out[method] = {c: v[1] for c, v in by_c.items()}
    return out


def _write_csv(path: Path, data: Dict[str, Dict[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "concurrency", "steps_per_min"])
        for method, by_c in sorted(data.items()):
            for c, steps_per_sec in sorted(by_c.items()):
                w.writerow([method, c, steps_per_sec * 60.0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", default="evaluation/outputs")
    ap.add_argument("--out", default="evaluation/reports/hle_qwen3_8b_throughput.png")
    ap.add_argument("--out-csv", default="evaluation/reports/hle_qwen3_8b_throughput.csv")
    ap.add_argument("--title", default="ToolOrchestra (HLE) - Qwen3-8B")
    ap.add_argument("--annotate-ratio", action="store_true", default=True)
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    data = _collect(outputs_dir)
    if not data:
        print(f"No runs found in {outputs_dir}. Expect hle_paper_* dirs with combined_summary.json.")
        return 1

    _write_csv(Path(args.out_csv), data)

    order = ["baseline", "continuum", "thunderagent"]
    label_map = {
        "baseline": "baseline",
        "continuum": "continuum",
        "thunderagent": "thunderagent",
    }
    style = {
        "baseline": {"color": "#5aa6b1", "marker": "^"},
        "continuum": {"color": "#4ca356", "marker": "o"},
        "thunderagent": {"color": "#e5745b", "marker": "s"},
    }

    plt.figure(figsize=(7.0, 4.8), dpi=160)
    for method in order:
        if method not in data:
            continue
        by_c = data[method]
        xs = sorted(by_c.keys())
        ys = [by_c[c] * 60.0 for c in xs]
        st = style.get(method, {})
        plt.plot(xs, ys, label=label_map.get(method, method), color=st.get("color"), marker=st.get("marker"))

    plt.xlabel("Parallel workflow number")
    plt.ylabel("Throughput (step/min)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if args.annotate_ratio and "baseline" in data:
        base = data["baseline"]
        max_c = max(base.keys())
        base_val = base.get(max_c)
        if base_val:
            for method in ["thunderagent", "continuum"]:
                if method in data and max_c in data[method]:
                    ratio = data[method][max_c] / base_val
                    xs = max_c
                    ys = data[method][max_c] * 60.0
                    plt.text(xs + 0.5, ys + 0.1, f"{ratio:.2f}×", color=style[method]["color"], fontsize=12)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
