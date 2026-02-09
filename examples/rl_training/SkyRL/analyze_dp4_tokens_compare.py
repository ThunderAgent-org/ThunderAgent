#!/usr/bin/env python3
"""Compare per-step rollout tokens/sec for DP=4 default vs TR(atw=0.1).

Definition (same as analyze_all.py):
  tokens_per_sec = (TRAJS_PER_STEP * avg_response_length) / generate_duration_seconds

Inputs are SLURM job IDs. The script reads slurm-<JOBID>.err and prints a
focused markdown table to stdout.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

TRAJS_PER_STEP_DEFAULT = 396  # 99 prompts x 4 samples
ANSI_RE = re.compile(r"\x1b\[\d+(?:;\d+)*m")
STEP_START_RE = re.compile(r"Started: 'step'")
GEN_FINISH_RE = re.compile(r"Finished: 'generate', time cost: ([\d.]+)s")
REWARDS_RE = re.compile(
    r"avg_final_rewards:\s*([\d.eE+-]+),\s*avg_response_length:\s*([\d.eE+-]+)"
)


@dataclass
class StepInfo:
    step_num: int
    gen_duration: Optional[float] = None
    avg_resp_len: Optional[float] = None

    def tokens_per_sec(self, trajs_per_step: int) -> Optional[float]:
        if self.gen_duration is None or self.avg_resp_len is None or self.gen_duration <= 0:
            return None
        return (trajs_per_step * self.avg_resp_len) / self.gen_duration


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_slurm_err(path: str) -> List[StepInfo]:
    steps: List[StepInfo] = []
    current: Optional[StepInfo] = None
    step_num = 0

    with open(path, "r", errors="replace") as f:
        for raw in f:
            line = strip_ansi(raw)

            if STEP_START_RE.search(line):
                step_num += 1
                current = StepInfo(step_num=step_num)
                steps.append(current)
                continue

            m = GEN_FINISH_RE.search(line)
            if m and current:
                current.gen_duration = float(m.group(1))
                continue

            m = REWARDS_RE.search(line)
            if m and current:
                current.avg_resp_len = float(m.group(2))
                continue

    return steps


def steps_to_tps(steps: List[StepInfo], trajs_per_step: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for s in steps:
        tps = s.tokens_per_sec(trajs_per_step)
        if tps is not None:
            out[s.step_num] = tps
    return out


def fmt(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-job", required=True, help="SLURM job ID for DP=4 default run")
    parser.add_argument("--tr-job", required=True, help="SLURM job ID for DP=4 TR(atw=0.1) run")
    parser.add_argument(
        "--base-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory containing slurm-<JOBID>.err (default: script directory)",
    )
    parser.add_argument(
        "--trajs-per-step",
        type=int,
        default=TRAJS_PER_STEP_DEFAULT,
        help=f"Trajectories per rollout step (default: {TRAJS_PER_STEP_DEFAULT})",
    )
    parser.add_argument("--default-label", default="dp4_default")
    parser.add_argument("--tr-label", default="dp4_tr_atw01")
    args = parser.parse_args()

    default_path = os.path.join(args.base_dir, f"slurm-{args.default_job}.err")
    tr_path = os.path.join(args.base_dir, f"slurm-{args.tr_job}.err")

    missing = [p for p in [default_path, tr_path] if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"ERROR: missing log file: {p}")
        raise SystemExit(1)

    default_steps = parse_slurm_err(default_path)
    tr_steps = parse_slurm_err(tr_path)
    default_tps = steps_to_tps(default_steps, args.trajs_per_step)
    tr_tps = steps_to_tps(tr_steps, args.trajs_per_step)

    all_step_ids = sorted(set(default_tps.keys()) | set(tr_tps.keys()))

    print(
        f"# DP=4 Rollout Tokens/sec Comparison ({args.default_label} vs {args.tr_label})"
    )
    print()
    print(f"- default job: `{args.default_job}`")
    print(f"- tr job: `{args.tr_job}`")
    print(
        f"- formula: `tokens/sec = ({args.trajs_per_step} * avg_response_length) / generate_duration_s`"
    )
    print()

    headers = [
        "Step",
        f"{args.default_label} tokens/sec",
        f"{args.tr_label} tokens/sec",
        "Delta (tr-default)",
        "Delta %",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|---:|---:|---:|---:|---:|")

    paired_default: List[float] = []
    paired_tr: List[float] = []

    for step_id in all_step_ids:
        d = default_tps.get(step_id)
        t = tr_tps.get(step_id)

        delta = None
        delta_pct = None
        if d is not None and t is not None:
            delta = t - d
            if d != 0:
                delta_pct = (delta / d) * 100.0
            paired_default.append(d)
            paired_tr.append(t)

        print(
            "| "
            + " | ".join(
                [
                    str(step_id),
                    fmt(d),
                    fmt(t),
                    fmt(delta),
                    fmt(delta_pct),
                ]
            )
            + " |"
        )

    if paired_default and paired_tr:
        avg_d = sum(paired_default) / len(paired_default)
        avg_t = sum(paired_tr) / len(paired_tr)
        avg_delta = avg_t - avg_d
        avg_delta_pct = (avg_delta / avg_d) * 100.0 if avg_d != 0 else None
        print(
            "| "
            + " | ".join(
                [
                    "**Average (paired steps)**",
                    f"**{fmt(avg_d)}**",
                    f"**{fmt(avg_t)}**",
                    f"**{fmt(avg_delta)}**",
                    f"**{fmt(avg_delta_pct)}**",
                ]
            )
            + " |"
        )
    else:
        print("\nNo paired completed rollout steps found yet.")


if __name__ == "__main__":
    main()
