#!/usr/bin/env python3
"""Plot stacked prefill/decode token usage per step using multistep averages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt  # type: ignore[import-not-found]

PREFILL_COLOR = "#FFD166"  # bright golden
DECODE_COLOR = "#26547C"   # deep blue


def load_series(payload_path: Path, *, key: str) -> list[float]:
    with payload_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    series = payload.get(key)
    if series is None:
        raise ValueError(f"{payload_path} does not contain '{key}'.")
    if not isinstance(series, Sequence):
        raise ValueError(f"Expected list-like series for key '{key}' in {payload_path}.")
    return [float(value) for value in series]


def pad_series(series: list[float], length: int) -> list[float]:
    if len(series) >= length:
        return series[:length]
    return series + [0.0] * (length - len(series))


def plot_stacked_tokens(
    steps: list[int],
    prefill: list[float],
    decode: list[float],
    output_path: Path,
    *,
    dpi: int = 320,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 0.6), 6))
    ax.bar(steps, prefill, color=PREFILL_COLOR, label="Prefill Tokens", align="edge", width=0.8)
    ax.bar(steps, decode, bottom=prefill, color=DECODE_COLOR, label="Decode Tokens", align="edge", width=0.8)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Tokens", fontsize=12)
    ax.set_title("Prefill vs Decode Tokens per Step", fontsize=14, pad=15)
    ax.set_xlim(min(steps) - 0.1, max(steps) + 0.9)
    ax.set_xticks(steps)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved stacked token chart to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefill-json",
        type=Path,
        default=Path("input_tokens_by_step.json"),
        help="JSON output from collect_input_tokens_by_step.py (default: %(default)s).",
    )
    parser.add_argument(
        "--decode-json",
        type=Path,
        default=Path("trajectories_output_tokens.json"),
        help="JSON output from collect_output_and_total_tokens.py (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("token_breakdown.png"),
        help="Output image path (default: %(default)s).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Limit to first N steps to display (default: 50).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=320,
        help="DPI for the saved figure (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.prefill_json.exists():
        raise SystemExit(f"Prefill JSON not found: {args.prefill_json}")
    if not args.decode_json.exists():
        raise SystemExit(f"Decode JSON not found: {args.decode_json}")

    prefill_series = load_series(args.prefill_json, key="multistep")
    decode_series = load_series(args.decode_json, key="multistep")

    length = max(len(prefill_series), len(decode_series))
    if args.max_steps is not None:
        length = min(length, args.max_steps)
    prefill_series = pad_series(prefill_series, length)
    decode_series = pad_series(decode_series, length)

    steps = list(range(1, length + 1))
    plot_stacked_tokens(steps, prefill_series, decode_series, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
