"""Measure docker pull throughput for SWE-bench Lite test images."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

try:
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'datasets' package is required. Install it via `pip install datasets` before running this script."
    ) from exc


# Ensure authenticated access to Hugging Face to avoid rate limits.
os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", "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")


DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate docker pull throughput across batch sizes using SWE-bench Lite test images.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-Bench_Lite",
        help="HuggingFace dataset name for SWE-bench (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pull_measurements.json"),
        help="Where to write measurement results (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum concurrent docker pull processes. Defaults to the batch size itself.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing docker operations.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-image pull progress (command output).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout (seconds) for each docker pull command.",
    )
    return parser.parse_args()


def load_images(dataset_name: str, split: str) -> list[str]:
    ds = load_dataset(dataset_name, split=split)  # type: ignore[arg-type]
    images: list[str] = []
    for record in ds:
        image = record.get("image_name")
        if isinstance(image, str) and image:
            images.append(image)
            continue
        instance_id = record.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            docker_id = instance_id.replace("__", "_1776_").lower()
            fallback = f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest"
            images.append(fallback)
    if not images:
        raise RuntimeError("No image names found in dataset. Check dataset fields.")
    return images


def unique_in_order(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def remove_images(images: Iterable[str], dry_run: bool) -> None:
    for image in images:
        cmd = ["docker", "image", "rm", "-f", image]
        if dry_run:
            print("[dry-run]", " ".join(cmd))
            continue
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def pull_image(image: str, verbose: bool, dry_run: bool, timeout: int | None = None) -> tuple[str, int]:
    cmd = ["docker", "pull", image]
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return image, 0
    completed = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=timeout, check=False)
    if verbose and not dry_run:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
    return image, completed.returncode


def pull_batch(images: list[str], batch_size: int, max_workers: int | None, verbose: bool, dry_run: bool, timeout: int | None) -> dict[str, Any]:
    # ensure unique images and limit to batch size
    unique_images = unique_in_order(images)[:batch_size]
    remove_images(unique_images, dry_run=dry_run)

    start_time = time.time()
    exit_codes: dict[str, int] = {}

    if dry_run:
        for image in unique_images:
            pull_image(image, verbose=verbose, dry_run=True, timeout=timeout)
            exit_codes[image] = 0
    else:
        workers = max_workers or batch_size
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(pull_image, image, verbose, False, timeout): image
                for image in unique_images
            }
            for future in as_completed(futures):
                image, code = future.result()
                exit_codes[image] = code
    end_time = time.time()

    failed = [img for img, code in exit_codes.items() if code != 0]
    if failed:
        print(f"Warning: {len(failed)} pulls failed for batch size {batch_size}: {failed}")

    if not dry_run:
        remove_images(unique_images, dry_run=False)

    return {
        "batch_size": batch_size,
        "num_images": len(unique_images),
        "elapsed_seconds": end_time - start_time,
        "failed": failed,
    }


def main() -> None:
    args = parse_args()
    args.batch_sizes = sorted(set(args.batch_sizes))

    images = load_images(args.dataset, args.split)

    results: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        if batch_size <= 0:
            continue
        if batch_size > len(images):
            print(f"Skipping batch size {batch_size}: dataset only has {len(images)} records.")
            continue
        print(f"\n=== Batch size {batch_size} ===")
        batch_images = images[:batch_size]
        measurement = pull_batch(
            batch_images,
            batch_size=batch_size,
            max_workers=args.max_workers,
            verbose=args.verbose,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
        print(
            f"Batch {batch_size}: pulled {measurement['num_images']} images in {measurement['elapsed_seconds']:.2f} s"
        )
        results.append(measurement)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved measurements to {args.output}")


if __name__ == "__main__":
    import sys
    main()

