"""
Pre-pull SWE-bench Docker images required by the training/eval dataset.

Reads the parquet data files, computes Docker image names using the same logic
as mini_swe_utils.get_docker_image_name(), and pulls them in parallel.

Usage:
    python pull_swebench_images.py --data_dir /scratch/data/swe_gym_subset [--parallel 4]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def get_docker_image_name(instance_id: str, data_source: str) -> str:
    """Mirror of mini_swe_utils.get_docker_image_name logic."""
    if "swe-gym" in data_source.lower():
        id_docker = instance_id.replace("__", "_s_")
        return f"xingyaoww/sweb.eval.x86_64.{id_docker}:latest".lower()
    elif "swe-bench" in data_source.lower():
        id_docker = instance_id.replace("__", "_1776_")
        return f"swebench/sweb.eval.x86_64.{id_docker}:latest".lower()
    else:
        raise ValueError(f"Unknown data_source: {data_source}")


CORRUPTION_PATTERNS = ["invalid tar header", "invalid deflate data", "layers from manifest don't match"]


def _is_corruption_error(msg: str) -> bool:
    return any(p in msg for p in CORRUPTION_PATTERNS)


def pull_image(image: str) -> tuple[str, bool, str]:
    """Pull a single Docker image. Returns (image, success, message).

    If the pull fails due to corrupted cached layers, removes the image
    and retries once.
    """
    for attempt in range(2):
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                return (image, True, "ok")
            err = result.stderr.strip().split("\n")[-1] if result.stderr else "unknown"
            if attempt == 0 and _is_corruption_error(err):
                # Remove corrupted cached layers and retry
                subprocess.run(["docker", "rmi", "-f", image], capture_output=True, timeout=60)
                continue
            return (image, False, err)
        except subprocess.TimeoutExpired:
            return (image, False, "timeout")
        except Exception as e:
            return (image, False, str(e))
    return (image, False, "failed after retry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory with train.parquet and validation.parquet")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel pulls")
    parser.add_argument("--train_only", action="store_true", help="Only pull training images (skip eval)")
    args = parser.parse_args()

    # Collect unique images from dataset
    images = set()
    parquet_files = [("train", os.path.join(args.data_dir, "train.parquet"))]
    if not args.train_only:
        parquet_files.append(("validation", os.path.join(args.data_dir, "validation.parquet")))

    for split, path in parquet_files:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {split}")
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            instance = row["instance"]
            iid = instance["instance_id"] if isinstance(instance, dict) else instance.get("instance_id", None)
            data_source = row["data_source"]
            if iid:
                images.add(get_docker_image_name(iid, data_source))
        print(f"{split}: {len(df)} rows, {len(images)} unique images so far")

    print(f"\nTotal unique images to pull: {len(images)}")

    # Pull in parallel (docker pull skips already-cached images automatically)
    succeeded = 0
    failed = 0
    failed_images = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(pull_image, img): img for img in sorted(images)}
        for i, future in enumerate(as_completed(futures), 1):
            img, success, msg = future.result()
            if success:
                succeeded += 1
                print(f"  [{i}/{len(images)}] OK: {img}")
            else:
                failed += 1
                failed_images.append((img, msg))
                print(f"  [{i}/{len(images)}] FAILED: {img} ({msg})")

    print(f"\nDone: {succeeded} succeeded, {failed} failed out of {len(images)} total")
    if failed_images:
        print("Failed images:")
        for img, msg in failed_images:
            print(f"  {img}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
