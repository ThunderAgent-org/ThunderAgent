#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

def main(batch_dir: Path) -> None:
    batch_dir = batch_dir.resolve()

    success_ids = {p.name for p in batch_dir.iterdir() if p.is_dir()}
    print(f"Found {len(success_ids)} success directories.")

    preds_path = batch_dir / "preds.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"{preds_path} not found")

    content = preds_path.read_text().strip()
    if not content:
        print("preds.json is empty", flush=True)
        return

    filtered = []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            iterator = data
            for obj in iterator:
                if obj.get("instance_id") in success_ids:
                    filtered.append(obj)
        elif isinstance(data, dict):
            for obj in data.values():
                if obj.get("instance_id") in success_ids:
                    filtered.append(obj)
        else:
            # single object
            if data.get("instance_id") in success_ids:
                filtered.append(data)
    except json.JSONDecodeError:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            obj = json.loads(line)
            if obj.get("instance_id") in success_ids:
                filtered.append(obj)

    out_path = batch_dir / "preds_success.jsonl"
    out_path.write_text("\n".join(json.dumps(obj) for obj in filtered))
    print(f"Wrote {len(filtered)} predictions to {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter preds.json to only keep success instances.")
    parser.add_argument("--batch-dir", required=True, type=Path, help="Path to run-batch output directory")
    args = parser.parse_args()
    main(args.batch_dir)
