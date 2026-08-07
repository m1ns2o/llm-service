#!/usr/bin/env python3
"""Verify and split the 14B GGUF, then emit a browser shard manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen36-14b-a3b-fablevibes-q4km"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_spec() -> dict:
    data = json.loads((ROOT / "configs/models.json").read_text(encoding="utf-8"))
    model = next(item for item in data["models"] if item["id"] == MODEL_ID)
    return model["artifacts"][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="complete original Q4_K_M GGUF")
    parser.add_argument("output_dir", type=Path, help="directory for generated shards")
    parser.add_argument("--base-url", required=True, help="public URL prefix containing the shards")
    parser.add_argument("--splitter", default=shutil.which("llama-gguf-split"))
    parser.add_argument("--max-size", default="512M")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "browser/model-shards/qwen36-14b-a3b-fablevibes-q4km.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.splitter:
        raise SystemExit("llama-gguf-split was not found; install or pass --splitter")
    source = args.input.resolve()
    spec = model_spec()
    if not source.is_file():
        raise SystemExit(f"input file not found: {source}")
    if source.stat().st_size != spec["size_bytes"]:
        raise SystemExit(
            f"size mismatch: expected {spec['size_bytes']}, got {source.stat().st_size}"
        )
    observed = sha256_file(source)
    if observed != spec["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: expected {spec['sha256']}, got {observed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / source.stem
    subprocess.run(
        [args.splitter, "--split", "--split-max-size", args.max_size, str(source), str(prefix)],
        check=True,
    )
    shards = sorted(args.output_dir.glob(f"{source.stem}-*-of-*.gguf"))
    if len(shards) < 2:
        raise SystemExit("splitter did not produce multiple GGUF shards")

    base_url = args.base_url.rstrip("/")
    records = []
    for shard in shards:
        records.append(
            {
                "filename": shard.name,
                "url": f"{base_url}/{shard.name}",
                "size_bytes": shard.stat().st_size,
                "sha256": sha256_file(shard),
            }
        )

    manifest = {
        "version": 1,
        "status": "ready",
        "model_id": MODEL_ID,
        "format": "GGUF split",
        "quantization": spec["quantization"],
        "source": {
            "filename": spec["filename"],
            "url": spec["url"],
            "size_bytes": spec["size_bytes"],
            "sha256": spec["sha256"],
            "verified_before_split": True,
        },
        "split": {
            "max_size": args.max_size,
            "first_shard_url": records[0]["url"],
            "total_size_bytes": sum(item["size_bytes"] for item in records),
            "shard_count": len(records),
        },
        "shards": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
