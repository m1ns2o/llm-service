#!/usr/bin/env python3
"""Validate manifest artifact shape, with optional remote HEAD checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_bench.manifest import load_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "configs/models.json"))
    parser.add_argument("--remote", action="store_true", help="also compare HF HEAD size and LFS SHA-256")
    args = parser.parse_args()
    models, _ = load_manifest(args.manifest)
    checked = 0
    for model in models.values():
        for artifact in model.artifacts:
            checked += 1
            if not args.remote:
                continue
            headers = subprocess.check_output(
                ["curl", "-fsSI", "-L", "--retry", "3", artifact["url"]],
                text=True,
            )
            values: dict[str, list[str]] = {}
            for line in headers.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values.setdefault(key.lower(), []).append(value.strip())
            remote_size = int((values.get("x-linked-size") or values.get("content-length") or ["0"])[-1])
            remote_sha = (values.get("x-linked-etag") or [""])[-1].strip('"')
            if remote_size != artifact["size_bytes"]:
                raise ValueError(f"{model.id}: remote size {remote_size} != {artifact['size_bytes']}")
            if remote_sha and remote_sha.lower() != artifact["sha256"].lower():
                raise ValueError(f"{model.id}: remote SHA-256 does not match manifest")
            print(f"remote ok: {artifact['filename']} ({remote_size} bytes)")
    print(json.dumps({"status": "ok", "models": len(models), "artifacts": checked, "remote": args.remote}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
