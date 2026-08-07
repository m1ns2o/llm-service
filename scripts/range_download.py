#!/usr/bin/env python3
"""Download a large public artifact with parallel HTTP byte ranges.

This is intentionally limited to an explicit URL, output path, and byte size;
it does not use credentials or a cache and leaves the source artifact intact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("size", type=int)
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--direct", action="store_true", help="write ranges directly into the preallocated output")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.stat().st_size == args.size:
        print(f"already complete: {args.output}")
        return 0

    headers = subprocess.check_output(["curl", "-sSI", args.url], text=True)
    locations = [line.split(":", 1)[1].strip() for line in headers.splitlines() if line.lower().startswith("location:")]
    if not locations:
        raise RuntimeError("source did not return a redirect location")
    download_url = locations[-1]

    part_dir = args.output.parent / f".{args.output.name}.parts"
    if not args.direct:
        part_dir.mkdir(exist_ok=True)
    # Keep direct-mode boundaries aligned to MiB so dd can seek efficiently.
    width = max(1024 * 1024, ((args.size + args.parts * 1024 * 1024 - 1) // (args.parts * 1024 * 1024)) * 1024 * 1024)
    if args.direct:
        with args.output.open("wb") as destination:
            destination.truncate(args.size)

    def fetch(index: int) -> Path:
        start = index * width
        end = args.size - 1 if index == args.parts - 1 else min(args.size - 1, (index + 1) * width - 1)
        expected = end - start + 1
        if args.direct:
            curl = subprocess.Popen([
                "curl", "-sS", "-L", "--fail", "--retry", "3",
                "--range", f"{start}-{end}", download_url,
            ], stdout=subprocess.PIPE)
            assert curl.stdout is not None
            dd = subprocess.Popen([
                "dd", f"of={args.output}", "bs=1m", f"seek={start // (1024 * 1024)}",
                "conv=notrunc", "status=none",
            ], stdin=curl.stdout)
            curl.stdout.close()
            if dd.wait() != 0 or curl.wait() != 0:
                raise RuntimeError(f"range {index} direct write failed")
        else:
            part = part_dir / f"{index:04d}.part"
            if not part.exists() or part.stat().st_size != expected:
                command = [
                    "curl", "-sS", "-L", "--fail", "--retry", "3",
                    "--range", f"{start}-{end}", download_url, "-o", str(part),
                ]
                subprocess.run(command, check=True)
            if part.stat().st_size != expected:
                raise RuntimeError(f"range {index} size mismatch: {part.stat().st_size} != {expected}")
        print(f"part {index + 1}/{args.parts} ready", flush=True)
        return part if not args.direct else args.output

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parts) as pool:
        parts = list(pool.map(fetch, range(args.parts)))

    if args.direct:
        if args.output.stat().st_size != args.size:
            raise RuntimeError(f"direct output size mismatch: {args.output.stat().st_size} != {args.size}")
        print(f"complete: {args.output} ({args.size} bytes)")
        return 0

    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    if temporary.stat().st_size != args.size:
        raise RuntimeError(f"assembled size mismatch: {temporary.stat().st_size} != {args.size}")
    temporary.replace(args.output)
    print(f"complete: {args.output} ({args.size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
