#!/usr/bin/env python3
"""Run one multimodal browser benchmark with agent-browser and Chrome."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
TERMINAL_STATES = {"complete", "failed", "runtime_blocked"}
ARTIFACTS = {
    "lfm25-vl16b": {
        "model_name": "LFM2.5-VL-1.6B-Q4_K_M.gguf",
        "model_size": 730_896_256,
        "model_sha256": "aefc3c97c9eb30d9c0dd6af4c38250f5f5106b57c8cf92de7914c7d0a9c94da2",
        "mmproj_name": "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
        "mmproj_size": 583_109_888,
        "mmproj_sha256": "2ce89e610c56f3198ece2b86cf61743a08b9307279c89125eb2412ebb908689d",
    },
    "qwen35-4b": {
        "model_name": "Qwen3.5-4B.Q4_K_M.gguf",
        "model_size": 2_708_804_800,
        "model_sha256": "51eafbc127f35598c8f1d2ec58b2520d6126c7d1195c4eca26832e63a2939d39",
        "mmproj_name": "Qwen3.5-4B.mmproj-Q8_0.gguf",
        "mmproj_size": 366_894_656,
        "mmproj_sha256": "40a4f07d7bbdbb43011d6cf35ef751e4b1829ff47ee8aa4964c6296f571725ad",
    },
    "qwen35-2b": {
        "model_name": "Qwen3.5-2B.Q4_K_M.gguf",
        "model_size": 1_270_808_896,
        "model_sha256": "d772079a853f3494be962e1bde20b4dbf1454c89d1da4c686cf701de19fc73f1",
        "mmproj_name": "Qwen3.5-2B.mmproj-Q8_0.gguf",
        "mmproj_size": 364_664_384,
        "mmproj_sha256": "526dbf85f350baf3a5107b1f14e629e94571c7cbab4277476fbdaaa8c4a31a64",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(ARTIFACTS), required=True)
    parser.add_argument("--model-gguf", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000/vlm-browser-compare.html"
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--port",
        type=int,
        help="Deprecated compatibility option; agent-browser uses a private pipe.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, expected_name: str, size: int, sha256: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"artifact missing: {path}")
    if path.name != expected_name:
        raise SystemExit(f"artifact filename mismatch: {path.name} != {expected_name}")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise SystemExit(f"artifact size mismatch: {path.name}: {actual_size} != {size}")
    actual_hash = sha256_file(path)
    if actual_hash != sha256:
        raise SystemExit(f"artifact SHA-256 mismatch: {path.name}: {actual_hash}")
    return {
        "name": path.name,
        "size_bytes": actual_size,
        "sha256": actual_hash,
        "verified": True,
    }


def run_cli(npx: str, session: str, arguments: list[str], timeout: int = 60) -> str:
    command = [npx, "--yes", "agent-browser", "--session", session, *arguments]
    # agent-browser keeps a detached daemon. Temporary files avoid a Windows pipe
    # handle inherited by that daemon from making subprocess.run wait forever.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        completed = subprocess.run(
            command,
            check=False,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
        )
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read().decode("utf-8", errors="replace").strip()
        error_output = stderr.read().decode("utf-8", errors="replace").strip()
    if completed.returncode:
        raise RuntimeError(error_output or output or f"agent-browser exited {completed.returncode}")
    return output


def decode_eval_json(output: str) -> dict:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("agent-browser eval returned no output")
    value = json.loads(lines[-1])
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise RuntimeError("agent-browser eval did not return an object")
    return value


def main() -> int:
    args = parse_args()
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise SystemExit("npx is required to run agent-browser")
    if not CHROME.is_file():
        raise SystemExit(f"Chrome not found: {CHROME}")
    expected = ARTIFACTS[args.model]
    host_artifacts = [
        verify_artifact(
            args.model_gguf,
            expected["model_name"],
            expected["model_size"],
            expected["model_sha256"],
        ),
        verify_artifact(
            args.mmproj,
            expected["mmproj_name"],
            expected["mmproj_size"],
            expected["mmproj_sha256"],
        ),
    ]
    args.profile.mkdir(parents=True, exist_ok=True)
    session = f"vlm-{args.model}-{uuid.uuid4().hex[:8]}"
    result: dict = {}
    try:
        run_cli(
            npx,
            session,
            [
                "--executable-path",
                str(CHROME),
                "--profile",
                str(args.profile.resolve()),
                "--args",
                "--enable-unsafe-webgpu,--disable-gpu-sandbox",
                "open",
                f"{args.url}?model={args.model}",
            ],
            timeout=60,
        )
        run_cli(
            npx,
            session,
            [
                "upload",
                "#model-files",
                str(args.model_gguf.resolve()),
                str(args.mmproj.resolve()),
            ],
            timeout=60,
        )
        run_cli(
            npx,
            session,
            [
                "eval",
                "setTimeout(() => window.__vlmBrowserBenchmarkControl.runBenchmark(), 0); 'started'",
            ],
            timeout=60,
        )
        deadline = time.monotonic() + args.timeout
        last_report = 0.0
        while time.monotonic() < deadline:
            output = run_cli(
                npx,
                session,
                ["eval", "JSON.stringify(window.__vlmBrowserBenchmarkResult)"],
                timeout=60,
            )
            result = decode_eval_json(output)
            now = time.monotonic()
            if now - last_report >= 10:
                print(
                    f"status={result.get('status')} message={result.get('status_message')} "
                    f"requests={len(result.get('requests', []))}/{result.get('task_count', 8)}",
                    flush=True,
                )
                last_report = now
            if result.get("status") in TERMINAL_STATES:
                break
            time.sleep(2)
        else:
            result["status"] = "runtime_blocked"
            result["status_message"] = f"benchmark timed out after {args.timeout}s"

        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            run_cli(
                npx,
                session,
                ["screenshot", str(args.screenshot.resolve()), "--full"],
                timeout=60,
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as error:
        if not result:
            result = {
                "schema_version": 1,
                "benchmark": "browser-vlm-synthetic-v1",
                "status": "runtime_blocked",
                "status_message": str(error),
            }
        result["automation_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        try:
            run_cli(npx, session, ["close"], timeout=30)
        except (subprocess.SubprocessError, OSError):
            pass

    result["host_artifacts"] = host_artifacts
    result["download_verified"] = True
    result["automation"] = "agent-browser/Chrome debug pipe"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"result: {args.output}", flush=True)
    return 0 if result.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
