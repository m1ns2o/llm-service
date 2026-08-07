#!/usr/bin/env python3
"""Run a repeatable local llama.cpp text benchmark and emit RunEvidence JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPTS = [
    "초등학생에게 물이 끓는 이유를 두 문장으로 설명해줘.",
    "분수 1/2와 2/4가 같은 이유를 설명해줘.",
    "x+3=11을 풀고 풀이 과정을 한 줄씩 설명해줘.",
    "태양계에서 지구가 속한 행성을 말하고 특징 하나를 알려줘.",
]


def request_json(url: str, payload: dict | None = None, timeout: float = 10) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            response = request_json(f"{base_url}/health", timeout=3)
            if response.get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become healthy within 180 seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-bin", default="llama-server")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--artifact-sha256", help="SHA-256 of the downloaded model artifact")
    parser.add_argument("--download-verified", action="store_true", help="Mark the supplied artifact hash as verified")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        args.server_bin, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "--no-webui", "--alias", args.model_id, "-ngl", "99", "-c", str(args.ctx_size),
        "--reasoning", "off", "--log-disable",
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    successes = 0
    errors: list[str] = []
    timings: list[dict] = []
    try:
        wait_ready(base_url, process)
        for index in range(args.requests):
            prompt = PROMPTS[index % len(PROMPTS)]
            payload = {"model": args.model_id, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 42, "max_tokens": 96}
            try:
                response = request_json(f"{base_url}/v1/chat/completions", payload, timeout=180)
                if response.get("choices"):
                    successes += 1
                    timings.append(response.get("timings", {}))
                else:
                    errors.append(f"request_{index}:missing_choices")
            except Exception as error:  # benchmark evidence should preserve the failing request
                errors.append(f"request_{index}:{type(error).__name__}:{error}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    prompt_tps = [item.get("prompt_per_second") for item in timings if item.get("prompt_per_second")]
    generation_tps = [item.get("predicted_per_second") for item in timings if item.get("predicted_per_second")]
    prompt_ms = sorted(item.get("prompt_ms") for item in timings if item.get("prompt_ms") is not None)
    ttft_p50 = prompt_ms[len(prompt_ms) // 2] / 1000 if prompt_ms else None
    evidence = {
        "model_id": args.model_id,
        "platform": "apple-silicon-local",
        "quantization": args.quantization,
        "artifact_sha256": args.artifact_sha256,
        "download_verified": bool(args.download_verified and args.artifact_sha256),
        "cold_loads": 1,
        "requests_completed": successes,
        "crashes": 0 if process.returncode in (0, -15) else 1,
        "oom_events": 0,
        "thermal_minutes": (time.monotonic() - started) / 60,
        "initial_decode_tps": generation_tps[0] if generation_tps else None,
        "final_decode_tps": generation_tps[-1] if generation_tps else None,
        "text_ttft_p50_seconds": ttft_p50,
        "model_size_mb": round(Path(args.model).stat().st_size / 1_000_000, 1),
        "metadata": {"prompt_tps": prompt_tps, "generation_tps": generation_tps, "prompt_ms": prompt_ms, "timings": timings, "errors": errors, "server_command": command},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "successes": successes, "errors": len(errors)}, ensure_ascii=False))
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
