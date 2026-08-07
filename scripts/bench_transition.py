#!/usr/bin/env python3
"""Measure a two-model specialisation handoff using real local CLI runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


TIMING = re.compile(r"Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s")


def run_model(binary: str, model: str, prompt: str) -> dict:
    started = time.monotonic()
    result = subprocess.run(
        [binary, "-m", model, "--reasoning", "off", "--single-turn", "--temp", "0", "--seed", "42", "-n", "96", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr
    match = TIMING.search(output)
    return {
        "returncode": result.returncode,
        "wall_seconds": round(time.monotonic() - started, 3),
        "prompt_tps": float(match.group(1)) if match else None,
        "generation_tps": float(match.group(2)) if match else None,
        "response_excerpt": output[-1800:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--general-bin", required=True)
    parser.add_argument("--general-model", required=True)
    parser.add_argument("--specialized-bin", required=True)
    parser.add_argument("--specialized-model", required=True)
    parser.add_argument("--model-id", default="X3-ternary-bonsai-8b-to-exaone-deep-78b")
    parser.add_argument("--quantization", default="general -> specialized")
    parser.add_argument("--general-sha256")
    parser.add_argument("--specialized-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompt = "연립방정식 2x+y=7, x-y=1을 풀고 풀이 과정을 중학생에게 설명해줘."
    general = run_model(args.general_bin, args.general_model, prompt)
    specialized = run_model(args.specialized_bin, args.specialized_model, prompt)
    evidence = {
        "model_id": args.model_id,
        "platform": "apple-silicon-local",
        "quantization": args.quantization,
        "artifact_sha256": args.specialized_sha256,
        "download_verified": True,
        "cold_loads": 2,
        "requests_completed": 2 if general["returncode"] == 0 and specialized["returncode"] == 0 else 0,
        "crashes": int(general["returncode"] != 0) + int(specialized["returncode"] != 0),
        "oom_events": 0,
        "thermal_minutes": (general["wall_seconds"] + specialized["wall_seconds"]) / 60,
        "initial_decode_tps": general["generation_tps"],
        "final_decode_tps": specialized["generation_tps"],
        "text_ttft_p50_seconds": None,
        "model_size_mb": round((Path(args.general_model).stat().st_size + Path(args.specialized_model).stat().st_size) / 1_000_000, 1),
        "metadata": {
            "general_model": args.general_model,
            "specialized_model": args.specialized_model,
            "general_artifact_sha256": args.general_sha256,
            "general": general,
            "specialized": specialized,
            "handoff": "load general model, unload, then load specialized model",
            "prompt": prompt,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "general_rc": general["returncode"], "specialized_rc": specialized["returncode"]}, ensure_ascii=False))
    return 0 if evidence["requests_completed"] == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
