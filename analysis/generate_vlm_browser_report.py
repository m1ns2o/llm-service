#!/usr/bin/env python3
"""Combine the measured LFM/Qwen browser VLM runs into one comparison report."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
SOURCES = {
    "lfm25-vl16b": RESULTS / "lfm25-vl16b-browser-webgpu-multimodal.json",
    "qwen35-4b": RESULTS / "qwen35-4b-browser-webgpu-multimodal.json",
}
COMBINED = RESULTS / "lfm25-vl16b-vs-qwen35-4b-browser-webgpu.json"
REPORT = ROOT / "analysis" / "generated" / "vlm-browser-multimodal-comparison.md"

OFFICIAL = {
    "lfm25-vl16b": {
        "source": "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B",
        "scores": {
            "MMMU": 40.56,
            "RealWorldQA": 64.84,
            "MMBench_variant": 76.96,
            "OCRBench_variant": 41.44,
        },
        "notes": "MMMU Val; MMBench average; OCRBench v2; full-precision vendor evaluation.",
    },
    "qwen35-4b": {
        "source": "https://huggingface.co/Qwen/Qwen3.5-4B",
        "scores": {
            "MMMU": 77.6,
            "RealWorldQA": 79.5,
            "MMBench_variant": 89.4,
            "OCRBench_variant": 85.0,
        },
        "notes": "MMBench EN-DEV-v1.1 and unspecified OCRBench version; full-precision vendor evaluation.",
    },
}


def load_result(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "complete":
        raise ValueError(f"incomplete benchmark result: {path}")
    if result.get("task_count") != 8 or len(result.get("requests", [])) != 8:
        raise ValueError(f"expected eight tasks: {path}")
    if not result.get("image_input_supported"):
        raise ValueError(f"model did not confirm image input: {path}")
    if not result.get("download_verified"):
        raise ValueError(f"artifact hashes were not verified: {path}")
    return result


def summarize(result: dict) -> dict:
    requests = result["requests"]
    checks = [check for request in requests for check in request["quality"]["checks"]]
    passed_checks = sum(bool(check["passed"]) for check in checks)
    passed_tasks = sum(bool(request["quality"]["passed"]) for request in requests)
    return {
        "model": result["model"],
        "generation_settings": result["generation_settings"],
        "artifact_size_bytes": sum(item["size_bytes"] for item in result["artifacts"]),
        "model_load_ms": result["model_load_ms"],
        "median_first_visible_token_ms": round(
            statistics.median(request["first_visible_token_ms"] for request in requests), 2
        ),
        "median_decode_tok_per_s": round(
            statistics.median(request["decode_tok_per_s"] for request in requests), 2
        ),
        "mean_prompt_tok_per_s": round(
            statistics.mean(request["prompt_tok_per_s"] for request in requests), 2
        ),
        "mean_end_to_end_ms": round(
            statistics.mean(request["total_ms"] for request in requests), 2
        ),
        "quality_assertions_passed": passed_checks,
        "quality_assertions_total": len(checks),
        "quality_percent": round(passed_checks / len(checks) * 100, 2),
        "tasks_passed": passed_tasks,
        "tasks_total": len(requests),
        "requests": requests,
    }


def ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 2)


def make_markdown(comparison: dict) -> str:
    lfm = comparison["models"]["lfm25-vl16b"]
    qwen = comparison["models"]["qwen35-4b"]
    ratios = comparison["ratios"]
    lines = [
        "# LFM2.5-VL-1.6B vs Qwen3.5-4B 브라우저 멀티모달 비교",
        "",
        "## 결론",
        "",
        "LFM은 브라우저 배포 크기와 속도에서 확실히 유리하지만, 멀티모달 품질은 Qwen3.5-4B가 우세하다. "
        "따라서 빠른 간단 이미지 질의에는 LFM이 적합하고, 한국어 OCR·표 계산·복합 시각 추론의 정답률이 우선이면 Qwen을 유지해야 한다.",
        "",
        "## 동일 브라우저 실측",
        "",
        "| 항목 | LFM2.5-VL-1.6B | Qwen3.5-4B | 해석 |",
        "|---|---:|---:|---|",
        f"| 모델+mmproj | {lfm['artifact_size_bytes'] / 1_000_000_000:.2f} GB | {qwen['artifact_size_bytes'] / 1_000_000_000:.2f} GB | LFM {ratios['artifact_size_qwen_over_lfm']}배 작음 |",
        f"| 모델 로드 | {lfm['model_load_ms'] / 1000:.2f}s | {qwen['model_load_ms'] / 1000:.2f}s | LFM {ratios['load_time_qwen_over_lfm']}배 빠름 |",
        f"| 첫 표시 토큰 중앙값 | {lfm['median_first_visible_token_ms'] / 1000:.2f}s | {qwen['median_first_visible_token_ms'] / 1000:.2f}s | LFM {ratios['first_visible_qwen_over_lfm']}배 빠름 |",
        f"| 디코드 중앙값 | {lfm['median_decode_tok_per_s']:.2f} tok/s | {qwen['median_decode_tok_per_s']:.2f} tok/s | LFM {ratios['decode_lfm_over_qwen']}배 빠름 |",
        f"| 평균 전체 응답 | {lfm['mean_end_to_end_ms'] / 1000:.2f}s | {qwen['mean_end_to_end_ms'] / 1000:.2f}s | LFM {ratios['end_to_end_qwen_over_lfm']}배 빠름 |",
        f"| 정답 필드 | {lfm['quality_assertions_passed']}/{lfm['quality_assertions_total']} ({lfm['quality_percent']:.1f}%) | {qwen['quality_assertions_passed']}/{qwen['quality_assertions_total']} ({qwen['quality_percent']:.1f}%) | Qwen 우세 |",
        f"| 완전 통과 문항 | {lfm['tasks_passed']}/{lfm['tasks_total']} | {qwen['tasks_passed']}/{qwen['tasks_total']} | Qwen 우세 |",
        "",
        "조건: Chrome 151, AMD RDNA3 WebGPU, wllama 3.5.1, Q4_K_M 언어 모델 + Q8_0 비전 프로젝터, temperature 0, 동일한 합성 이미지 8개. 첫 요청 전에는 이미지 경로 워밍업 1회를 수행했다. 비전 토큰은 LFM 공식 권장 64–256, Qwen 런타임 요구 1024로 설정했다.",
        "",
        "## 공식 공통 벤치마크 참고",
        "",
        "| 벤치마크 | LFM2.5-VL-1.6B | Qwen3.5-4B |",
        "|---|---:|---:|",
        f"| MMMU | {OFFICIAL['lfm25-vl16b']['scores']['MMMU']:.2f} | {OFFICIAL['qwen35-4b']['scores']['MMMU']:.1f} |",
        f"| RealWorldQA | {OFFICIAL['lfm25-vl16b']['scores']['RealWorldQA']:.2f} | {OFFICIAL['qwen35-4b']['scores']['RealWorldQA']:.1f} |",
        "",
        "이 수치는 각 제조사 모델 카드의 full-precision 평가이며 로컬 양자화 실측과 직접 합산하지 않는다. MMBench와 OCRBench는 표기된 변형·버전이 달라 방향성 참고만 가능하다.",
        "",
        "## 제한",
        "",
        "- 로컬 품질 평가는 합성 OCR·도표·공간·계수 문항 8개인 기능성 벤치마크이며 MMMU 대체물이 아니다.",
        "- Qwen GGUF는 공식 원본을 mradermacher가 변환한 고정 revision이고, LFM GGUF는 LiquidAI 공식 배포본이다.",
        "- LFM 아티팩트가 2.34배 작으므로 속도 차이는 모델 구조뿐 아니라 모델 규모 차이도 포함한다. 이는 실제 브라우저 배포 비교에는 유효하지만 순수 아키텍처 비교는 아니다.",
        "- 비전 토큰 예산도 각 모델의 정상 품질 설정이 달라, 속도는 동일 연산량 비교가 아니라 실제 권장 배포 설정의 지연시간이다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    raw = {key: load_result(path) for key, path in SOURCES.items()}
    models = {key: summarize(value) for key, value in raw.items()}
    lfm = models["lfm25-vl16b"]
    qwen = models["qwen35-4b"]
    comparison = {
        "schema_version": 1,
        "benchmark": "browser-vlm-synthetic-v1",
        "observed_at": max(value["finished_at"] for value in raw.values()),
        "environment": {
            "browser": raw["lfm25-vl16b"]["user_agent"],
            "webgpu": raw["lfm25-vl16b"]["webgpu"],
            "runtime": raw["lfm25-vl16b"]["runtime"],
            "common_generation_settings": {
                "temperature": 0,
                "top_k": 1,
                "max_tokens": 96,
                "n_ctx": 4096,
            },
            "vision_token_budget": {
                key: {
                    "image_min_tokens": value["generation_settings"]["image_min_tokens"],
                    "image_max_tokens": value["generation_settings"]["image_max_tokens"],
                }
                for key, value in raw.items()
            },
        },
        "models": models,
        "ratios": {
            "artifact_size_qwen_over_lfm": ratio(qwen["artifact_size_bytes"], lfm["artifact_size_bytes"]),
            "load_time_qwen_over_lfm": ratio(qwen["model_load_ms"], lfm["model_load_ms"]),
            "first_visible_qwen_over_lfm": ratio(qwen["median_first_visible_token_ms"], lfm["median_first_visible_token_ms"]),
            "decode_lfm_over_qwen": ratio(lfm["median_decode_tok_per_s"], qwen["median_decode_tok_per_s"]),
            "end_to_end_qwen_over_lfm": ratio(qwen["mean_end_to_end_ms"], lfm["mean_end_to_end_ms"]),
        },
        "official_reference": OFFICIAL,
        "decision": {
            "speed_and_footprint_winner": "lfm25-vl16b",
            "multimodal_quality_winner": "qwen35-4b",
            "recommendation": "Use LFM for latency/footprint-sensitive browser or Android features; use Qwen when multimodal correctness is the primary gate.",
        },
        "limitations": [
            "The local quality set has eight synthetic tasks and is not a replacement for MMMU or RealWorldQA.",
            "Vendor benchmark scores are full-precision and were not produced by the browser quantizations.",
            "The Qwen GGUF is a pinned third-party conversion; the LFM GGUF is an official LiquidAI artifact.",
            "Model-size and architecture effects are intentionally combined because this is a deployment comparison.",
            "Vision token budgets follow each model's supported quality setting, so latency is not a fixed-compute comparison.",
        ],
    }
    COMBINED.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(make_markdown(comparison), encoding="utf-8")
    print(COMBINED)
    print(REPORT)


if __name__ == "__main__":
    main()
