#!/usr/bin/env python3
"""Combine measured LFM/Qwen browser VLM runs into one comparison report."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
SOURCES = {
    "lfm25-vl16b": RESULTS / "lfm25-vl16b-browser-webgpu-multimodal.json",
    "qwen35-2b": RESULTS / "qwen35-2b-browser-webgpu-multimodal.json",
    "qwen35-4b": RESULTS / "qwen35-4b-browser-webgpu-multimodal.json",
}
COMBINED = RESULTS / "lfm25-vl16b-vs-qwen35-2b-4b-browser-webgpu.json"
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
    "qwen35-2b": {
        "source": "https://huggingface.co/Qwen/Qwen3.5-2B",
        "scores": {
            "MMMU": 64.2,
            "RealWorldQA": 74.5,
            "MMBench_variant": 83.3,
            "OCRBench_variant": 84.5,
        },
        "notes": "First value from the model card's paired Qwen3.5-2B vision scores; full-precision vendor evaluation.",
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
    models = comparison["models"]
    lfm = models["lfm25-vl16b"]
    qwen2 = models["qwen35-2b"]
    qwen4 = models["qwen35-4b"]
    lines = [
        "# LFM2.5-VL-1.6B vs Qwen3.5-2B/4B 브라우저 멀티모달 비교",
        "",
        "## 결론",
        "",
        comparison["decision"]["recommendation_ko"],
        "",
        "## 동일 브라우저 실측",
        "",
        "| 항목 | LFM2.5-VL-1.6B | Qwen3.5-2B | Qwen3.5-4B |",
        "|---|---:|---:|---:|",
        f"| 모델+mmproj | {lfm['artifact_size_bytes'] / 1_000_000_000:.2f} GB | {qwen2['artifact_size_bytes'] / 1_000_000_000:.2f} GB | {qwen4['artifact_size_bytes'] / 1_000_000_000:.2f} GB |",
        f"| 모델 로드 | {lfm['model_load_ms'] / 1000:.2f}s | {qwen2['model_load_ms'] / 1000:.2f}s | {qwen4['model_load_ms'] / 1000:.2f}s |",
        f"| 첫 표시 토큰 중앙값 | {lfm['median_first_visible_token_ms'] / 1000:.2f}s | {qwen2['median_first_visible_token_ms'] / 1000:.2f}s | {qwen4['median_first_visible_token_ms'] / 1000:.2f}s |",
        f"| 디코드 중앙값 | {lfm['median_decode_tok_per_s']:.2f} tok/s | {qwen2['median_decode_tok_per_s']:.2f} tok/s | {qwen4['median_decode_tok_per_s']:.2f} tok/s |",
        f"| 평균 전체 응답 | {lfm['mean_end_to_end_ms'] / 1000:.2f}s | {qwen2['mean_end_to_end_ms'] / 1000:.2f}s | {qwen4['mean_end_to_end_ms'] / 1000:.2f}s |",
        f"| 정답 필드 | {lfm['quality_assertions_passed']}/{lfm['quality_assertions_total']} ({lfm['quality_percent']:.1f}%) | {qwen2['quality_assertions_passed']}/{qwen2['quality_assertions_total']} ({qwen2['quality_percent']:.1f}%) | {qwen4['quality_assertions_passed']}/{qwen4['quality_assertions_total']} ({qwen4['quality_percent']:.1f}%) |",
        f"| 완전 통과 문항 | {lfm['tasks_passed']}/{lfm['tasks_total']} | {qwen2['tasks_passed']}/{qwen2['tasks_total']} | {qwen4['tasks_passed']}/{qwen4['tasks_total']} |",
        "",
        "조건: Chrome 151, AMD RDNA3 WebGPU, wllama 3.5.1, Q4_K_M 언어 모델 + Q8_0 비전 프로젝터, temperature 0, 동일 합성 이미지 8개입니다. 비전 토큰은 LFM 공식 권장 64~256, Qwen 런타임 요구 1024로 설정했습니다.",
        "",
        "## 공식 벤치마크 참고",
        "",
        "| 벤치마크 | LFM2.5-VL-1.6B | Qwen3.5-2B | Qwen3.5-4B |",
        "|---|---:|---:|---:|",
        f"| MMMU | {OFFICIAL['lfm25-vl16b']['scores']['MMMU']:.2f} | {OFFICIAL['qwen35-2b']['scores']['MMMU']:.1f} | {OFFICIAL['qwen35-4b']['scores']['MMMU']:.1f} |",
        f"| RealWorldQA | {OFFICIAL['lfm25-vl16b']['scores']['RealWorldQA']:.2f} | {OFFICIAL['qwen35-2b']['scores']['RealWorldQA']:.1f} | {OFFICIAL['qwen35-4b']['scores']['RealWorldQA']:.1f} |",
        "",
        "공식 수치는 각 제조사 모델 카드의 full-precision 평가입니다. MMBench와 OCRBench는 표기 변형 또는 버전이 달라 방향 참고만 가능합니다.",
        "",
        "## 제한",
        "",
        "- 로컬 평가는 합성 OCR·도표·공간·계수 문항 8개인 기능성 벤치마크이며 MMMU 대체물이 아닙니다.",
        "- Qwen GGUF는 고정 revision의 제3자 변환이며, LFM GGUF는 LiquidAI 공식 배포본입니다.",
        "- 모델 크기와 아키텍처 효과가 함께 반영된 실제 배포 비교입니다.",
        "- 모델별 정상 품질 설정을 사용했으므로 고정 연산량 비교가 아니라 권장 배포 설정의 지연시간입니다.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    raw = {key: load_result(path) for key, path in SOURCES.items()}
    models = {key: summarize(value) for key, value in raw.items()}
    lfm = models["lfm25-vl16b"]
    qwen2 = models["qwen35-2b"]
    qwen4 = models["qwen35-4b"]
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
            "qwen2_over_lfm": {
                "artifact_size": ratio(qwen2["artifact_size_bytes"], lfm["artifact_size_bytes"]),
                "load_time": ratio(qwen2["model_load_ms"], lfm["model_load_ms"]),
                "first_visible": ratio(qwen2["median_first_visible_token_ms"], lfm["median_first_visible_token_ms"]),
                "decode_speed": ratio(qwen2["median_decode_tok_per_s"], lfm["median_decode_tok_per_s"]),
                "end_to_end": ratio(qwen2["mean_end_to_end_ms"], lfm["mean_end_to_end_ms"]),
            },
            "qwen4_over_lfm": {
                "artifact_size": ratio(qwen4["artifact_size_bytes"], lfm["artifact_size_bytes"]),
                "load_time": ratio(qwen4["model_load_ms"], lfm["model_load_ms"]),
                "first_visible": ratio(qwen4["median_first_visible_token_ms"], lfm["median_first_visible_token_ms"]),
                "decode_speed": ratio(qwen4["median_decode_tok_per_s"], lfm["median_decode_tok_per_s"]),
                "end_to_end": ratio(qwen4["mean_end_to_end_ms"], lfm["mean_end_to_end_ms"]),
            },
        },
        "official_reference": OFFICIAL,
        "decision": {
            "speed_and_footprint_winner": "lfm25-vl16b",
            "multimodal_quality_winner": "qwen35-4b",
            "balanced_candidate": "qwen35-2b",
            "recommendation_ko": "Qwen3.5-2B가 현재 조건의 균형형 1순위입니다. LFM보다 0.32GB 큰 대신 정답 필드가 13/19에서 17/19로 늘었고, Qwen 4B보다 1.44GB 작으면서 디코드는 약 1.9배 빠른데 정답 필드 차이는 17/19 대 18/19였습니다. 최저 지연시간이 절대 기준이면 LFM, 최대 정확도가 기준이면 Qwen 4B를 선택할 수 있습니다.",
        },
        "limitations": [
            "The local quality set has eight synthetic tasks and is not a replacement for MMMU or RealWorldQA.",
            "Vendor benchmark scores are full-precision and were not produced by the browser quantizations.",
            "Qwen GGUF files are pinned third-party conversions; the LFM GGUF is an official LiquidAI artifact.",
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
