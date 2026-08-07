#!/usr/bin/env python3
"""Build a self-contained quantitative report from benchmark result JSON files.

The report deliberately separates throughput, storage, handoff cost, and
platform smoke results. It never treats a missing quality score as zero.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
OUT = ROOT / "analysis" / "generated"


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def safe(value: object) -> str:
    return html.escape(str(value))


def number(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def gate_status(data: dict) -> str:
    missing = []
    if not data.get("artifact_sha256") or not data.get("download_verified"):
        missing.append("artifact")
    if data.get("cold_loads", 0) < 3:
        missing.append("cold")
    if data.get("requests_completed", 0) < 20:
        missing.append("requests")
    if data.get("thermal_minutes", 0) < 20:
        missing.append("thermal")
    if data.get("initial_decode_tps") is None or data.get("final_decode_tps") is None:
        missing.append("tps")
    if data.get("text_ttft_p50_seconds") is None:
        missing.append("ttft")
    if data.get("crashes", 0) or data.get("oom_events", 0):
        return "fail"
    return "pass" if not missing else "incomplete"


def result_row(
    ident: str,
    name: str,
    category: str,
    source: str,
    data: dict,
    tps: float | None,
    *,
    secondary_tps: float | None = None,
    wall_seconds: float | None = None,
    note: str = "",
) -> dict:
    return {
        "id": ident,
        "name": name,
        "category": category,
        "platform": data.get("platform", ""),
        "tps": tps,
        "secondary_tps": secondary_tps,
        "ttft_ms": None if data.get("text_ttft_p50_seconds") is None else data["text_ttft_p50_seconds"] * 1000,
        "size_mb": data.get("model_size_mb"),
        "requests": data.get("requests_completed", 0),
        "gate": gate_status(data),
        "wall_seconds": wall_seconds,
        "source": source,
        "note": note,
    }


def build_rows() -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    singles_spec = [
        ("L1", "Qwen3.5-4B", "qwen35-4b-apple-v2.json"),
        ("L2", "A.X 4.0 Light", "ax-light-7b-apple.json"),
        ("L3", "Qwen3-8B", "qwen3-8b-apple.json"),
        ("L4", "Kanana 1.5 8B", "kanana-15-8b-apple.json"),
        ("L5", "EXAONE 3.5 7.8B", "exaone-35-78b-apple.json"),
        ("L6", "Ternary Bonsai 8B", "ternary-bonsai-8b-apple.json"),
        ("L7", "Ternary Bonsai 4B", "ternary-bonsai-4b-apple.json"),
        ("L8", "EXAONE Deep 7.8B", "exaone-deep-78b-apple.json"),
    ]
    singles = [
        result_row(i, n, "LLM 단독", f"benchmarks/results/{f}", d := read(f), d.get("final_decode_tps"))
        for i, n, f in singles_spec
    ]

    vlm_spec = [
        ("S1", "Qwen3.5-4B", "qwen35-4b-vlm-smoke.json", "metadata", "generation_tps"),
        ("S2", "A.X 4.0 VL Light", "S2-ax-vl-light-transformers.json", "", ""),
        ("S3", "Qwen3-VL-8B", "qwen3-vl-8b-vlm-smoke.json", "metadata", "generation_tps"),
        ("S4", "Gemma 4 E4B", "gemma4-e4b-vlm-smoke.json", "metadata", "generation_tps"),
    ]
    vlm: list[dict] = []
    for ident, name, filename, _, _ in vlm_spec:
        d = read(filename)
        tps = d.get("final_decode_tps")
        if tps is None:
            tps = d.get("metadata", {}).get("generation_tps")
        if ident == "S2":
            tps = d["metadata"]["image_smoke"]["generation_tps"]
        vlm.append(result_row(ident, name, "통합 VLM", f"benchmarks/results/{filename}", d, tps))

    pair_files = {
        "P1": ("A.X Light + Qwen3-VL-4B", "P1-ax-qwen3vl-apple.json"),
        "P2": ("A.X Light + A.X VL Light", "P2-ax-axvl-apple.json"),
        "P3": ("Qwen3-8B + Qwen3-VL-4B", "P3-qwen3-qwen3vl-apple.json"),
        "P4": ("Kanana + Qwen3-VL-4B", "P4-kanana-qwen3vl-apple.json"),
        "P5": ("EXAONE 3.5 + Qwen3-VL-4B", "P5-exaone35-qwen3vl-apple.json"),
        "P6": ("Ternary 8B + Qwen3-VL-4B", "P6-ternary-qwen3vl-apple.json"),
        "P7": ("Ternary 4B + Qwen3-VL-4B", "P7-ternary4b-qwen3vl-apple.json"),
        "P8": ("Qwen3-8B + A.X VL Light", "P8-qwen3-axvl-apple.json"),
    }
    pairs: list[dict] = []
    for ident, (name, filename) in pair_files.items():
        d = read(filename)
        meta = d.get("metadata", {})
        vlm_tps = meta.get("vlm_generation_tps")
        llm = meta.get("llm", {})
        llm_tps = meta.get("llm_generation_tps") or meta.get("llm", {}).get("generation_tps")
        if ident in {"P1", "P3", "P4", "P5", "P6", "P7"}:
            llm_tps = meta.get("llm_generation_tps")
        wall = meta.get("vlm_generation_seconds", 0) + (llm.get("wall_seconds", 0) if isinstance(llm, dict) else 0)
        pairs.append(result_row(ident, name, "분리형 P 조합", f"benchmarks/results/{filename}", d, llm_tps, secondary_tps=vlm_tps, wall_seconds=wall))

    transitions: list[dict] = []
    for ident, name, filename in [
        ("X1", "A.X Light → EXAONE Deep", "X1-ax-exaone-deep-apple.json"),
        ("X2", "Kanana → EXAONE Deep", "X2-kanana-exaone-deep-apple.json"),
        ("X3", "Ternary 8B → EXAONE Deep", "X3-ternary-exaone-deep-apple.json"),
    ]:
        d = read(filename)
        meta = d.get("metadata", {})
        general = meta.get("general", {})
        specialized = meta.get("specialized", {})
        transitions.append(result_row(ident, name, "특화 전환 X 조합", f"benchmarks/results/{filename}", d, specialized.get("generation_tps"), secondary_tps=general.get("generation_tps"), wall_seconds=(general.get("wall_seconds", 0) + specialized.get("wall_seconds", 0))))

    platform = [
        result_row("Android-Q35", "Qwen3.5-4B", "Android 에뮬레이터", "benchmarks/results/qwen35-4b-android-emulator.json", read("qwen35-4b-android-emulator.json"), 5.5, note="CLI smoke"),
        result_row("Android-T8", "Ternary Bonsai 8B", "Android 에뮬레이터", "benchmarks/results/ternary-bonsai-8b-android-emulator.json", read("ternary-bonsai-8b-android-emulator.json"), 0.2, note="CLI smoke"),
        result_row("Browser-Q35", "Qwen3.5-4B", "로컬 브라우저", "benchmarks/results/qwen35-4b-browser-local.json", read("qwen35-4b-browser-local.json"), 31.04, note="UI → localhost server"),
        result_row("Browser-WebGPU-Q35", "Qwen3.5-4B", "브라우저 WebGPU 직접", "benchmarks/results/qwen35-4b-browser-webgpu.json", read("qwen35-4b-browser-webgpu.json"), 16.5103, note="cold 16.51 tok/s; warm 21.35 tok/s; 브라우저 캐시·클라이언트 GPU 직접 생성"),
        result_row("Browser-WebGPU-Q35-9B", "Qwen3.5-9B", "브라우저 WebGPU 직접", "benchmarks/results/qwen35-9b-browser-webgpu.json", read("qwen35-9b-browser-webgpu.json"), None, note="4.8GB/127개 파라미터 파일 다운로드 완료; WebGPU 커널 컴파일 중 renderer 응답 중단, 추론 미측정"),
        result_row("Browser-WebGPU-Qwen3", "Qwen3-8B", "브라우저 WebGPU 직접", "benchmarks/results/qwen3-8b-browser-webgpu.json", read("qwen3-8b-browser-webgpu.json"), 12.6498, note="cold 12.65 tok/s; warm 14.56 tok/s; 브라우저 캐시·클라이언트 GPU 직접 생성"),
    ]
    return singles, vlm, pairs, transitions, platform


def svg_bar(path: Path, title: str, rows: list[tuple[str, float | None]], unit: str, color: str = "#2563eb", max_value: float | None = None) -> None:
    rows = [(a, v) for a, v in rows if v is not None]
    width, left, right, top, row_h = 1120, 260, 90, 78, 38
    max_value = max_value or max(v for _, v in rows) * 1.18
    height = top + row_h * len(rows) + 66
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{safe(title)}">', '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033} .muted{fill:#64748b;font-size:14px}</style>']
    out.append(f'<text x="24" y="32" font-size="22" font-weight="600">{safe(title)}</text>')
    chart_w = width - left - right
    for idx, (label, value) in enumerate(rows):
        y = top + idx * row_h
        bar_w = chart_w * value / max_value
        out.append(f'<text x="{left-14}" y="{y+22}" text-anchor="end" font-size="15">{safe(label)}</text>')
        out.append(f'<rect x="{left}" y="{y+5}" width="{bar_w:.1f}" height="24" rx="5" fill="{color}" opacity="0.88"/>')
        out.append(f'<text x="{left+bar_w+10:.1f}" y="{y+22}" font-size="15">{value:.2f} {safe(unit)}</text>')
    out.append(f'<line x1="{left}" x2="{width-right}" y1="{top+row_h*len(rows)+11}" y2="{top+row_h*len(rows)+11}" stroke="#cbd5e1"/>')
    out.append(f'<text x="{left}" y="{height-18}" class="muted">0</text><text x="{width-right}" y="{height-18}" text-anchor="end" class="muted">{max_value:.1f} {safe(unit)}</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def svg_grouped(path: Path, title: str, rows: list[tuple[str, float, float]], labels: tuple[str, str], colors: tuple[str, str]) -> None:
    width, left, right, top, group_w = 1180, 220, 50, 82, 112
    height = top + group_w * len(rows) + 70
    max_value = max(max(a, b) for _, a, b in rows) * 1.2
    chart_w = width - left - right
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{safe(title)}">', '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}.muted{fill:#64748b;font-size:13px}</style>']
    out.append(f'<text x="24" y="32" font-size="22" font-weight="600">{safe(title)}</text>')
    for j, (legend, color) in enumerate(zip(labels, colors)):
        x = width - 310 + j * 145
        out.append(f'<rect x="{x}" y="17" width="13" height="13" rx="3" fill="{color}"/><text x="{x+20}" y="29" font-size="13">{safe(legend)}</text>')
    for idx, (name, first, second) in enumerate(rows):
        y = top + idx * group_w
        out.append(f'<text x="{left-12}" y="{y+38}" text-anchor="end" font-size="14">{safe(name)}</text>')
        for j, value in enumerate((first, second)):
            bw = chart_w * value / max_value
            by = y + 12 + j * 31
            out.append(f'<rect x="{left}" y="{by}" width="{bw:.1f}" height="20" rx="4" fill="{colors[j]}" opacity="0.88"/>')
            out.append(f'<text x="{left+bw+8:.1f}" y="{by+15}" font-size="13">{value:.2f}</text>')
    out.append(f'<line x1="{left}" x2="{width-right}" y1="{top+group_w*len(rows)+10}" y2="{top+group_w*len(rows)+10}" stroke="#cbd5e1"/>')
    out.append(f'<text x="{left}" y="{height-18}" class="muted">0 tok/s</text><text x="{width-right}" y="{height-18}" text-anchor="end" class="muted">{max_value:.1f} tok/s</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def svg_scatter(path: Path, title: str, rows: list[dict]) -> None:
    width, height, left, right, top, bottom = 1120, 620, 100, 55, 80, 80
    points = [r for r in rows if r.get("size_mb") and r.get("tps") is not None]
    max_x = max(r["size_mb"] for r in points) * 1.08
    max_y = max(r["tps"] for r in points) * 1.2
    sx = lambda x: left + x / max_x * (width-left-right)
    sy = lambda y: height-bottom - y / max_y * (height-top-bottom)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{safe(title)}">', '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}.muted{fill:#64748b;font-size:13px}</style>']
    out.append(f'<text x="24" y="32" font-size="22" font-weight="600">{safe(title)}</text>')
    out.append(f'<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#94a3b8"/><line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#94a3b8"/>')
    for tick in range(0, 6):
        value = max_y * tick / 5
        y = sy(value)
        out.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        out.append(f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" class="muted">{value:.0f}</text>')
    for row in points:
        x, y = sx(row["size_mb"]), sy(row["tps"])
        color = "#dc2626" if row["id"] == "L7" else "#2563eb"
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" opacity="0.9"/><text x="{x+10:.1f}" y="{y-10:.1f}" font-size="13">{safe(row["id"])}</text>')
    out.append(f'<text x="{(left+width-right)/2}" y="{height-20}" text-anchor="middle" class="muted">모델 파일 용량 (MB)</text><text x="20" y="{(top+height-bottom)/2}" transform="rotate(-90 20 {(top+height-bottom)/2})" text-anchor="middle" class="muted">최종 생성 속도 (tok/s)</text>')
    out.append('</svg>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    singles, vlm, pairs, transitions, platform = build_rows()
    browser_gpu = read("qwen35-4b-browser-webgpu.json")
    browser_qwen35_9b = read("qwen35-9b-browser-webgpu.json")
    browser_qwen3 = read("qwen3-8b-browser-webgpu.json")
    coverage = read("coverage.json")
    audit = coverage.get("verification_audit", {})
    all_rows = singles + vlm + pairs + transitions + platform
    with (OUT / "normalized_metrics.csv").open("w", newline="", encoding="utf-8") as fp:
        fields = list(all_rows[0])
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "normalized_metrics.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    svg_bar(OUT / "single-throughput.svg", "LLM 단독: 최종 생성 속도", [(r["id"] + "  " + r["name"], r["tps"]) for r in singles], "tok/s", "#2563eb")
    svg_scatter(OUT / "single-size-speed.svg", "LLM 단독: 용량과 속도의 trade-off", singles)
    svg_bar(OUT / "vlm-throughput.svg", "통합 VLM: 이미지 생성 속도", [(r["id"] + "  " + r["name"], r["tps"]) for r in vlm], "tok/s", "#0f766e", 35)
    svg_grouped(OUT / "pair-throughput.svg", "분리형 P 조합: VLM → LLM 생성 속도", [(r["id"], r["secondary_tps"] or 0, r["tps"] or 0) for r in pairs], ("VLM", "메인 LLM"), ("#f59e0b", "#2563eb"))
    svg_grouped(OUT / "transition-throughput.svg", "특화 전환 X 조합: 일반 → 전문 모델 생성 속도", [(r["id"], r["secondary_tps"] or 0, r["tps"] or 0) for r in transitions], ("일반 모델", "전문 모델"), ("#64748b", "#dc2626"))
    svg_bar(OUT / "platform-throughput.svg", "플랫폼 smoke: 생성 속도", [(r["id"] + "  " + r["name"], r["tps"]) for r in platform], "tok/s", "#7c3aed", 35)

    fastest = max(singles, key=lambda r: r["tps"] or -1)
    smallest = min(singles, key=lambda r: r["size_mb"] or math.inf)
    vlm_fastest = max(vlm, key=lambda r: r["tps"] or -1)
    q35 = next(r for r in singles if r["id"] == "L1")["tps"]
    p2 = next(r for r in pairs if r["id"] == "P2")
    p8 = next(r for r in pairs if r["id"] == "P8")
    incomplete = sum(r["gate"] == "incomplete" for r in all_rows)
    md = f"""# 로컬 LLM/VLM 검증 결과 분석

생성일: 2026-08-07  
원본 범위: `benchmarks/results/coverage.json`의 S/L/P/X 23개 조합  
측정 정의: `measured`는 이름 그대로의 모델 또는 handoff가 실제 로컬 런타임에서 최소 1회 실행됐다는 뜻입니다. 품질 점수는 수집하지 않았으므로 속도·용량을 품질 우열로 해석하지 않습니다.

## 핵심 결론

1. 단일 LLM 속도 1위는 **{fastest['id']} {fastest['name']} ({fastest['tps']:.2f} tok/s)**이고, 가장 작은 단일 파일은 **{smallest['id']} {smallest['name']} ({smallest['size_mb']:.1f} MB)**입니다.
2. Qwen3.5-4B(L1)는 {q35:.2f} tok/s로 통합 멀티모달 기준점 역할을 합니다. 통합 VLM 중 생성 속도는 **{vlm_fastest['id']} {vlm_fastest['name']} ({vlm_fastest['tps']:.2f} tok/s)**가 가장 높았습니다.
3. A.X VL Light(S2)는 공식 BF16 원본을 직접 로드했지만 이미지 생성이 **{next(r for r in vlm if r['id']=='S2')['tps']:.4f} tok/s**로 매우 느렸습니다. P2/P8도 LLM 자체보다 VLM 단계({p2['wall_seconds']:.1f}초, {p8['wall_seconds']:.1f}초)가 전체 지연을 지배합니다.
4. P2의 메인 LLM은 {p2['tps']:.2f} tok/s, P8의 메인 LLM은 {p8['tps']:.2f} tok/s였지만, 두 조합 모두 VLM BF16 원본을 포함해 약 17.7GB/20.4GB 저장공간이 필요합니다.
5. 표준 게이트(3 cold load·20회 요청·20분 열 안정성·TTFT)를 완전히 충족하지 못한 결과가 **{incomplete}개**입니다. 현재 자료는 후보 선별용 1차 실행 분석이지 배포 인증 시험이 아닙니다.

## 전체 조합 검증 감사

- 요청 조합: **{audit.get('requested_combinations', 23)}개**, 실제 로컬 실행 증거 보유: **{audit.get('local_execution_evidence_present', 0)}개**, 누락: **{audit.get('local_execution_evidence_missing', 0)}개**
- 조합 실행 상태: measured **{audit.get('combination_status_measured', 0)}개**, not_run **{audit.get('combination_status_not_run', 0)}개**
- 표준 배포 게이트: pass **{audit.get('standard_gate_pass', 0)}개**, incomplete **{audit.get('standard_gate_incomplete', 0)}개**
- 브라우저 WebGPU 직접 실행은 Qwen3.5-4B·Qwen3-8B cold/warm smoke입니다. Qwen3.5-9B는 4.8GB 다운로드까지 완료했지만 WebGPU 커널 컴파일 단계에서 renderer가 응답하지 않아 추론 미측정으로 기록했습니다. Android는 Qwen3.5-4B·Ternary Bonsai 8B·EXAONE Deep 7.8B 에뮬레이터 smoke입니다. 나머지 조합은 해당 플랫폼에서 실행됐다고 간주하지 않습니다.

## 그래프

![LLM 단독 속도](single-throughput.svg)

![용량-속도 trade-off](single-size-speed.svg)

![통합 VLM 속도](vlm-throughput.svg)

![분리형 조합](pair-throughput.svg)

![특화 전환 조합](transition-throughput.svg)

![플랫폼 smoke](platform-throughput.svg)

## 해석 가이드

- `tok/s`는 생성 단계 속도이며 prompt 처리속도나 첫 토큰 지연과 동일하지 않습니다.
- P 조합의 VLM/LLM 속도는 서로 다른 모델 단계의 값입니다. 합산해 단일 모델 속도로 비교하면 안 되고, end-to-end 지연은 `wall_seconds`와 VLM 생성 시간을 함께 봐야 합니다.
- Android 값은 Pixel 9 API 35 에뮬레이터 CLI smoke입니다. 실제 APK UI·물리기기·20분 열 안정성 결과가 아닙니다.
- `Browser-Q35`는 Headless Chrome UI에서 Apple Silicon의 localhost `llama-server`로 보낸 텍스트 경로입니다. `Browser-WebGPU-Q35`는 WebLLM 모델을 브라우저 캐시에 받고 WebGPU에서 직접 생성한 결과입니다.
- 브라우저 WebGPU 직접 측정은 Qwen3.5-4B cold load 1회(238.433초), decode 16.51 tok/s, TTFT 2.854초, Qwen3-8B cold load 1회({browser_qwen3['load_seconds']:.3f}초), decode {browser_qwen3['initial_decode_tps']:.2f} tok/s, TTFT {browser_qwen3['text_ttft_p50_seconds']:.3f}초입니다. Qwen3.5-9B는 파라미터 다운로드 {browser_qwen35_9b['metadata']['download_seconds']}초 후 컴파일에서 중단됐습니다. 모델 파일 개별 SHA-256과 장시간 열 안정성은 아직 검증하지 않았습니다.
- 동일 브라우저 캐시를 재사용한 warm load는 6.647초, decode 21.35 tok/s, TTFT 0.833초였습니다. 이 warm 값은 별도 1회 측정이라 반복 중앙값으로 해석하면 안 됩니다.
- Qwen3-8B warm load는 {browser_qwen3['warm_load_seconds']:.3f}초, decode {browser_qwen3['warm_decode_tps']:.2f} tok/s, TTFT {browser_qwen3['warm_text_ttft_seconds']:.3f}초였습니다.
- 기존 브라우저 측정은 성능 비교를 위해 `max_tokens=96`으로 실행했습니다. 이는 모델 한도가 아니라 테스트 페이지의 설정이며, 페이지 기본값은 학생 답변용으로 256으로 늘렸습니다.
- 품질·안전·한국어 자연스러움은 고정 task catalog에 대한 별도 채점이 필요합니다. 이 보고서는 실행 성능과 자원 비용만 정량화합니다.

정규화 데이터: [normalized_metrics.csv](normalized_metrics.csv)  
원본 coverage: [coverage.json](../../benchmarks/results/coverage.json)
"""
    (OUT / "benchmark-analysis.md").write_text(md, encoding="utf-8")

    def img(name: str, alt: str) -> str:
        return f'<figure><img src="{name}" alt="{safe(alt)}"><figcaption>{safe(alt)}</figcaption></figure>'

    table_rows = "\n".join(
        f"<tr><td>{safe(r['id'])}</td><td>{safe(r['name'])}</td><td>{safe(r['category'])}</td><td>{number(r['tps'])}</td><td>{number(r['secondary_tps'])}</td><td>{number(r['size_mb'], 1)}</td><td>{safe(r['gate'])}</td></tr>"
        for r in all_rows
    )
    html_doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>로컬 LLM/VLM 벤치마크 분석</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1180px;margin:0 auto;padding:32px;color:#172033;background:#f8fafc}}h1{{margin-bottom:4px}}.meta{{color:#64748b}}section{{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:22px;margin:20px 0;box-shadow:0 4px 14px #0f172a0d}}figure{{margin:22px 0}}figure img{{width:100%;height:auto;border:1px solid #e2e8f0;border-radius:10px;background:white}}figcaption{{color:#64748b;font-size:13px;margin-top:6px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9;position:sticky;top:0}}.scroll{{overflow:auto;max-height:620px}}.warn{{border-left:4px solid #f59e0b;padding-left:12px}}</style></head>
<body><h1>로컬 LLM/VLM 검증 결과 분석</h1><p class="meta">2026-08-07 · 23개 조합 measured · 속도·용량 중심</p>
<section><h2>핵심 수치</h2><ul><li>단일 LLM 최고 속도: <b>{safe(fastest['id'])} {safe(fastest['name'])} {fastest['tps']:.2f} tok/s</b></li><li>단일 모델 최소 용량: <b>{safe(smallest['id'])} {smallest['name']} {smallest['size_mb']:.1f} MB</b></li><li>통합 VLM 최고 생성 속도: <b>{safe(vlm_fastest['id'])} {safe(vlm_fastest['name'])} {vlm_fastest['tps']:.2f} tok/s</b></li><li>A.X VL Light CPU BF16: <b>{next(r for r in vlm if r['id']=='S2')['tps']:.4f} tok/s</b></li><li>브라우저 WebGPU 직접: <b>cold {browser_gpu['initial_decode_tps']:.2f} tok/s · warm {browser_gpu['warm_decode_tps']:.2f} tok/s · cold load {browser_gpu['load_seconds']:.3f}초</b></li></ul><p class="warn">모든 행은 최소 1회 실행 증거입니다. 3 cold load·20 requests·20분 thermal gate는 별도 판정이며 현재 완료된 행이 아닙니다.</p></section>
<section><h2>전체 조합 검증 감사</h2><ul><li>요청 조합 <b>{audit.get('requested_combinations', 23)}개</b> 중 로컬 실행 증거 <b>{audit.get('local_execution_evidence_present', 0)}개</b>, 누락 <b>{audit.get('local_execution_evidence_missing', 0)}개</b></li><li>표준 배포 게이트: pass <b>{audit.get('standard_gate_pass', 0)}개</b>, incomplete <b>{audit.get('standard_gate_incomplete', 0)}개</b></li><li>브라우저 WebGPU 직접: Qwen3.5-4B·Qwen3-8B cold/warm smoke; Qwen3.5-9B는 다운로드 후 컴파일 중단. Android: Qwen3.5-4B·Ternary Bonsai 8B·EXAONE Deep 7.8B 에뮬레이터 smoke.</li></ul><p class="warn">모든 조합이 Android 또는 브라우저 WebGPU에서 동작한다는 의미는 아닙니다. 플랫폼별 런타임·모델 아티팩트 가용성을 별도로 검증해야 합니다.</p></section>
<section><h2>속도 그래프</h2>{img('single-throughput.svg','LLM 단독 최종 생성 속도')}{img('single-size-speed.svg','LLM 단독 용량과 속도')}{img('vlm-throughput.svg','통합 VLM 이미지 생성 속도')}{img('pair-throughput.svg','분리형 P 조합 VLM과 메인 LLM 속도')}{img('transition-throughput.svg','특화 전환 X 조합 일반과 전문 모델 속도')}{img('platform-throughput.svg','Android·브라우저 smoke 속도')}</section>
<section><h2>정규화 원자료</h2><div class="scroll"><table><thead><tr><th>ID</th><th>모델/조합</th><th>구분</th><th>주 속도 tok/s</th><th>보조 속도 tok/s</th><th>용량 MB</th><th>Gate</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
<section><h2>주의사항</h2><p>tok/s는 생성 속도이며 품질 점수가 아닙니다. P 조합은 VLM 단계를 먼저 수행하는 handoff 구조라 VLM 생성 시간이 전체 지연을 지배할 수 있습니다. Android는 에뮬레이터 CLI smoke입니다. 브라우저는 localhost server 경로와 WebGPU 직접 경로를 구분해 기록했습니다. 기존 브라우저 속도 측정은 max_tokens=96으로 실행했으며, 이는 모델 제한이 아니라 테스트 설정입니다. 현재 테스트 페이지 기본값은 256입니다. WebGPU 결과는 cold/warm 각 1회만 측정했으므로 학생 기기 배포 전 반복·열 안정성 시험이 필요합니다. 품질·안전·한국어 자연스러움은 task catalog 기반 별도 채점이 필요합니다.</p><p><a href="benchmark-analysis.md">Markdown 보고서</a> · <a href="normalized_metrics.csv">CSV 원자료</a> · <a href="../../benchmarks/results/coverage.json">coverage 원본</a></p></section>
</body></html>"""
    (OUT / "benchmark-analysis.html").write_text(html_doc, encoding="utf-8")
    print(json.dumps({"rows": len(all_rows), "output": str(OUT), "gate_incomplete": incomplete}, ensure_ascii=False))


if __name__ == "__main__":
    main()
