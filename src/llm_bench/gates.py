from __future__ import annotations

from dataclasses import dataclass

from .schema import RunEvidence


@dataclass(frozen=True)
class GateResult:
    status: str  # pass, fail, incomplete
    reasons: tuple[str, ...]


def evaluate_gate(evidence: RunEvidence) -> GateResult:
    failures: list[str] = []
    missing: list[str] = []

    if evidence.artifact_sha256 is None:
        missing.append("artifact_sha256")
    elif not evidence.download_verified:
        failures.append("download_sha256_not_verified")
    if evidence.cold_loads < 3:
        missing.append("cold_loads>=3")
    if evidence.requests_completed < 20:
        missing.append("requests_completed>=20")
    if evidence.crashes:
        failures.append("crashes_detected")
    if evidence.oom_events:
        failures.append("oom_detected")
    if evidence.thermal_minutes < 20:
        missing.append("thermal_minutes>=20")

    required_metrics = {
        "initial_decode_tps": evidence.initial_decode_tps,
        "final_decode_tps": evidence.final_decode_tps,
        "text_ttft_p50_seconds": evidence.text_ttft_p50_seconds,
    }
    missing.extend(name for name, value in required_metrics.items() if value is None)
    if evidence.initial_decode_tps is not None and evidence.final_decode_tps is not None:
        if evidence.final_decode_tps < evidence.initial_decode_tps * 0.70:
            failures.append("thermal_decode_drop_over_30_percent")
    if evidence.text_ttft_p50_seconds is not None and evidence.text_ttft_p50_seconds > 5:
        failures.append("text_ttft_over_5_seconds")
    if evidence.vision_ttft_p50_seconds is not None and evidence.vision_ttft_p50_seconds > 15:
        failures.append("vision_ttft_over_15_seconds")

    if failures:
        return GateResult("fail", tuple(failures + missing))
    if missing:
        return GateResult("incomplete", tuple(missing))
    return GateResult("pass", ())
