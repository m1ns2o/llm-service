from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunEvidence:
    """Evidence collected from one model/platform/quantization run.

    Missing values are deliberate: a model must not pass a gate merely because
    an unmeasured field was omitted.
    """

    model_id: str
    platform: str
    quantization: str
    artifact_sha256: str | None = None
    download_verified: bool = False
    download_mb: float | None = None
    download_seconds: float | None = None
    cold_loads: int = 0
    requests_completed: int = 0
    crashes: int = 0
    oom_events: int = 0
    thermal_minutes: float = 0.0
    initial_decode_tps: float | None = None
    final_decode_tps: float | None = None
    text_ttft_p50_seconds: float | None = None
    vision_ttft_p50_seconds: float | None = None
    peak_rss_mb: float | None = None
    model_size_mb: float | None = None
    quality_score: float | None = None
    safety_score: float | None = None
    # Cross-platform runtime provenance and metric aliases used by the Android
    # and llama.cpp WebGPU harnesses. The original fields remain supported.
    artifact_url: str | None = None
    artifact_filename: str | None = None
    runtime: str | None = None
    backend: str | None = None
    model_load_ms: float | None = None
    ttft_ms: float | None = None
    prompt_tok_per_s: float | None = None
    decode_tok_per_s: float | None = None
    thermal_status: str | None = None
    device: str | None = None
    soc: str | None = None
    android_api: int | None = None
    abi: str | None = None
    status: str | None = None
    blocked_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RunEvidence":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"unknown run evidence fields: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
