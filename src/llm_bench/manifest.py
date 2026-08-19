from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


PLATFORMS = {"browser-windows", "android"}
MODEL_KINDS = {"llm", "vlm"}
MODEL_BACKENDS = {
    "vulkan",
    "opencl",
    "opencl-experimental",
    "cpu-arm64",
    "webgpu-wllama",
    "wasm-cpu-size-blocked",
}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    kind: str
    upstream: str
    license: str
    artifact: str
    runtimes: tuple[str, ...]
    platform_status: dict[str, str]
    notes: str
    source_url: str
    artifacts: tuple[dict[str, Any], ...] = ()
    minimum_ram_gb: float | None = None
    minimum_storage_gb: float | None = None
    supported_platforms: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    validation_status: str | None = None
    derivation: dict[str, Any] | None = None
    browser_artifact_manifest: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelSpec":
        return cls(
            id=value["id"], name=value["name"], kind=value["kind"],
            upstream=value["upstream"], license=value["license"],
            artifact=value["artifact"], runtimes=tuple(value["runtimes"]),
            platform_status=dict(value["platform_status"]),
            notes=value["notes"], source_url=value["source_url"],
            artifacts=tuple(dict(item) for item in value.get("artifacts", [])),
            minimum_ram_gb=value.get("minimum_ram_gb"),
            minimum_storage_gb=value.get("minimum_storage_gb"),
            supported_platforms=tuple(value.get("supported_platforms", [])),
            backends=tuple(value.get("backends", [])),
            validation_status=value.get("validation_status"),
            derivation=dict(value["derivation"]) if value.get("derivation") else None,
            browser_artifact_manifest=value.get("browser_artifact_manifest"),
        )


def load_manifest(path: str | Path) -> tuple[dict[str, ModelSpec], dict[str, dict[str, Any]]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    models = [ModelSpec.from_dict(item) for item in raw["models"]]
    by_id = {model.id: model for model in models}
    if len(by_id) != len(models):
        raise ValueError("duplicate model id in manifest")
    validate_models(by_id)
    combinations = {item["id"]: item for item in raw["combinations"]}
    validate_combinations(combinations, by_id)
    return by_id, combinations


def validate_models(models: dict[str, ModelSpec]) -> None:
    for model in models.values():
        if model.kind not in MODEL_KINDS:
            raise ValueError(f"{model.id}: invalid kind {model.kind}")
        if not model.runtimes:
            raise ValueError(f"{model.id}: at least one runtime is required")
        if set(model.platform_status) != PLATFORMS:
            raise ValueError(f"{model.id}: platform_status must contain {sorted(PLATFORMS)}")
        if not model.source_url.startswith("https://"):
            raise ValueError(f"{model.id}: source_url must be HTTPS")
        if model.validation_status and model.validation_status not in {"experimental", "blocked", "passed"}:
            raise ValueError(f"{model.id}: invalid validation_status {model.validation_status}")
        if model.artifacts:
            for artifact in model.artifacts:
                required = {"filename", "url", "quantization", "size_bytes", "sha256"}
                missing = required - artifact.keys()
                if missing:
                    raise ValueError(f"{model.id}: artifact missing {sorted(missing)}")
                if not str(artifact["url"]).startswith("https://"):
                    raise ValueError(f"{model.id}: artifact URL must be HTTPS")
                if not isinstance(artifact["size_bytes"], int) or artifact["size_bytes"] <= 0:
                    raise ValueError(f"{model.id}: artifact size_bytes must be positive integer")
                if not re.fullmatch(r"[0-9a-fA-F]{64}", str(artifact["sha256"])):
                    raise ValueError(f"{model.id}: artifact sha256 must be 64 hex characters")
        if model.supported_platforms and not set(model.supported_platforms) <= PLATFORMS:
            raise ValueError(f"{model.id}: unsupported platform in supported_platforms")
        if model.backends and not set(model.backends) <= MODEL_BACKENDS:
            raise ValueError(f"{model.id}: unsupported backend")
        if model.minimum_ram_gb is not None and model.minimum_ram_gb <= 0:
            raise ValueError(f"{model.id}: minimum_ram_gb must be positive")
        if model.minimum_storage_gb is not None and model.minimum_storage_gb <= 0:
            raise ValueError(f"{model.id}: minimum_storage_gb must be positive")


def validate_combinations(combinations: dict[str, dict[str, Any]], models: dict[str, ModelSpec]) -> None:
    for combo_id, combo in combinations.items():
        if combo.get("kind") not in {"single", "pair"}:
            raise ValueError(f"{combo_id}: invalid combination kind")
        for key in ("model", "main_llm", "vlm", "general", "specialized"):
            if key in combo and combo[key] not in models:
                raise ValueError(f"{combo_id}: unknown model {combo[key]}")
        if combo["kind"] == "single" and "model" not in combo:
            raise ValueError(f"{combo_id}: single combination requires model")
        if combo["kind"] == "pair" and not ({"main_llm", "vlm"} <= combo.keys() or {"general", "specialized"} <= combo.keys()):
            raise ValueError(f"{combo_id}: pair requires main_llm/vlm or general/specialized")
        if {"main_llm", "vlm"} <= combo.keys():
            if models[combo["main_llm"]].kind != "llm":
                raise ValueError(f"{combo_id}: main_llm must reference an LLM")
            if models[combo["vlm"]].kind != "vlm":
                raise ValueError(f"{combo_id}: vlm must reference a VLM")
        if {"general", "specialized"} <= combo.keys():
            if models[combo["general"]].kind != "llm" or models[combo["specialized"]].kind != "llm":
                raise ValueError(f"{combo_id}: general/specialized must reference LLMs")
