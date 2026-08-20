from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BACKENDS = {"opencl", "vulkan", "cpu-arm64"}
BLOCKED_STATUSES = {"compile_blocked", "memory_blocked", "runtime_blocked"}


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_runtime_config(config)
    return config


def validate_runtime_config(config: dict[str, Any]) -> None:
    android = config.get("android")
    browser = config.get("browser")
    benchmark = config.get("fixed_benchmark")
    if not isinstance(android, dict) or not isinstance(browser, dict):
        raise ValueError("runtime config requires android and browser sections")
    if not isinstance(benchmark, dict) or benchmark.get("student_input_allowed") is not False:
        raise ValueError("fixed_benchmark must explicitly disallow student input")
    priority = android.get("backend_priority")
    if priority != ["vendor-adaptive", "cpu-arm64"]:
        raise ValueError("android backend priority must be vendor-adaptive with ARM CPU fallback")
    vendor_priority = android.get("backend_priority_by_gpu_vendor")
    expected_vendor_priority = {
        "qualcomm-adreno": ["opencl", "vulkan", "cpu-arm64"],
        "default": ["vulkan", "opencl", "cpu-arm64"],
    }
    if vendor_priority != expected_vendor_priority:
        raise ValueError("android vendor backend priorities are invalid")
    if any(set(backends) - BACKENDS for backends in vendor_priority.values()):
        raise ValueError("android vendor backend priority contains an unknown backend")
    if browser.get("wasm_cpu_fallback") is not True or browser.get("webgpu_priority") is not True:
        raise ValueError("browser must enable WebGPU priority and WASM CPU fallback")
    if set(browser.get("blocked_statuses", [])) != BLOCKED_STATUSES:
        raise ValueError("browser blocked statuses are incomplete")
    if browser.get("model_weights_bundled") is not False or android.get("model_weights_bundled") is not False:
        raise ValueError("model weights must not be bundled")
