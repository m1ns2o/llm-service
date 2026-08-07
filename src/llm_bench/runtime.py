from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BACKENDS = {"vulkan", "opencl-experimental", "cpu-arm64"}
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
    if priority != ["vulkan", "opencl-experimental", "cpu-arm64"]:
        raise ValueError("android backend priority must be Vulkan, OpenCL, ARM CPU")
    if set(priority) - BACKENDS:
        raise ValueError("android backend priority contains an unknown backend")
    if browser.get("wasm_cpu_fallback") is not True or browser.get("webgpu_priority") is not True:
        raise ValueError("browser must enable WebGPU priority and WASM CPU fallback")
    if set(browser.get("blocked_statuses", [])) != BLOCKED_STATUSES:
        raise ValueError("browser blocked statuses are incomplete")
    if browser.get("model_weights_bundled") is not False or android.get("model_weights_bundled") is not False:
        raise ValueError("model weights must not be bundled")
