#!/usr/bin/env bash
set -euo pipefail

# Reproducible native build entry point. An immutable ref is required so a
# benchmark result can always identify the exact llama.cpp source used.
: "${LLAMA_CPP_REF:?Set LLAMA_CPP_REF to an immutable llama.cpp commit or tag}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-${LLAMA_RUNTIME_TARGET:-}}"
SOURCE_DIR="${LLAMA_CPP_SOURCE_DIR:-${ROOT_DIR}/third_party/llama.cpp}"
BUILD_ROOT="${LLAMA_BUILD_DIR:-${ROOT_DIR}/build/llama-cpp}"
JOBS="${LLAMA_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN)}"
BUILD_TARGETS="${LLAMA_BUILD_TARGETS:-}"

if [[ "${TARGET}" != "android" && "${TARGET}" != "browser" ]]; then
  echo "usage: LLAMA_CPP_REF=<commit-or-tag> $0 android|browser" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SOURCE_DIR}")"
  git clone https://github.com/ggml-org/llama.cpp.git "${SOURCE_DIR}"
fi
git -C "${SOURCE_DIR}" fetch --quiet --tags origin "${LLAMA_CPP_REF}"
git -C "${SOURCE_DIR}" checkout --quiet --detach "${LLAMA_CPP_REF}"
RESOLVED_COMMIT="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"

if [[ "${TARGET}" == "android" ]]; then
  : "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to the Android NDK directory}"
  CMAKE_ARGS=(
    -DANDROID_ABI=arm64-v8a
    -DANDROID_PLATFORM=android-26
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake"
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384"
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-z,max-page-size=16384"
    -DGGML_VULKAN=ON
    -DGGML_OPENCL=ON
    -DGGML_NATIVE=OFF
  )
  CMAKE_CMD=(cmake)
else
  : "${EMDAWNWEBGPU_DIR:?Set EMDAWNWEBGPU_DIR to the Dawn emdawnwebgpu package}"
  command -v emcmake >/dev/null || { echo "emcmake (Emscripten) is required" >&2; exit 3; }
  CMAKE_ARGS=(
    -DGGML_WEBGPU=ON
    -DGGML_WASM_SIMD=ON
    -DGGML_NATIVE=OFF
    -DGGML_OPENMP=OFF
    -DEMDAWNWEBGPU_DIR="${EMDAWNWEBGPU_DIR}"
  )
  CMAKE_CMD=(emcmake cmake)
fi

BUILD_DIR="${BUILD_ROOT}/${TARGET}"
mkdir -p "${BUILD_DIR}"
"${CMAKE_CMD[@]}" -S "${SOURCE_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARGS[@]}" ${LLAMA_CMAKE_EXTRA_FLAGS:-}
if [[ -n "${BUILD_TARGETS}" ]]; then
  read -r -a TARGET_ARGS <<< "${BUILD_TARGETS}"
  cmake --build "${BUILD_DIR}" --config Release --parallel "${JOBS}" --target "${TARGET_ARGS[@]}"
else
  cmake --build "${BUILD_DIR}" --config Release --parallel "${JOBS}"
fi

python3 - "${BUILD_DIR}/build-metadata.json" "${TARGET}" "${RESOLVED_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "status": "built",
    "target": sys.argv[2],
    "llama_cpp_commit": sys.argv[3],
    "cmake_flags": "recorded in the invoking shell and CMakeCache.txt",
}, indent=2) + "\n", encoding="utf-8")
PY
echo "built ${TARGET} with llama.cpp ${RESOLVED_COMMIT}"
