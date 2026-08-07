#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${WLLAMA_BUILD_ROOT:-${ROOT_DIR}/build/wllama-webgpu}"
SOURCE_DIR="${BUILD_ROOT}/source"
OUTPUT_DIR="${ROOT_DIR}/browser/vendor/wllama"
WLLAMA_REF="${WLLAMA_REF:-766d28e03eeac044fe055327d06b83d3f9b84544}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-dd4623a74f0c85e6b1dd9ee99a92b9c67cac3708}"
BUILD_JOBS="${WLLAMA_BUILD_JOBS:-2}"

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not running" >&2; exit 3; }
command -v npm >/dev/null || { echo "npm is required" >&2; exit 4; }

mkdir -p "${BUILD_ROOT}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --recurse-submodules https://github.com/ngxson/wllama.git "${SOURCE_DIR}"
fi

actual_ref="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_ref}" != "${WLLAMA_REF}" ]]; then
  echo "Unexpected wllama checkout: ${actual_ref}; expected ${WLLAMA_REF}" >&2
  exit 5
fi

git -C "${SOURCE_DIR}" submodule update --init --depth 1
actual_llama_ref="$(git -C "${SOURCE_DIR}/llama.cpp" rev-parse HEAD)"
if [[ "${actual_llama_ref}" != "${LLAMA_CPP_REF}" ]]; then
  echo "Unexpected llama.cpp checkout: ${actual_llama_ref}; expected ${LLAMA_CPP_REF}" >&2
  exit 6
fi

# Upstream currently invokes make with an unbounded -j. Keep this bounded on
# developer laptops; the full llama.cpp model registry otherwise exhausts the
# Docker VM during the WASM build.
compose_file="${SOURCE_DIR}/scripts/docker-compose.yml"
perl -0pi -e "s/emmake make wllama -j(?:[0-9]+)?/emmake make wllama -j${BUILD_JOBS}/" "${compose_file}"

(
  cd "${SOURCE_DIR}"
  npm ci
  SKIP_COMPAT=1 npm run build:wasm
  npm run build
)

mkdir -p "${OUTPUT_DIR}/wasm"
install -m 0644 "${SOURCE_DIR}/esm/index.min.js" "${OUTPUT_DIR}/index.js"
install -m 0644 "${SOURCE_DIR}/esm/wasm/wllama.wasm" "${OUTPUT_DIR}/wasm/wllama.wasm"
install -m 0644 "${SOURCE_DIR}/LICENCE" "${OUTPUT_DIR}/LICENSE"

echo "Built wllama ${WLLAMA_REF} with llama.cpp ${LLAMA_CPP_REF}"
shasum -a 256 "${OUTPUT_DIR}/index.js" "${OUTPUT_DIR}/wasm/wllama.wasm"
