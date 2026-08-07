#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python3 -m json.tool configs/models.json >/dev/null
python3 -m json.tool configs/runtime.json >/dev/null
python3 -m json.tool browser/vendor/wllama/build-metadata.json >/dev/null
python3 -m json.tool browser/model-shards/qwen36-14b-a3b-fablevibes-q4km.json >/dev/null
python3 -m json.tool benchmarks/results/qwen36-14b-a3b-browser-webgpu-port.json >/dev/null

PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m llm_bench validate-manifest
PYTHONPATH=src python3 -m llm_bench validate-runtime

if command -v node >/dev/null; then
  node --check browser/llama-webgpu-experimental.js
else
  echo "node not found: skipped JavaScript syntax check" >&2
fi

if [[ "${VERIFY_ANDROID:-0}" == "1" ]]; then
  (
    cd android
    ./gradlew --no-daemon assembleDebug
  )

  apk_path="${ROOT_DIR}/android/app/build/outputs/apk/debug/app-debug.apk"
  android_sdk_dir="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
  if [[ -n "${android_sdk_dir}" ]]; then
    zipalign_bin="$(find "${android_sdk_dir}/build-tools" -type f -name zipalign -perm -111 2>/dev/null | sort -V | tail -1)"
    if [[ -n "${zipalign_bin}" ]]; then
      "${zipalign_bin}" -c -P 16 -v 4 "${apk_path}" >/dev/null
      echo "APK 16 KB ZIP alignment passed"
    fi

    readelf_bin="$(find "${android_sdk_dir}/ndk/26.1.10909125" -name llvm-readelf 2>/dev/null | head -1)"
    if [[ -n "${readelf_bin}" ]] && command -v unzip >/dev/null; then
      elf_tmp="$(mktemp -d)"
      trap 'rm -r "${elf_tmp}"' EXIT
      unzip -q "${apk_path}" lib/arm64-v8a/libllama_runtime.so -d "${elf_tmp}"
      "${readelf_bin}" -l "${elf_tmp}/lib/arm64-v8a/libllama_runtime.so" \
        | awk '$1 == "LOAD" && $NF != "0x4000" { bad = 1 } END { exit bad }'
      echo "Android ELF LOAD alignment 0x4000 passed"
    fi
  else
    echo "ANDROID_SDK_ROOT/ANDROID_HOME not set: skipped explicit 16 KB APK/ELF checks" >&2
  fi
fi

echo "Portable checkout verification passed"
