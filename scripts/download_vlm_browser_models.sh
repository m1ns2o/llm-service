#!/usr/bin/env bash
set -euo pipefail

destination="${1:-/mnt/c/llm-cache/browser-vlm}"
mkdir -p "$destination"

qwen36_venv="${QWEN36_VENV:-${HOME}/.venvs/qwen36}"
if [[ -f "$qwen36_venv/bin/activate" ]]; then
  # The existing ROCm work environment already contains huggingface_hub + hf-xet.
  source "$qwen36_venv/bin/activate"
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI is required (install huggingface_hub with hf-xet)." >&2
  exit 1
fi

hf download LiquidAI/LFM2.5-VL-1.6B-GGUF \
  LFM2.5-VL-1.6B-Q4_K_M.gguf \
  mmproj-LFM2.5-VL-1.6b-Q8_0.gguf \
  --revision 0df8719db7180cedababc2bc589abfe5e8ebcd1f \
  --local-dir "$destination"

hf download mradermacher/Qwen3.5-4B-GGUF \
  Qwen3.5-4B.Q4_K_M.gguf \
  Qwen3.5-4B.mmproj-Q8_0.gguf \
  --revision 1a5df2c0cba51dae8ac5888420360d8703707171 \
  --local-dir "$destination"

(
  cd "$destination"
  sha256sum -c <<'EOF'
aefc3c97c9eb30d9c0dd6af4c38250f5f5106b57c8cf92de7914c7d0a9c94da2  LFM2.5-VL-1.6B-Q4_K_M.gguf
2ce89e610c56f3198ece2b86cf61743a08b9307279c89125eb2412ebb908689d  mmproj-LFM2.5-VL-1.6b-Q8_0.gguf
51eafbc127f35598c8f1d2ec58b2520d6126c7d1195c4eca26832e63a2939d39  Qwen3.5-4B.Q4_K_M.gguf
40a4f07d7bbdbb43011d6cf35ef751e4b1829ff47ee8aa4964c6296f571725ad  Qwen3.5-4B.mmproj-Q8_0.gguf
EOF
)
