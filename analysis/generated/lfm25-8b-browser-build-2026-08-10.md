# LFM2.5-8B browser WebGPU build report

Date: 2026-08-10
Host: Chrome 151, Windows, AMD RDNA 3 WebGPU

## Outcome

The quality-preserving browser candidate is the official LFM2.5-8B-A1B Q4_K_M GGUF running through wllama/llama.cpp WebGPU. Its 5,155,564,768-byte source artifact was verified as SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0` before the run. With a 4,096-token context it completed all four Korean/English smoke prompts, kept the final answers separate from internal reasoning, and produced the expected 4.25-hour battery calculation.

The official Q4F16 ONNX model is not currently a usable WebGPU path. Both Transformers.js 4.0.0 and 4.2.0 load the model, but generation stops at the first QMoE layer because ONNX Runtime WebGPU rejects explicit QMoE zero points.

## Actual measurements

| Candidate | Size | Load | Median TTFT | Median decode | Result |
|---|---:|---:|---:|---:|---|
| LFM2.5 official GGUF Q4_K_M, context 4096 | 5.156 GB | 16.340 s | 1.968 s | 19.83 tok/s | 4/4 smoke pass; recommended LFM2.5 path |
| LFM2.5 symmetric Q4F16, shift-only | 4.924 GB external tensors | 5.569 s | 0.358 s | 60.97 tok/s | Runtime pass, Korean quality gate fail |
| LFM2.5 symmetric Q4F16, best-MSE | 4.924 GB external tensors | 5.624 s | 0.315 s | 61.03 tok/s | Runtime pass, reasoning and language gate fail |
| LFM2.5 official ONNX Q4F16 | 5.045 GB | 5.522 s to load | n/a | n/a | Generation blocked by WebGPU QMoE zero points |
| LFM2-8B official ONNX Q4F16 | 4.785 GB | 5.182 s | 0.282 s | 65.96 tok/s | Transformers.js 4.2.0 regression pass |
| Qwen3.5-4B MLC q4f16 | 1.544 GB | 6.156 s | 0.355 s | 38.115 tok/s | Size/latency baseline |

All decimal GB values are bytes divided by 1e9. Load times depend strongly on browser cache state and are not cross-device claims.

## Symmetric QMoE experiment

`scripts/symmetrize_qmoe_onnx.py` converts only the 44 asymmetric INT4 expert tensors to the implicit midpoint-8 format accepted by the WebGPU QMoE kernel. It removes 44 zero-point initializers, preserves every non-QMoE external tensor byte-for-byte, rewrites ten shards, records SHA-256 hashes, and leaves tokenizer/config metadata outside the transform.

The best-MSE strategy reduced aggregate relative QMoE RMSE from 9.83% in the first transform to 6.56%, but the four-prompt gate still regressed: the linear battery problem was interpreted as exponential decay and one Korean response switched to English. Shift-only raised relative RMSE to 7.67% but preserved most representable source values exactly; it recovered the 4.25-hour calculation while still failing Korean language/script purity. This demonstrates why tensor RMSE alone is insufficient for model promotion.

The symmetric build remains explicitly experimental. It is useful for measuring the potential 61 tok/s ONNX WebGPU path, but it is not the quality-first model.

## Browser fixes made during the run

- Upgraded Transformers.js to 4.2.0 and re-ran the LFM2-8B baseline.
- Raised the selectable generation limit from 512 to 2,048 tokens.
- Raised the wllama context from 1,024 to 4,096 so reasoning does not consume the final-answer budget.
- Separated `reasoning_content` from `content`; reasoning is neither rendered nor stored in benchmark answers.
- Added streamed load/generation status, local GGUF SHA verification in the automation runner, and an explicit warning for the unsupported ONNX path.

## Evidence

- `benchmarks/results/lfm25-8b-browser-webgpu-gguf-context4096.json`
- `benchmarks/results/lfm25-8b-browser-transformersjs-v4.2.0.json`
- `benchmarks/results/lfm25-8b-browser-transformersjs-symmetric-q4f16-quality.json`
- `benchmarks/results/lfm25-8b-browser-transformersjs-symmetric-v2-quality.json`
- `benchmarks/results/lfm25-8b-browser-transformersjs-symmetric-shift-quality.json`
- `benchmarks/results/lfm2-8b-browser-transformersjs-v4.2.0-regression.json`
- `benchmarks/results/qwen35-4b-browser-webgpu-cached.json`
- `benchmarks/results/lfm25-8b-browser-build-summary.json`

This is a four-prompt regression smoke, not a broad benchmark. A release decision still requires a larger held-out Korean/English evaluation and mobile-device memory/thermal testing.

## Reproduction

Model weights are intentionally excluded from Git. Install the host-side tools with `python -m pip install -r requirements-browser.txt`, start `python scripts/serve_browser.py --model-cache C:\\llm-cache\\downloads`, and open `http://127.0.0.1:8000/browser-model-compare.html?model=lfm25-8b`. Select the verified official GGUF in the file input before loading. `scripts/run_browser_model_compare.py --help` exposes the same path for automated Chrome evidence capture.
