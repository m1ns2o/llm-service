# Student Local LLM Bench

초·중학생용 로컬 LLM/VLM 후보를 Windows 브라우저와 Android에서 별도로 검증하기 위한 실행 기반입니다. 이 저장소에는 모델 가중치를 포함하지 않습니다. 가중치는 각 장비에서 명시한 upstream 리비전과 양자화 아티팩트로 받아 SHA-256을 기록해야 합니다.

## 현재 구현

- `configs/models.json`: S/L/P/X 23개 조합과 13개 정규화 모델의 체크포인트·라이선스·런타임·플랫폼 상태. Qwen3.5-9B는 최신 추가 후보이며 기존 23개 조합에는 포함하지 않고 브라우저 WebGPU를 별도 검증합니다.
- `benchmarks/task_catalog.json`: 자동 평가 문항 스키마와 초·중학생용 대표 문항
- `llm_bench.gates`: 다운로드 검증, 콜드 로드, 20회 요청, OOM/충돌, 20분 열 안정성, TTFT/TPS 하드 게이트
- `llm_bench.scoring`: 품질·안전·속도·지연·용량·메모리의 비지배 Pareto 후보 계산
- `examples/evidence/`: 장비 실행 결과를 채우는 JSON 예시

### 플랫폼별 llama.cpp 실험 경로

이번 확장에는 `qwen36-14b-a3b-fablevibes-q4km`(Q4_K_M)와
`qwen36-28b-a3b-reap20-iq3xxs`(IQ3_XXS)를 추가했습니다. 파일명·크기·SHA-256·출처·파생 관계는
`configs/models.json`에 있으며, Android/WebGPU 정책과 고정 비식별 프롬프트는 `configs/runtime.json`에 있습니다.
28B는 브라우저 목록에서 제외하고, 14B 브라우저 경로는 `browser/llama-webgpu-experimental.html`의
실험 경로로만 취급합니다. 기존 `browser/llm-webgpu-test.html`의 WebLLM/MLC 기준선은 유지합니다.

Android 앱은 `android/`에 있습니다. 모델 가중치는 APK에 포함하지 않으며, 앱은 다운로드 재개·용량/RAM 사전 검사·SHA-256 검증을 제공합니다. JNI는 실제 llama.cpp 모델 로딩과 생성을 수행하며 Qwen3.5 0.8B/2B/4B Q4_K_M 중 2B가 기본입니다. 빌드는 arm64-v8a 전용이고 Snapdragon/Adreno에서는 최적화 OpenCL, 그 외 지원 GPU에서는 Vulkan, 마지막으로 KleidiAI ARM CPU 경로를 선택합니다. NDK의 16KB 비호환 `libomp.so`는 포함하지 않고 ggml ARM 스레드 풀과 Q4 런타임 재패킹, 16KB ELF 정렬을 사용합니다.

```powershell
.\scripts\prepare_android_runtime.ps1
.\scripts\build_android.ps1
```

Windows 빌드 스크립트는 저장소 전용 JDK/SDK 경로와 Visual Studio C++ 호스트 환경을 설정한 뒤 Nuxt 정적 UI, Vulkan 셰이더, ARM64 JNI 런타임을 함께 빌드합니다. Vulkan 1.1 코어 심볼을 사용하는 네이티브 경로이므로 최소 Android API는 28입니다. 결과 APK는 `android/app/build/outputs/apk/debug/app-debug.apk`에 생성됩니다.

Qwen3.6 FableVibes를 WebLLM/MLC로 변환하는 작업은 `qwen3_5_moe_text` 모델 타입이 현재 MLC에 없어
`artifacts/compat-runtime/qwen36-14b-a3b-fablevibes-webllm/`에 `compile_blocked` 로그를 남겼습니다.
MLC PR #3449는 Qwen3.5 dense GatedDeltaNet 지원이며 이 MoE checkpoint에 직접 적용할 수 없습니다.

브라우저 포팅은 wllama 3.5.1과 Qwen3.5/3.6 MoE 로더가 들어 있는 llama.cpp를 고정 커밋으로 빌드합니다.
Docker 기반 재현 빌드는 `scripts/build_wllama_webgpu.sh`, COOP/COEP가 적용된 로컬 실행은
`python3 scripts/serve_browser.py`를 사용합니다. 산출물과 소스 커밋·SHA-256은
`browser/vendor/wllama/build-metadata.json`에 보존합니다.

14B 단일 8.46GB GGUF를 브라우저 ArrayBuffer로 받지 않습니다. 완성 원본의 크기와 SHA-256을 먼저 검증한 뒤
`scripts/prepare_browser_gguf.py <GGUF> <output-dir> --base-url <URL>`로 512MB 이하 shard와 해시 매니페스트를
생성합니다. 현재 원본 다운로드가 완성되지 않아 shard 매니페스트는 `pending_split`이며, 이 상태에서는
14B 실행을 `runtime_blocked`로 기록합니다. 런타임 자체는 페이지의 소형 모델 smoke로 별도 검증할 수 있습니다.
브라우저가 보고하는 총 RAM은 정보로만 기록하고 여유 RAM으로 사전 차단하지 않습니다. WebGPU가 없으면
14B CPU fallback은 WASM 선형 메모리 한계 때문에 `memory_blocked`입니다. WebGPU/Memory64/JSPI/격리 조건을
만족하지 않으면 속도 수치를 만들지 않고 `compile_blocked`, `memory_blocked`, `runtime_blocked` 중 하나를
JSON에 기록합니다.

## 로컬 검증

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m llm_bench validate-manifest
PYTHONPATH=src python -m llm_bench validate-runtime
PYTHONPATH=src python -m llm_bench gate examples/evidence/qwen3-8b-browser.json
```

`gate`는 미측정 결과를 `incomplete`로 처리합니다. 예시 파일은 의도적으로 통과하지 않습니다.

### 다른 환경에서 재현

Git 저장소에는 소스, 고정 런타임 설정, 브라우저용 wllama JS/WASM, 작은 벤치마크 결과를 포함합니다.
GGUF·safetensors 가중치, Android SDK 경로, Gradle/CMake 캐시와 빌드 결과는 포함하지 않습니다.

```bash
git clone <repository-url> llm-service
cd llm-service
./scripts/verify_portable_checkout.sh
python3 scripts/serve_browser.py
```

Chrome에서 `http://127.0.0.1:8000/llama-webgpu-experimental.html`을 열고 먼저
`WASM/WebGPU smoke`를 실행합니다. 14B는 `browser/model-shards/qwen36-14b-a3b-fablevibes-q4km.json`이
`ready`인 checkout에서만 본체 다운로드·실행이 활성화됩니다. 브라우저 런타임을 다시 만들려면 Docker를 실행한 뒤
`scripts/build_wllama_webgpu.sh`를 사용합니다.

Android debug 빌드는 JDK 17과 Android SDK 35, NDK `26.1.10909125`, CMake `3.22.1`, Visual Studio C++ 빌드 도구를 설치한 환경에서 다음처럼 검증합니다.

```powershell
.\scripts\prepare_android_runtime.ps1
.\scripts\build_android.ps1
# 또는 저장소 루트에서 Android까지 포함한 전체 검증
$env:VERIFY_ANDROID=1; bash ./scripts/verify_portable_checkout.sh
```

## 실제 로컬 실행 기록 (2026-08-07)

현재 MacBook Pro M1 Pro(16GB)에서 실제 모델 아티팩트를 내려받아 실행했습니다. 23개 조합 모두 최소 1회 로컬 실행 증거가 있지만, 모든 조합이 표준 하드 게이트를 통과했다는 뜻은 아닙니다.

전체 조합 감사 결과: S/L/P/X **23/23개 실행 증거 보유**, 누락 0개, JSON 원본 23개 유효. 표준 배포 게이트(3회 cold load·20회 요청·20분 열 안정성)는 **0/23 통과, 23/23 incomplete**입니다. 브라우저 WebGPU 직접 실행은 Qwen3.5-4B·Qwen3-8B cold/warm smoke, Android 에뮬레이터 smoke는 Qwen3.5-4B·Ternary Bonsai 8B·EXAONE Deep 7.8B에 한정됩니다.

A.X 4.0 VL Light는 공식 4-shard Transformers 체크포인트를 직접 다운로드한 뒤 S2 텍스트·이미지 smoke를 실행했습니다. 같은 VLM의 시각 사실을 A.X Light와 Qwen3-8B에 각각 handoff해 P2/P8도 실행했습니다. 원본은 `benchmarks/results/S2-ax-vl-light-transformers.json`, P2/P8은 해당 조합 JSON에서 확인할 수 있습니다. 이 세 결과는 CPU BF16 VLM과 고정 합성 이미지 1회 handoff라 표준 3 cold-load/20 요청/20분 게이트는 아직 미완료입니다.

| 실행 | 결과 |
| --- | --- |
| Qwen3.5-4B Q4_K_M 단독 | 20/20 요청, decode 32.00 → 31.49 tok/s, TTFT p50 78 ms |
| A.X 4.0 VL Light BF16 원본 단독(S2) | 공식 4-shard 로드, 텍스트·합성 이미지 응답 생성, CPU BF16 약 0.05 tok/s |
| Qwen3-8B Q4_K_M 단독 | 20/20 요청, decode 21.99 → 23.15 tok/s, TTFT p50 45 ms |
| A.X 4.0 Light IQ1_M | 20/20 요청, decode 22.93 → 22.24 tok/s, TTFT p50 167 ms |
| Kanana 1.5 8B Q2_K | 20/20 요청, decode 25.89 → 25.02 tok/s, TTFT p50 41 ms |
| EXAONE 3.5 7.8B Q2_K | 20/20 요청, decode 25.63 → 25.63 tok/s, TTFT p50 42 ms |
| EXAONE Deep 7.8B Q2_K | 20/20 요청, decode 25.73 → 25.83 tok/s, TTFT p50 40 ms |
| Ternary Bonsai 8B Q2_0 | Prism fork 필요. 20/20 요청, decode 30.67 → 30.76 tok/s, TTFT p50 34 ms |
| Qwen3-VL-4B + mmproj | 고정 합성 이미지에서 색상·사탕 인식 응답 확인 |
| Qwen3-VL-8B Q4_K_M + mmproj | 합성 기하 패턴 이미지 설명 성공, 59.4 prompt / 24.9 generation tok/s |
| Gemma 4 E4B IQ3_M + mmproj | 통합 이미지 입력 성공, 70.9 prompt / 31.1 generation tok/s |
| Ternary Bonsai 4B Q2_0 | Prism fork에서 20/20 요청, decode 46.10 → 50.67 tok/s, TTFT p50 21 ms |
| P3 Qwen3 + Qwen3-VL | VLM 시각 사실을 Qwen3에 handoff해 한국어 설명 생성 |
| P2 A.X Light + A.X VL Light | 공식 A.X VL 시각 사실을 A.X Light GGUF에 handoff, LLM 24.2 tok/s |
| P4 Kanana + Qwen3-VL | VLM 시각 사실을 Kanana에 handoff해 한국어 설명 생성 |
| P6 Ternary + Qwen3-VL | VLM 시각 사실을 Prism Ternary에 handoff해 한국어 설명 생성 |
| P8 Qwen3 + A.X VL Light | 공식 A.X VL 시각 사실을 Qwen3-8B에 handoff, LLM 10.57 tok/s |
| X3 Ternary → EXAONE Deep | 두 런타임을 순차 로드, 각 응답과 전환 wall time 기록 |
| Android Pixel_9_API_35 API 35 | Prism Android arm64 런타임으로 Qwen3.5-4B IQ2 텍스트 smoke 성공, 5.5 tok/s |
| Galaxy S25 SM-S931N / Adreno 830 | Qwen3.5-0.8B Q4_K_M 네이티브 OpenCL smoke 성공. warm 생성 17.39 tok/s, 동일 CPU 1.05 tok/s. Vulkan 1.3은 노출되지만 llama.cpp 셰이더 파이프라인 생성은 드라이버 오류로 실패 |
| Android Ternary Bonsai 8B | 로드·응답 성공, generation 0.2 tok/s; 에뮬레이터 성능상 실사용 부적합 |
| Android EXAONE Deep 7.8B | adb 전송·모델 로드 성공, 180초 내 완성 응답 없이 중단 |
| 로컬 브라우저 | Headless Chrome에서 `llama-ui`를 열고 localhost 서버로 한국어 문항을 전송·응답 확인, 172 tokens/5.5 s |
| 브라우저 WebGPU 직접 | WebLLM의 Qwen3.5-4B MLC를 브라우저 캐시에 다운로드 후 WebGPU에서 직접 생성, cold 16.51 tok/s·warm 21.35 tok/s |
| 브라우저 WebGPU 직접 | WebLLM의 Qwen3-8B MLC를 브라우저 캐시에 다운로드 후 WebGPU에서 직접 생성, cold 12.65 tok/s·warm 14.56 tok/s |
| 브라우저 WebGPU 직접 | Qwen3.5-9B MLC를 선택 가능한 모델 목록에 추가했으며, 실제 cold/warm 측정 결과는 `benchmarks/results/qwen35-9b-browser-webgpu.json`에 기록합니다 |
| 브라우저 llama.cpp WebGPU 실험 | Qwen3.6-14B-A3B용 wllama/llama.cpp WASM 빌드와 실제 WebGPU 소형 모델 생성 통과. 14B 원본 shard 미완성으로 `runtime_blocked`; 속도 수치 미보고 |

상세 원본과 23개 조합의 측정/미측정 범위는 `benchmarks/results/coverage.json`과 `benchmarks/results/`에 있습니다. Android는 에뮬레이터 기록 외에 Galaxy S25 APK/JNI 실기기 smoke를 완료했지만 20분 열 안정성은 아직 측정하지 않았습니다. EXAONE Deep Android는 모델 로드까지 확인했지만 180초 내 완성 응답이 없어 성능 성공으로 판정하지 않았습니다. 브라우저 결과는 `qwen35-4b-browser-local.json`(localhost 서버 경로)과 `qwen35-4b-browser-webgpu.json`(서버 없이 WebGPU 직접 경로)로 분리했습니다. 직접 경로의 상세 실행 페이지는 `browser/llm-webgpu-test.html`입니다.

Qwen3.5-4B의 GGUF는 공식 Qwen GGUF가 아니라 `unsloth/Qwen3.5-4B-GGUF` 커뮤니티 변환본을 사용했습니다. A.X, Kanana, EXAONE 계열도 GGUF 커뮤니티 변환본/공식 GGUF의 저비트 파일을 사용했습니다. 호스트 저장공간 제약 때문에 A.X·Kanana·EXAONE 3.5의 원본 GGUF는 해시와 결과 JSON을 보존한 뒤 순차 실험 후 제거했으며, 결과 재현에는 같은 upstream 파일을 다시 받아야 합니다. Ternary Bonsai는 일반 Homebrew `llama.cpp`에서 로드 오류가 발생해 Prism fork로만 측정했습니다.

### 1차 해석

- 저장공간·속도 우선이면 Ternary Bonsai 4B/8B가 가장 유리했지만, 4B는 저비트 품질 하한선으로 별도 취급해야 합니다.
- 통합 멀티모달 후보는 Qwen3.5-4B가 2.7GB 모델 파일과 약 32 tok/s를 보여 가장 현실적인 기준점이었습니다. Qwen3-VL-8B는 이미지 설명 품질 기준점이지만 모델+mmproj가 약 6.2GB입니다.
- 분리형 P 조합은 시각 사실을 텍스트로 넘기는 handoff가 작동했으나, 두 모델을 동시에 유지하면 P3 약 8.4GB, P5 약 6.4GB가 필요해 8GB급 모바일에서는 메모리 측정이 필수입니다.
- A.X VL Light 원본은 약 14.67GiB의 BF16 shard이며 M1 Pro CPU에서 약 0.05 tok/s로 생성되어, 현재 형태는 학생 로컬·모바일 배포 후보가 아니라 정확도 기준용입니다. P2는 약 17.0GB, P8은 약 20.4GB의 구성 용량을 필요로 해 8GB급 기기에는 부적합합니다.
- 같은 고정 시각 사실을 넘긴 handoff에서 A.X Light(P2)는 24.2 tok/s, Qwen3-8B(P8)는 10.57 tok/s를 보였지만, VLM CPU 생성 시간이 약 173초이므로 전체 응답 지연은 VLM 병목입니다.
- A.X·Kanana·EXAONE의 한국어 응답은 모두 생성됐지만, 이번 수치는 서로 다른 저비트 양자화와 짧은 고정 문항에 대한 실행 지표이지 품질 우열 점수가 아닙니다. 품질 점수는 `task_catalog.json`의 문항을 고정해 별도 사람/자동 채점해야 합니다.

Pareto 계산 입력은 다음 필드를 가진 JSON 배열입니다.

```json
[{"id":"qwen3-8b","quality":82,"safety":95,"decode_tps":10,
  "ttft_seconds":3,"size_mb":4600,"peak_rss_mb":6500,
  "platform":"android"}]
```

```bash
PYTHONPATH=src python -m llm_bench pareto metrics.json
```

## 실제 런타임 어댑터 연결 규칙

1. 모델 파일을 최초 Wi-Fi 다운로드하고, 다운로드 완료 후 SHA-256을 `RunEvidence.artifact_sha256`에 기록합니다.
2. Windows 브라우저는 Chrome/Edge WebGPU와 WebLLM/MLC를 사용합니다. 커스텀 WASM은 소스 리비전·컴파일 명령·아티팩트 URL을 함께 보존합니다.
3. Android는 일반 GGUF를 `llama.cpp` JNI, Gemma 4 E4B를 LiteRT-LM, Ternary Bonsai를 Prism fork로 실행합니다. 런타임이 다르면 결과에 반드시 기록합니다.
4. P 조합은 VLM이 `ocr_text`, `visual_facts`, `uncertainty`를 생성한 뒤 모델을 내리고 LLM을 로드합니다. 두 모델의 저장공간 합계와 전환 시간을 별도 측정합니다.
5. 학생의 실제 입력·사진·개인정보는 이 벤치마크에 넣지 않습니다. 고정된 비식별 문항과 합성·라이선스 확인 이미지만 사용합니다.

## 중요한 판정 제한

- 브라우저 텍스트 실행 성공은 브라우저 VLM 성공을 의미하지 않습니다.
- EXAONE 모델은 현재 매니페스트에서 연구·비상업 라이선스로 표시되어 있으며 상용 배포 후보가 아닙니다.
- Ternary Bonsai의 Q2_0은 일반 llama.cpp와 호환된다고 가정하지 않습니다.
- 자동 점수는 실제 학생의 이해도·학습 효과·사용성 검증이 아닙니다.

## 정량 분석자료

실행 결과를 모델 단독·통합 VLM·분리형 P 조합·플랫폼 smoke로 정규화한 그래프와 표는 [HTML 분석보고서](analysis/generated/benchmark-analysis.html)에서 확인할 수 있습니다. 인쇄·공유용 [Markdown 보고서](analysis/generated/benchmark-analysis.md), [CSV 원자료](analysis/generated/normalized_metrics.csv), [PNG 미리보기](analysis/generated/benchmark-analysis-preview.png)도 함께 생성했습니다. 재생성 명령은 `python3 analysis/generate_report.py`입니다.
