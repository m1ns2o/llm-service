# Local Qwen Chat

Qwen3.5-2B와 Qwen3.5-4B를 WebLLM WebGPU로 브라우저 안에서 직접 실행하는 Nuxt 채팅 서비스입니다. 대화와 추론은 별도 추론 서버로 전송되지 않습니다.

공식 Qwen3.6에는 현재 2B·4B 모델이 없습니다. 작은 공식 멀티모달 계열인 Qwen3.5-2B와 Qwen3.5-4B를 정확한 이름으로 제공합니다.

## 실행

```bash
npm install
npm run build
npm run preview -- --port=3001 --host=127.0.0.1
```

Chrome 또는 Edge에서 `http://127.0.0.1:3001`을 엽니다. 첫 실행에는 선택한 MLC 모델을 내려받으며 이후에는 브라우저 캐시를 재사용합니다. UI 개발에는 `npm run dev`를 사용할 수 있고, 실제 모델 검증에는 Worker가 빌드된 위 프로덕션 미리보기 명령을 사용합니다.

## 모델

| 서비스 모드 | WebLLM 모델 | 예상 VRAM | 용도 |
|---|---|---:|---|
| 빠른 기본 | `Qwen3.5-2B-q4f16_1-MLC` | 약 2.25GB | 일반 채팅, 저사양 내장 GPU |
| 품질 우선 | `Qwen3.5-4B-q4f16_1-MLC` | 약 3.87GB | 더 높은 답변 품질 |

모델을 바꾸면 현재 엔진을 언로드하고 Web Worker를 종료한 뒤 새 모델을 로드합니다. 두 모델이 GPU 메모리에 동시에 남지 않으며, 생성 중에는 모델 선택이 잠깁니다.

## 검증

```bash
npm run typecheck
npm run build
```

WebGPU가 없으면 입력창을 활성화하지 않고 지원 브라우저 안내를 표시합니다. 생성 결과는 스트리밍되며 로딩률, 경과 시간, 첫 토큰과 디코드 속도를 화면에 표시합니다.
