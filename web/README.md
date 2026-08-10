# Local AI Chat

Nuxt UI에서 LFM2.5-8B-A1B는 wllama WebGPU로, Qwen3.5-4B는 WebLLM WebGPU로 직접 실행하는 로컬 채팅 앱입니다.

```bash
npm install
npm run dev
```

LFM을 사용하려면 화면에서 공식 `LFM2.5-8B-A1B-Q4_K_M.gguf` 파일을 선택하세요. 검증된 파일 크기는 `5,155,564,768` bytes, SHA-256은 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`입니다. 현재 개발 장비에서는 `C:\llm-cache\downloads\lfm25-8b\` 아래에 있습니다.

Qwen을 선택하면 최초 1회 약 2.3GB를 받고 이후 브라우저 캐시를 사용합니다. 모델 파일과 대화 내용은 서버로 전송되지 않습니다.
