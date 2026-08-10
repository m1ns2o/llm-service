import type { Wllama, WllamaChatMessage } from '@wllama/wllama/esm/index.js'
import wllamaWasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url'

export const LFM_MODEL_LABEL = 'LFM2.5-8B-A1B Q4_K_M'
export const LFM_MODEL_SIZE_BYTES = 5_155_564_768
export const LFM_MODEL_SHA256 = '4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0'
export const LFM_CONTEXT_WINDOW_SIZE = 4096

type ModelState = 'idle' | 'checking' | 'loading' | 'ready' | 'error'

export interface LfmGenerationMetrics {
  completionTokens?: number
  decodeTokensPerSecond?: number
  timeToFirstTokenSeconds?: number
  timeToFirstVisibleTokenSeconds?: number
  reasoningChunks: number
}

let runtime: Wllama | null = null
let runtimePromise: Promise<Wllama> | null = null
let selectedFile: File | null = null
let abortController: AbortController | null = null
let interrupted = false

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    void runtime?.exit()
    runtime = null
    runtimePromise = null
    selectedFile = null
  })
}

export function useLocalLfm() {
  const modelState = useState<ModelState>('lfm-model-state', () => 'idle')
  const modelProgress = useState<number>('lfm-model-progress', () => 0)
  const modelStatusText = useState<string>('lfm-model-status-text', () => '공식 GGUF 파일을 선택해 주세요')
  const modelError = useState<string | null>('lfm-model-error', () => null)
  const webgpuAvailable = useState<boolean | null>('lfm-webgpu', () => null)
  const modelFileName = useState<string | null>('lfm-model-file-name', () => null)

  async function checkWebGPU() {
    if (!import.meta.client) return false

    modelState.value = 'checking'
    const gpu = navigator.gpu
    const adapter = gpu ? await gpu.requestAdapter({ powerPreference: 'high-performance' }) : null
    webgpuAvailable.value = Boolean(adapter)

    if (!adapter) {
      modelState.value = 'error'
      modelError.value = 'WebGPU를 사용할 수 없습니다. 최신 Chrome 또는 Edge가 필요합니다.'
      modelStatusText.value = 'WebGPU를 찾지 못했습니다'
      return false
    }

    modelState.value = runtime ? 'ready' : 'idle'
    modelStatusText.value = runtime ? `${LFM_MODEL_LABEL} 준비됨` : selectedFile ? 'WebGPU 확인됨 · 모델을 불러올 수 있습니다' : '공식 GGUF 파일을 선택해 주세요'
    return true
  }

  function selectModelFile(file: File | null) {
    if (runtime && file && selectedFile && file !== selectedFile) {
      throw new Error('다른 LFM 파일을 사용하려면 페이지를 새로고침해 주세요.')
    }

    selectedFile = file
    modelFileName.value = file?.name || null
    modelError.value = null
    modelProgress.value = runtime ? 1 : 0
    if (!runtime) {
      modelState.value = 'idle'
      modelStatusText.value = file ? `${file.name} 선택됨` : '공식 GGUF 파일을 선택해 주세요'
    }
  }

  function validateModelFile(file: File) {
    if (!file.name.toLowerCase().endsWith('.gguf')) {
      throw new Error('GGUF 모델 파일만 사용할 수 있습니다.')
    }
    if (file.size !== LFM_MODEL_SIZE_BYTES) {
      throw new Error(`공식 Q4_K_M 파일 크기가 아닙니다. 예상 ${LFM_MODEL_SIZE_BYTES.toLocaleString()} bytes, 실제 ${file.size.toLocaleString()} bytes`)
    }
  }

  async function loadModel(file = selectedFile) {
    if (runtime) return runtime
    if (runtimePromise) return runtimePromise
    if (!file) throw new Error('LFM2.5 공식 Q4_K_M GGUF 파일을 먼저 선택해 주세요.')
    validateModelFile(file)
    if (!(await checkWebGPU())) throw new Error(modelError.value || 'WebGPU unavailable')

    selectedFile = file
    modelFileName.value = file.name
    modelState.value = 'loading'
    modelProgress.value = 0.05
    modelError.value = null
    modelStatusText.value = `${LFM_MODEL_LABEL} 로드·WebGPU 컴파일 중`

    runtimePromise = import('@wllama/wllama/esm/index.js')
      .then(async ({ LoggerWithoutDebug, Wllama }) => {
        const candidate = new Wllama(
          { default: wllamaWasmUrl },
          { parallelDownloads: 1, logger: LoggerWithoutDebug }
        )
        candidate.setCompat(null)
        await candidate.loadModel([file], {
          n_ctx: LFM_CONTEXT_WINDOW_SIZE,
          n_gpu_layers: 99999,
          n_threads: Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2))),
          warmup: false,
          reasoning: true
        })
        runtime = candidate
        modelState.value = 'ready'
        modelProgress.value = 1
        modelStatusText.value = `${LFM_MODEL_LABEL} 준비됨`
        return candidate
      })
      .catch((error: unknown) => {
        runtimePromise = null
        runtime = null
        modelState.value = 'error'
        modelProgress.value = 0
        modelError.value = error instanceof Error ? error.message : String(error)
        modelStatusText.value = 'LFM 모델을 불러오지 못했습니다'
        throw error
      })

    return runtimePromise
  }

  function beginGeneration() {
    interrupted = false
    abortController = new AbortController()
  }

  function wasInterrupted() {
    return interrupted
  }

  async function generate(
    messages: WllamaChatMessage[],
    maxTokens: number,
    onContent: (text: string) => void | Promise<void>,
    onReasoning: () => void | Promise<void>
  ): Promise<LfmGenerationMetrics> {
    const loaded = await loadModel()
    beginGeneration()
    const started = performance.now()
    let firstTokenAt: number | null = null
    let firstVisibleTokenAt: number | null = null
    let content = ''
    let reasoningChunks = 0
    let completionTokens: number | undefined
    let reportedDecode: number | undefined

    const stream = await loaded.createChatCompletion({
      messages,
      max_tokens: Math.max(1, Math.min(1536, maxTokens)),
      temperature: 0,
      top_k: 1,
      stream: true,
      abortSignal: abortController?.signal,
      cache_prompt: true
    })

    for await (const chunk of stream) {
      if (interrupted) break
      const delta = chunk.choices[0]?.delta as { content?: string | null, reasoning_content?: string | null } | undefined
      const reasoningPiece = delta?.reasoning_content || ''
      if (reasoningPiece) {
        reasoningChunks += 1
        if (firstTokenAt === null) firstTokenAt = performance.now()
        await onReasoning()
      }
      const piece = delta?.content || ''
      if (piece) {
        if (firstTokenAt === null) firstTokenAt = performance.now()
        if (firstVisibleTokenAt === null) firstVisibleTokenAt = performance.now()
        content += piece
        await onContent(content)
      }
      if (chunk.usage?.completion_tokens) completionTokens = chunk.usage.completion_tokens
      if (chunk.timings?.predicted_per_second) reportedDecode = chunk.timings.predicted_per_second
    }

    const finished = performance.now()
    const measuredSeconds = firstTokenAt === null ? 0 : (finished - firstTokenAt) / 1000
    const decodeTokensPerSecond = reportedDecode
      ?? (completionTokens && completionTokens > 1 && measuredSeconds > 0
        ? (completionTokens - 1) / measuredSeconds
        : undefined)

    return {
      completionTokens,
      decodeTokensPerSecond,
      timeToFirstTokenSeconds: firstTokenAt === null ? undefined : (firstTokenAt - started) / 1000,
      timeToFirstVisibleTokenSeconds: firstVisibleTokenAt === null ? undefined : (firstVisibleTokenAt - started) / 1000,
      reasoningChunks
    }
  }

  async function stopGeneration() {
    interrupted = true
    abortController?.abort()
    abortController = null
  }

  return {
    modelState,
    modelProgress,
    modelStatusText,
    modelError,
    modelFileName,
    webgpuAvailable,
    checkWebGPU,
    selectModelFile,
    loadModel,
    generate,
    wasInterrupted,
    stopGeneration
  }
}
