interface CompletionRequest {
  messages: Array<{ role: string, content: string }>
  temperature?: number
  max_tokens?: number
}

interface LocalEngine {
  chat: { completions: { create: (request: CompletionRequest & Record<string, unknown>) => Promise<AsyncIterable<any>> } }
  interruptGenerate: () => Promise<void>
}

interface NativeEvent {
  type: 'model-progress' | 'model-ready' | 'token' | 'done' | 'error'
  requestId: string
  data: Record<string, any>
}

declare global {
  interface Window {
    AndroidLLM?: {
      getCapabilities: () => string
      prepareModel: (requestId: string, modelId: string) => void
      generate: (requestId: string, requestJson: string) => void
      interrupt: () => void
      setColorScheme: (scheme: 'light' | 'dark') => void
    }
    __nativeLlmDispatch?: (event: string) => void
  }
}

export const QWEN_MODEL_ID = 'Qwen3.5-2B-q4f16_1-MLC'
export const QWEN_MODEL_SOURCE = 'https://huggingface.co/mlc-ai/Qwen3.5-2B-q4f16_1-MLC'
export const QWEN_CONTEXT_WINDOW_SIZE = 4096
export type QwenModelId
  = 'Qwen3.5-0.8B-q4f16_1-MLC'
    | 'Qwen3.5-2B-q4f16_1-MLC'
    | 'Qwen3.5-4B-q4f16_1-MLC'
export const QWEN_MODEL_OPTIONS: Array<{ label: string, value: QwenModelId, description: string }> = [
  { label: 'Qwen 0.8B', value: 'Qwen3.5-0.8B-q4f16_1-MLC', description: '저메모리 최신 모델' },
  { label: 'Qwen 2B', value: 'Qwen3.5-2B-q4f16_1-MLC', description: '빠른 기본 모델' },
  { label: 'Qwen 4B', value: 'Qwen3.5-4B-q4f16_1-MLC', description: '더 높은 답변 품질' }
]

type ModelState = 'idle' | 'checking' | 'loading' | 'ready' | 'error'

let engine: LocalEngine | null = null
let enginePromise: Promise<LocalEngine> | null = null
let loadedModelId: QwenModelId | null = null
let modelWorker: Worker | null = null
let interrupted = false
const nativeListeners = new Map<string, (event: NativeEvent) => void>()

function nativeBridge() {
  return import.meta.client ? window.AndroidLLM : undefined
}

function installNativeDispatcher() {
  if (!import.meta.client || window.__nativeLlmDispatch) return
  window.__nativeLlmDispatch = (encoded) => {
    const event = JSON.parse(encoded) as NativeEvent
    nativeListeners.get(event.requestId)?.(event)
  }
}

function createNativeEngine(): LocalEngine {
  installNativeDispatcher()
  return {
    chat: {
      completions: {
        async create(request) {
          const bridge = nativeBridge()
          if (!bridge) throw new Error('Android native bridge unavailable')
          const requestId = crypto.randomUUID()
          const queue: any[] = []
          const waiters: Array<{ resolve: (value: IteratorResult<any>) => void, reject: (reason: Error) => void }> = []
          let finished = false
          let failure: Error | null = null
          const startedAt = performance.now()
          let firstTokenAt: number | null = null

          const push = (value: any) => {
            const waiter = waiters.shift()
            if (waiter) waiter.resolve({ value, done: false })
            else queue.push(value)
          }
          nativeListeners.set(requestId, (event) => {
            if (event.type === 'token') {
              if (firstTokenAt === null) firstTokenAt = performance.now()
              push({ choices: [{ delta: { content: event.data.text || '' } }] })
            } else if (event.type === 'done') {
              const elapsedSeconds = Math.max(0.001, Number(event.data.elapsedMs || performance.now() - startedAt) / 1000)
              push({
                choices: [{ delta: { content: '' } }],
                usage: { extra: {
                  decode_tokens_per_s: Number(event.data.tokenPieces || 0) / elapsedSeconds,
                  time_to_first_token_s: firstTokenAt === null ? undefined : (firstTokenAt - startedAt) / 1000
                } }
              })
              finished = true
              nativeListeners.delete(requestId)
              while (waiters.length) waiters.shift()!.resolve({ value: undefined, done: true })
            } else if (event.type === 'error') {
              failure = new Error(String(event.data.message || '네이티브 생성 실패'))
              finished = true
              nativeListeners.delete(requestId)
              while (waiters.length) waiters.shift()!.reject(failure)
            }
          })
          bridge.generate(requestId, JSON.stringify({
            messages: request.messages,
            temperature: request.temperature ?? 0.7,
            maxTokens: request.max_tokens ?? 1536
          }))

          const iterator: AsyncIterableIterator<any> = {
            [Symbol.asyncIterator]() { return iterator },
            async next(): Promise<IteratorResult<any>> {
              if (queue.length) return { value: queue.shift(), done: false }
              if (failure) throw failure
              if (finished) return { value: undefined, done: true }
              return new Promise((resolve, reject) => waiters.push({ resolve, reject }))
            }
          }
          return iterator
        }
      }
    },
    async interruptGenerate() {
      nativeBridge()?.interrupt()
    }
  }
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    modelWorker?.terminate()
    modelWorker = null
    engine = null
    enginePromise = null
    loadedModelId = null
  })
}

export function useLocalQwen() {
  const selectedModelId = useState<QwenModelId>('qwen-selected-model', () => QWEN_MODEL_ID)
  const modelState = useState<ModelState>('qwen-model-state', () => 'idle')
  const modelProgress = useState<number>('qwen-model-progress', () => 0)
  const modelStatusText = useState<string>('qwen-model-status-text', () => '초기 설정을 확인하고 있습니다')
  const modelError = useState<string | null>('qwen-model-error', () => null)
  const webgpuAvailable = useState<boolean | null>('qwen-webgpu', () => null)
  const nativeRuntime = useState<boolean>('qwen-native-runtime', () => false)
  const selectedModelLabel = computed(() => QWEN_MODEL_OPTIONS.find(option => option.value === selectedModelId.value)?.label || 'Qwen 2B')

  function isModelId(value: string): value is QwenModelId {
    return QWEN_MODEL_OPTIONS.some(option => option.value === value)
  }

  function restoreModelSelection() {
    if (!import.meta.client) return
    const savedModelId = localStorage.getItem('qwen-local-model-v1')
    if (savedModelId && isModelId(savedModelId)) {
      selectedModelId.value = savedModelId
      return
    }

    let ramMb: number | undefined
    const bridge = nativeBridge()
    if (bridge) {
      try {
        const capabilities = JSON.parse(bridge.getCapabilities()) as { ramMb?: number }
        ramMb = capabilities.ramMb
      } catch {
        // Keep the balanced 2B default when native capabilities are unavailable.
      }
    } else {
      const deviceMemoryGb = (navigator as Navigator & { deviceMemory?: number }).deviceMemory
      if (deviceMemoryGb) ramMb = deviceMemoryGb * 1024
    }

    selectedModelId.value = ramMb !== undefined && ramMb < 6144
      ? 'Qwen3.5-0.8B-q4f16_1-MLC'
      : QWEN_MODEL_ID
    localStorage.setItem('qwen-local-model-v1', selectedModelId.value)
  }

  async function checkWebGPU() {
    if (!import.meta.client) return false

    modelState.value = 'checking'
    if (nativeBridge()) {
      nativeRuntime.value = true
      webgpuAvailable.value = true
      modelState.value = engine ? 'ready' : 'idle'
      modelStatusText.value = engine ? `${selectedModelLabel.value} · 네이티브 GPU 준비됨` : '네이티브 가속 확인됨'
      return true
    }
    const gpu = navigator.gpu
    const adapter = gpu ? await gpu.requestAdapter() : null
    webgpuAvailable.value = Boolean(adapter)
    nativeRuntime.value = false

    if (!adapter) {
      modelState.value = 'error'
      modelError.value = 'WebGPU를 사용할 수 없습니다. 최신 Chrome 또는 Edge가 필요합니다.'
      modelStatusText.value = 'WebGPU를 찾지 못했습니다'
      return false
    }

    modelState.value = engine ? 'ready' : 'idle'
    modelStatusText.value = engine && loadedModelId === selectedModelId.value ? `${selectedModelLabel.value} 준비됨` : '기기 성능 확인됨'
    return true
  }

  async function loadModel() {
    const targetModelId = selectedModelId.value
    if (engine && loadedModelId === targetModelId) return engine
    if (enginePromise) return enginePromise
    if (!(await checkWebGPU())) throw new Error(modelError.value || 'GPU unavailable')

    modelState.value = 'loading'
    modelError.value = null
    modelProgress.value = 0
    modelStatusText.value = '초기 설정을 준비하고 있습니다'

    enginePromise = (async (): Promise<LocalEngine> => {
      const bridge = nativeBridge()
      if (bridge) {
        installNativeDispatcher()
        await new Promise<void>((resolve, reject) => {
          const requestId = crypto.randomUUID()
          nativeListeners.set(requestId, (event) => {
            if (event.type === 'model-progress') {
              modelProgress.value = Math.max(0, Math.min(1, Number(event.data.progress || 0)))
              modelStatusText.value = `${selectedModelLabel.value} 준비 중 · ${Math.round(modelProgress.value * 100)}%`
            } else if (event.type === 'model-ready') {
              nativeListeners.delete(requestId)
              modelStatusText.value = `${selectedModelLabel.value} · ${event.data.backend || 'native'} 준비됨`
              resolve()
            } else if (event.type === 'error') {
              nativeListeners.delete(requestId)
              reject(new Error(String(event.data.message || '네이티브 모델 준비 실패')))
            }
          })
          bridge.prepareModel(requestId, targetModelId)
        })
        return createNativeEngine()
      }
      if (engine) {
        await (engine as any).reload(targetModelId, {
          context_window_size: QWEN_CONTEXT_WINDOW_SIZE,
          max_history_size: 1
        })
        return engine
      }

      modelWorker = new Worker(new URL('../workers/qwen.worker.ts', import.meta.url), {
        type: 'module',
        name: 'qwen-webllm-worker'
      })

      const { CreateWebWorkerMLCEngine } = await import('@mlc-ai/web-llm')
      return CreateWebWorkerMLCEngine(
        modelWorker,
        targetModelId,
        {
          initProgressCallback(report) {
            modelProgress.value = Math.max(0, Math.min(1, report.progress || 0))
            modelStatusText.value = `${selectedModelLabel.value} 준비 중 · ${Math.round(modelProgress.value * 100)}%`
          }
        },
        {
          context_window_size: QWEN_CONTEXT_WINDOW_SIZE,
          max_history_size: 1
        }
      ) as unknown as LocalEngine
    })()
      .then((loadedEngine) => {
        engine = loadedEngine
        loadedModelId = targetModelId
        modelState.value = 'ready'
        modelProgress.value = 1
        modelStatusText.value = `${selectedModelLabel.value} 준비됨`
        return loadedEngine
      })
      .catch((error: unknown) => {
        modelWorker?.terminate()
        modelWorker = null
        engine = null
        loadedModelId = null
        modelState.value = 'error'
        modelError.value = error instanceof Error ? error.message : String(error)
        modelStatusText.value = '모델을 불러오지 못했습니다'
        throw error
      })
      .finally(() => {
        enginePromise = null
      })

    return enginePromise
  }

  async function selectModel(modelId: string) {
    if (!isModelId(modelId)) return
    selectedModelId.value = modelId
    if (import.meta.client) localStorage.setItem('qwen-local-model-v1', modelId)
    await loadModel()
  }

  function beginGeneration() {
    interrupted = false
  }

  function wasInterrupted() {
    return interrupted
  }

  async function stopGeneration() {
    interrupted = true
    await engine?.interruptGenerate()
  }

  function setNativeColorScheme(scheme: 'light' | 'dark') {
    nativeBridge()?.setColorScheme(scheme)
  }

  return {
    modelOptions: QWEN_MODEL_OPTIONS,
    selectedModelId,
    selectedModelLabel,
    modelState,
    modelProgress,
    modelStatusText,
    modelError,
    webgpuAvailable,
    nativeRuntime,
    setNativeColorScheme,
    checkWebGPU,
    restoreModelSelection,
    selectModel,
    loadModel,
    getEngine: () => engine,
    beginGeneration,
    wasInterrupted,
    stopGeneration
  }
}
