import type { MLCEngineInterface } from '@mlc-ai/web-llm'

export const QWEN_MODEL_ID = 'Qwen3.5-4B-q4f16_1-MLC'
export const QWEN_MODEL_SOURCE = 'https://huggingface.co/mlc-ai/Qwen3.5-4B-q4f16_1-MLC'
export const QWEN_CONTEXT_WINDOW_SIZE = 2048

type ModelState = 'idle' | 'checking' | 'loading' | 'ready' | 'error'

let engine: MLCEngineInterface | null = null
let enginePromise: Promise<MLCEngineInterface> | null = null
let modelWorker: Worker | null = null
let interrupted = false

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    modelWorker?.terminate()
    modelWorker = null
    engine = null
    enginePromise = null
  })
}

export function useLocalQwen() {
  const modelState = useState<ModelState>('qwen-model-state', () => 'idle')
  const modelProgress = useState<number>('qwen-model-progress', () => 0)
  const modelStatusText = useState<string>('qwen-model-status-text', () => '모델 확인을 시작합니다')
  const modelError = useState<string | null>('qwen-model-error', () => null)
  const webgpuAvailable = useState<boolean | null>('qwen-webgpu', () => null)

  async function checkWebGPU() {
    if (!import.meta.client) return false

    modelState.value = 'checking'
    const gpu = navigator.gpu
    const adapter = gpu ? await gpu.requestAdapter() : null
    webgpuAvailable.value = Boolean(adapter)

    if (!adapter) {
      modelState.value = 'error'
      modelError.value = 'WebGPU를 사용할 수 없습니다. 최신 Chrome 또는 Edge가 필요합니다.'
      modelStatusText.value = 'WebGPU를 찾지 못했습니다'
      return false
    }

    modelState.value = engine ? 'ready' : 'idle'
    modelStatusText.value = engine ? 'Qwen3.5-4B 준비됨' : 'WebGPU 확인됨'
    return true
  }

  async function loadModel() {
    if (engine) return engine
    if (enginePromise) return enginePromise
    if (!(await checkWebGPU())) throw new Error(modelError.value || 'WebGPU unavailable')

    modelState.value = 'loading'
    modelError.value = null
    modelProgress.value = 0
    modelStatusText.value = 'Qwen3.5-4B 다운로드를 시작합니다'

    modelWorker = new Worker(new URL('../workers/qwen.worker.ts', import.meta.url), {
      type: 'module',
      name: 'qwen-webllm-worker'
    })

    enginePromise = import('@mlc-ai/web-llm')
      .then(({ CreateWebWorkerMLCEngine }) => CreateWebWorkerMLCEngine(
        modelWorker!,
        QWEN_MODEL_ID,
        {
          initProgressCallback(report) {
            modelProgress.value = Math.max(0, Math.min(1, report.progress || 0))
            modelStatusText.value = report.text || `모델 준비 ${Math.round(modelProgress.value * 100)}%`
          }
        },
        {
          context_window_size: QWEN_CONTEXT_WINDOW_SIZE,
          max_history_size: 1
        }
      ))
      .then((loadedEngine) => {
        engine = loadedEngine
        modelState.value = 'ready'
        modelProgress.value = 1
        modelStatusText.value = 'Qwen3.5-4B 준비됨'
        return loadedEngine
      })
      .catch((error: unknown) => {
        enginePromise = null
        modelWorker?.terminate()
        modelWorker = null
        modelState.value = 'error'
        modelError.value = error instanceof Error ? error.message : String(error)
        modelStatusText.value = '모델을 불러오지 못했습니다'
        throw error
      })

    return enginePromise
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

  return {
    modelState,
    modelProgress,
    modelStatusText,
    modelError,
    webgpuAvailable,
    checkWebGPU,
    loadModel,
    getEngine: () => engine,
    beginGeneration,
    wasInterrupted,
    stopGeneration
  }
}
