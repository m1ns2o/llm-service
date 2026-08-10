import type { MLCEngineInterface } from '@mlc-ai/web-llm'

export type QwenModelChoice = 'qwen35-2b' | 'qwen35-4b'

export interface QwenModelDefinition {
  id: string
  label: string
  source: string
  description: string
  storageHint: string
  estimatedVramMb: number
}

export const QWEN_MODELS: Record<QwenModelChoice, QwenModelDefinition> = {
  'qwen35-2b': {
    id: 'Qwen3.5-2B-q4f16_1-MLC',
    label: 'Qwen3.5-2B',
    source: 'https://huggingface.co/mlc-ai/Qwen3.5-2B-q4f16_1-MLC',
    description: '빠른 기본 모델입니다. 약 2.25GB VRAM으로 WebGPU에서 실행됩니다.',
    storageHint: '최초 1회 모델을 받은 뒤 브라우저 캐시에서 다시 사용합니다.',
    estimatedVramMb: 2245.44
  },
  'qwen35-4b': {
    id: 'Qwen3.5-4B-q4f16_1-MLC',
    label: 'Qwen3.5-4B',
    source: 'https://huggingface.co/mlc-ai/Qwen3.5-4B-q4f16_1-MLC',
    description: '품질 우선 모델입니다. 약 3.87GB VRAM으로 WebGPU에서 실행됩니다.',
    storageHint: '최초 1회 모델을 받은 뒤 브라우저 캐시에서 다시 사용합니다.',
    estimatedVramMb: 3867.82
  }
}

export const QWEN_CONTEXT_WINDOW_SIZE = 4096

type ModelState = 'idle' | 'checking' | 'loading' | 'ready' | 'error'

let engine: MLCEngineInterface | null = null
let enginePromise: Promise<MLCEngineInterface> | null = null
let modelWorker: Worker | null = null
let loadedModel: QwenModelChoice | null = null
let interrupted = false

async function disposeEngine() {
  const activeEngine = engine
  engine = null
  enginePromise = null
  loadedModel = null
  if (activeEngine) {
    try {
      await activeEngine.unload()
    } catch {
      // Terminating the worker below is the final cleanup path.
    }
  }
  modelWorker?.terminate()
  modelWorker = null
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    void disposeEngine()
  })
}

export function useLocalQwen() {
  const modelState = useState<ModelState>('qwen-model-state', () => 'idle')
  const modelProgress = useState<number>('qwen-model-progress', () => 0)
  const modelStatusText = useState<string>('qwen-model-status-text', () => 'WebGPU 확인 전')
  const modelError = useState<string | null>('qwen-model-error', () => null)
  const webgpuAvailable = useState<boolean | null>('qwen-webgpu', () => null)
  const activeModel = useState<QwenModelChoice | null>('qwen-active-model', () => null)

  async function checkWebGPU() {
    if (!import.meta.client) return false

    modelState.value = 'checking'
    const gpu = navigator.gpu
    const adapter = gpu ? await gpu.requestAdapter({ powerPreference: 'high-performance' }) : null
    webgpuAvailable.value = Boolean(adapter)

    if (!adapter) {
      modelState.value = 'error'
      modelError.value = 'WebGPU를 사용할 수 없습니다. 최신 Chrome 또는 Edge와 WebGPU 지원 GPU가 필요합니다.'
      modelStatusText.value = 'WebGPU를 찾지 못했습니다'
      return false
    }

    modelState.value = engine ? 'ready' : 'idle'
    modelStatusText.value = engine && loadedModel
      ? `${QWEN_MODELS[loadedModel].label} 준비됨`
      : 'WebGPU 확인됨'
    return true
  }

  async function unloadModel() {
    await disposeEngine()
    activeModel.value = null
    modelProgress.value = 0
    modelError.value = null
    modelState.value = 'idle'
    modelStatusText.value = '모델을 선택해 주세요'
  }

  async function loadModel(model: QwenModelChoice) {
    const definition = QWEN_MODELS[model]
    if (engine && loadedModel === model) return engine
    if (enginePromise && loadedModel === model) return enginePromise
    if (!(await checkWebGPU())) throw new Error(modelError.value || 'WebGPU unavailable')

    if (engine || enginePromise || modelWorker) {
      modelState.value = 'loading'
      modelStatusText.value = '이전 모델 메모리 해제 중'
      await disposeEngine()
    }

    loadedModel = model
    activeModel.value = model
    modelState.value = 'loading'
    modelError.value = null
    modelProgress.value = 0
    modelStatusText.value = `${definition.label} 다운로드 준비 중`

    modelWorker = new Worker(new URL('../workers/qwen.worker.ts', import.meta.url), {
      type: 'module',
      name: `qwen-webllm-worker-${model}`
    })

    enginePromise = import('@mlc-ai/web-llm')
      .then(({ CreateWebWorkerMLCEngine }) => CreateWebWorkerMLCEngine(
        modelWorker!,
        definition.id,
        {
          initProgressCallback(report) {
            modelProgress.value = Math.max(0, Math.min(1, report.progress || 0))
            modelStatusText.value = report.text || `${definition.label} 준비 ${Math.round(modelProgress.value * 100)}%`
          }
        },
        {
          context_window_size: QWEN_CONTEXT_WINDOW_SIZE,
          max_history_size: 1
        }
      ))
      .then((loadedEngine) => {
        engine = loadedEngine
        enginePromise = null
        modelState.value = 'ready'
        modelProgress.value = 1
        modelStatusText.value = `${definition.label} 준비됨`
        return loadedEngine
      })
      .catch((error: unknown) => {
        enginePromise = null
        engine = null
        loadedModel = null
        activeModel.value = null
        modelWorker?.terminate()
        modelWorker = null
        modelState.value = 'error'
        modelError.value = error instanceof Error ? error.message : String(error)
        modelStatusText.value = `${definition.label}을 불러오지 못했습니다`
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
    activeModel,
    checkWebGPU,
    loadModel,
    unloadModel,
    getEngine: () => engine,
    beginGeneration,
    wasInterrupted,
    stopGeneration
  }
}
