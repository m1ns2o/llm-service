import { LOCAL_MODELS } from './localModelCatalog'

type ModelState = 'idle' | 'checking' | 'loading' | 'ready' | 'error'

interface LfmGenerationResult {
  text: string
  interrupted: boolean
  tokenCount: number
  firstTokenMs: number | null
  totalMs: number
  decodeTokensPerSecond: number
}

interface PendingGeneration {
  resolve: (result: LfmGenerationResult) => void
  reject: (error: Error) => void
  onToken: (chunk: string) => void
  onProgress?: (tokens: number, tokensPerSecond: number) => void
}

let worker: Worker | null = null
let loadPromise: Promise<void> | null = null
let resolveLoad: (() => void) | null = null
let rejectLoad: ((error: Error) => void) | null = null
let loadProgressByFile = new Map<string, { loaded: number, total: number }>()
const pendingGenerations = new Map<string, PendingGeneration>()

function lfmCacheKeyMatches(url: string) {
  let decoded = url
  try {
    decoded = decodeURIComponent(url)
  } catch {
    // An undecodable cache key can safely be checked in its raw form.
  }
  return decoded.includes('LiquidAI/LFM2-8B-A1B-ONNX')
    || decoded.includes('ae708d11dfe46fc80a99d3396f65d890a35061d0')
}

export function useLocalLfmWebGpu() {
  const modelState = useState<ModelState>('lfm-webgpu-model-state', () => 'idle')
  const modelProgress = useState<number>('lfm-webgpu-model-progress', () => 0)
  const modelStatusText = useState<string>('lfm-webgpu-model-status-text', () => 'LFM 속도 모드 대기 중')
  const modelError = useState<string | null>('lfm-webgpu-model-error', () => null)
  const active = useState<boolean>('lfm-webgpu-active', () => false)

  function updateAggregateProgress() {
    const parts = [...loadProgressByFile.values()].filter(item => item.total > 0)
    const loaded = parts.reduce((sum, item) => sum + Math.min(item.loaded, item.total), 0)
    const total = parts.reduce((sum, item) => sum + item.total, 0)
    if (total > 0) modelProgress.value = Math.max(0, Math.min(0.99, loaded / total))
  }

  function ensureWorker() {
    if (worker) return worker
    worker = new Worker(new URL('../workers/lfm.worker.ts', import.meta.url), {
      type: 'module',
      name: 'lfm2-8b-transformersjs-webgpu-worker'
    })
    worker.addEventListener('message', (event) => {
      const data = event.data || {}
      if (data.type === 'stage' && data.message) {
        modelStatusText.value = data.message
      } else if (data.type === 'load-progress') {
        const file = String(data.file || data.status || 'model')
        loadProgressByFile.set(file, {
          loaded: Number(data.loaded || 0),
          total: Number(data.total || 0)
        })
        updateAggregateProgress()
        const filename = file.split('/').at(-1)
        modelStatusText.value = filename
          ? `LFM2-8B 다운로드·컴파일 중 · ${filename}`
          : 'LFM2-8B 다운로드·컴파일 중'
      } else if (data.type === 'ready') {
        active.value = true
        modelState.value = 'ready'
        modelProgress.value = 1
        modelStatusText.value = 'LFM2-8B-A1B 준비됨'
        resolveLoad?.()
        loadPromise = null
        resolveLoad = null
        rejectLoad = null
      } else if (data.type === 'token') {
        pendingGenerations.get(data.requestId)?.onToken(String(data.text || ''))
      } else if (data.type === 'generation-progress') {
        pendingGenerations.get(data.requestId)?.onProgress?.(
          Number(data.tokenCount || 0),
          Number(data.decodeTokensPerSecond || 0)
        )
      } else if (data.type === 'complete') {
        const pending = pendingGenerations.get(data.requestId)
        if (!pending) return
        pendingGenerations.delete(data.requestId)
        pending.resolve({
          text: String(data.text || ''),
          interrupted: Boolean(data.interrupted),
          tokenCount: Number(data.tokenCount || 0),
          firstTokenMs: typeof data.firstTokenMs === 'number' ? data.firstTokenMs : null,
          totalMs: Number(data.totalMs || 0),
          decodeTokensPerSecond: Number(data.decodeTokensPerSecond || 0)
        })
      } else if (data.type === 'error' || data.type === 'worker-runtime-error') {
        const error = new Error(data.error?.message || 'LFM WebGPU 실행에 실패했습니다.')
        if (data.phase === 'load' || !data.requestId) {
          modelState.value = 'error'
          modelError.value = error.message
          modelStatusText.value = 'LFM2-8B을 불러오지 못했습니다'
          rejectLoad?.(error)
          loadPromise = null
          resolveLoad = null
          rejectLoad = null
        }
        if (data.requestId) {
          pendingGenerations.get(data.requestId)?.reject(error)
          pendingGenerations.delete(data.requestId)
        }
      }
    })
    worker.addEventListener('error', (event) => {
      const error = new Error(event.message || 'LFM WebGPU worker가 중단되었습니다.')
      modelState.value = 'error'
      modelError.value = error.message
      rejectLoad?.(error)
      for (const pending of pendingGenerations.values()) pending.reject(error)
      pendingGenerations.clear()
    })
    return worker
  }

  async function loadModel() {
    if (modelState.value === 'ready' && active.value) return
    if (loadPromise) return loadPromise
    modelState.value = 'loading'
    modelProgress.value = 0
    modelError.value = null
    modelStatusText.value = 'LFM2-8B 다운로드 준비 중'
    loadProgressByFile = new Map()
    const activeWorker = ensureWorker()
    loadPromise = new Promise<void>((resolve, reject) => {
      resolveLoad = resolve
      rejectLoad = reject
      activeWorker.postMessage({ type: 'load' })
    })
    return loadPromise
  }

  async function generate(
    messages: Array<{ role: 'system' | 'user' | 'assistant', content: string }>,
    maxNewTokens: number,
    onToken: (chunk: string) => void,
    onProgress?: (tokens: number, tokensPerSecond: number) => void
  ) {
    await loadModel()
    const requestId = crypto.randomUUID()
    return new Promise<LfmGenerationResult>((resolve, reject) => {
      pendingGenerations.set(requestId, { resolve, reject, onToken, onProgress })
      worker!.postMessage({ type: 'generate', requestId, messages, maxNewTokens })
    })
  }

  async function stopGeneration() {
    worker?.postMessage({ type: 'interrupt' })
  }

  async function unloadModel() {
    if (worker) {
      worker.postMessage({ type: 'unload' })
      worker.terminate()
    }
    worker = null
    loadPromise = null
    resolveLoad = null
    rejectLoad = null
    active.value = false
    modelState.value = 'idle'
    modelProgress.value = 0
    modelError.value = null
    modelStatusText.value = 'LFM 속도 모드 대기 중'
    for (const pending of pendingGenerations.values()) {
      pending.reject(new Error('모델이 메모리에서 해제되었습니다.'))
    }
    pendingGenerations.clear()
  }

  async function cachedRequestCount() {
    if (!import.meta.client || !('caches' in window)) return 0
    let count = 0
    for (const cacheName of await caches.keys()) {
      const cache = await caches.open(cacheName)
      for (const request of await cache.keys()) {
        if (lfmCacheKeyMatches(request.url)) count += 1
      }
    }
    return count
  }

  async function isModelCached() {
    return (await cachedRequestCount()) >= 5
  }

  async function deleteCachedModel() {
    await unloadModel()
    if (!import.meta.client || !('caches' in window)) return
    for (const cacheName of await caches.keys()) {
      const cache = await caches.open(cacheName)
      for (const request of await cache.keys()) {
        if (lfmCacheKeyMatches(request.url)) await cache.delete(request)
      }
    }
  }

  return {
    definition: LOCAL_MODELS['lfm2-8b'],
    modelState,
    modelProgress,
    modelStatusText,
    modelError,
    active,
    loadModel,
    generate,
    stopGeneration,
    unloadModel,
    isModelCached,
    deleteCachedModel
  }
}

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    worker?.terminate()
    worker = null
  })
}
