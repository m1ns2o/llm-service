import {
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
  pipeline
} from '@huggingface/transformers'

const MODEL_ID = 'LiquidAI/LFM2-8B-A1B-ONNX'
const MODEL_REVISION = 'ae708d11dfe46fc80a99d3396f65d890a35061d0'

env.allowLocalModels = false
env.allowRemoteModels = true
env.useBrowserCache = true

let generator: Awaited<ReturnType<typeof pipeline<'text-generation'>>> | null = null
let generatorPromise: Promise<Awaited<ReturnType<typeof pipeline<'text-generation'>>>> | null = null
let generating = false
let interrupted = false
const stoppingCriteria = new InterruptableStoppingCriteria()

function post(type: string, value: Record<string, unknown> = {}) {
  self.postMessage({ type, ...value })
}

function serializeError(error: unknown) {
  return {
    message: error instanceof Error ? error.message : String(error),
    stack: error instanceof Error ? error.stack : undefined
  }
}

async function loadModel() {
  if (generator) return generator
  if (generatorPromise) return generatorPromise

  const startedAt = performance.now()
  post('stage', { state: 'loading', message: 'LFM2-8B ONNX 다운로드·WebGPU 컴파일 중' })
  generatorPromise = pipeline('text-generation', MODEL_ID, {
    dtype: 'q4f16',
    device: 'webgpu',
    revision: MODEL_REVISION,
    progress_callback(info) {
      post('load-progress', {
        status: info.status,
        file: 'file' in info ? info.file : undefined,
        loaded: 'loaded' in info ? Number(info.loaded || 0) : 0,
        total: 'total' in info ? Number(info.total || 0) : 0,
        progress: 'progress' in info ? Number(info.progress || 0) : 0,
        elapsedMs: Math.round(performance.now() - startedAt)
      })
    }
  }).then((loaded) => {
    generator = loaded
    generatorPromise = null
    post('ready', {
      modelId: MODEL_ID,
      revision: MODEL_REVISION,
      loadMs: Math.round(performance.now() - startedAt)
    })
    return loaded
  }).catch((error: unknown) => {
    generatorPromise = null
    generator = null
    post('error', { phase: 'load', error: serializeError(error) })
    throw error
  })
  return generatorPromise
}

async function generate(request: {
  requestId: string
  messages: Array<{ role: 'system' | 'user' | 'assistant', content: string }>
  maxNewTokens: number
}) {
  if (generating) {
    post('error', {
      requestId: request.requestId,
      phase: 'generate',
      error: { message: '이미 답변을 생성하고 있습니다.' }
    })
    return
  }

  generating = true
  interrupted = false
  stoppingCriteria.reset()
  const startedAt = performance.now()
  let firstTokenAt: number | null = null
  let tokenCount = 0
  let text = ''

  try {
    const loaded = await loadModel()
    const streamer = new TextStreamer(loaded.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function(chunk: string) {
        if (!chunk) return
        text += chunk
        post('token', { requestId: request.requestId, text: chunk })
      },
      token_callback_function() {
        tokenCount += 1
        if (firstTokenAt === null) firstTokenAt = performance.now()
        if (tokenCount === 1 || tokenCount % 4 === 0) {
          const decodeSeconds = firstTokenAt === null ? 0 : (performance.now() - firstTokenAt) / 1000
          post('generation-progress', {
            requestId: request.requestId,
            tokenCount,
            firstTokenMs: firstTokenAt === null ? null : Math.round(firstTokenAt - startedAt),
            decodeTokensPerSecond: tokenCount > 1 && decodeSeconds > 0
              ? (tokenCount - 1) / decodeSeconds
              : 0
          })
        }
      }
    })

    post('stage', { requestId: request.requestId, state: 'generating', message: 'LFM2-8B 답변 생성 중' })
    await loaded(request.messages, {
      max_new_tokens: Math.max(1, Math.min(256, request.maxNewTokens || 160)),
      do_sample: false,
      streamer,
      stopping_criteria: stoppingCriteria
    })

    const finishedAt = performance.now()
    const decodeSeconds = firstTokenAt === null ? 0 : (finishedAt - firstTokenAt) / 1000
    post('complete', {
      requestId: request.requestId,
      interrupted,
      text,
      tokenCount,
      firstTokenMs: firstTokenAt === null ? null : Math.round(firstTokenAt - startedAt),
      totalMs: Math.round(finishedAt - startedAt),
      decodeTokensPerSecond: tokenCount > 1 && decodeSeconds > 0
        ? (tokenCount - 1) / decodeSeconds
        : 0
    })
  } catch (error: unknown) {
    post('error', { requestId: request.requestId, phase: 'generate', error: serializeError(error) })
  } finally {
    generating = false
  }
}

self.addEventListener('message', async (event) => {
  const data = event.data || {}
  if (data.type === 'load') {
    try {
      await loadModel()
    } catch {
      // The detailed load error was already posted to the main thread.
    }
  } else if (data.type === 'generate') {
    await generate(data)
  } else if (data.type === 'interrupt') {
    interrupted = true
    stoppingCriteria.interrupt()
  } else if (data.type === 'unload') {
    if (generating) stoppingCriteria.interrupt()
    try {
      await generator?.dispose()
    } finally {
      generator = null
      generatorPromise = null
      post('unloaded')
    }
  }
})

post('worker-ready', { modelId: MODEL_ID, revision: MODEL_REVISION })
