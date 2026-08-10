import {
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";

const VARIANTS = {
  "lfm2-8b": {
    modelId: "LiquidAI/LFM2-8B-A1B-ONNX",
    revision: "ae708d11dfe46fc80a99d3396f65d890a35061d0",
    localModelId: "lfm2-8b/transformersjs-onnx",
    externalDataShards: 10,
    label: "LFM2-8B-A1B",
  },
  "lfm25-8b": {
    modelId: "LiquidAI/LFM2.5-8B-A1B-ONNX",
    revision: "9151c307c5fb0e70fbddec06b77609db4fdd58ff",
    localModelId: "lfm25-8b/transformersjs-onnx",
    externalDataShards: 12,
    label: "LFM2.5-8B-A1B",
  },
  "lfm25-8b-symmetric": {
    modelId: "LiquidAI/LFM2.5-8B-A1B-ONNX",
    revision: "9151c307c5fb0e70fbddec06b77609db4fdd58ff",
    localModelId: "lfm25-8b/transformersjs-onnx-symmetric-shift",
    externalDataShards: 10,
    label: "LFM2.5-8B-A1B Symmetric Q4F16 (shift-only)",
  },
};
const VARIANT_KEY = new URL(self.location.href).searchParams.get("variant") || "lfm25-8b";
const VARIANT = VARIANTS[VARIANT_KEY] || VARIANTS["lfm25-8b"];
const MODEL_ID = VARIANT.modelId;
const MODEL_REVISION = VARIANT.revision;
const LOCAL_MODEL_ID = VARIANT.localModelId;

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/__model_cache__/";
env.useBrowserCache = false;

let generator = null;
let generatorPromise = null;
let generating = false;
let interrupted = false;
const stoppingCriteria = new InterruptableStoppingCriteria();
let lastLoadEvent = null;

function post(type, value = {}) {
  self.postMessage({ type, ...value });
}

function serializeError(error) {
  return {
    message: String(error?.message || error),
    stack: error?.stack || null,
  };
}

post("worker-ready", {
  model_id: MODEL_ID,
  revision: MODEL_REVISION,
  transformers_js: "4.2.0",
});

self.addEventListener("error", (event) => {
  post("worker-runtime-error", {
    error: serializeError(event.error || event.message || "worker error"),
  });
});

self.addEventListener("unhandledrejection", (event) => {
  post("worker-runtime-error", { error: serializeError(event.reason) });
});

async function loadModel() {
  if (generator) return generator;
  if (generatorPromise) return generatorPromise;

  generatorPromise = (async () => {
    const started = performance.now();
    lastLoadEvent = { status: "pipeline_start", elapsed_ms: 0 };
    post("stage", { state: "loading", message: `${VARIANT.label} ONNX 로드·WebGPU 컴파일 중` });
    const heartbeat = setInterval(() => {
      post("load-heartbeat", {
        elapsed_ms: Math.round(performance.now() - started),
        last_event: lastLoadEvent,
      });
    }, 5000);
    try {
      generator = await pipeline("text-generation", LOCAL_MODEL_ID, {
        dtype: "q4f16",
        device: "webgpu",
        local_files_only: true,
        subfolder: "",
        progress_callback: (info) => {
          lastLoadEvent = {
            status: info.status || null,
            file: info.file || null,
            loaded: Number(info.loaded ?? 0),
            total: Number(info.total ?? 0),
            progress: Number(info.progress ?? 0),
            elapsed_ms: Math.round(performance.now() - started),
          };
          post("load-event", lastLoadEvent);
          if (info.status === "progress_total" || info.status === "progress") {
            post("load-progress", lastLoadEvent);
          }
        },
      });
      post("ready", {
        model_id: MODEL_ID,
        revision: MODEL_REVISION,
        local_model_id: LOCAL_MODEL_ID,
        external_data_shards: VARIANT.externalDataShards,
        load_ms: Math.round(performance.now() - started),
      });
      return generator;
    } catch (error) {
      generatorPromise = null;
      post("error", { phase: "load", error: serializeError(error) });
      throw error;
    } finally {
      clearInterval(heartbeat);
    }
  })();
  return generatorPromise;
}

async function generate({ requestId, messages, maxNewTokens }) {
  if (generating) {
    post("error", {
      requestId,
      phase: "generate",
      error: { message: "이미 생성 중입니다." },
    });
    return;
  }

  generating = true;
  interrupted = false;
  stoppingCriteria.reset();
  const runStarted = performance.now();
  let firstTokenAt = null;
  let tokenCount = 0;
  let text = "";

  try {
    const loaded = await loadModel();
    const streamer = new TextStreamer(loaded.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        if (!chunk) return;
        text += chunk;
        post("token", { requestId, text: chunk });
      },
      token_callback_function: () => {
        tokenCount += 1;
        if (firstTokenAt === null) firstTokenAt = performance.now();
        if (tokenCount === 1 || tokenCount % 4 === 0) {
          const seconds = (performance.now() - firstTokenAt) / 1000;
          post("generation-progress", {
            requestId,
            token_count: tokenCount,
            first_token_ms: Math.round(firstTokenAt - runStarted),
            decode_tok_per_s: tokenCount > 1 && seconds > 0
              ? (tokenCount - 1) / seconds
              : 0,
          });
        }
      },
    });

    post("stage", { requestId, state: "generating", message: `${VARIANT.label} 생성 중` });
    await loaded(messages, {
      max_new_tokens: Math.max(1, Math.min(2048, Number(maxNewTokens) || 128)),
      do_sample: false,
      streamer,
      stopping_criteria: stoppingCriteria,
    });
    const finished = performance.now();
    const seconds = firstTokenAt === null ? 0 : (finished - firstTokenAt) / 1000;
    post("complete", {
      requestId,
      interrupted,
      text,
      token_count: tokenCount,
      first_token_ms: firstTokenAt === null ? null : Math.round(firstTokenAt - runStarted),
      total_ms: Math.round(finished - runStarted),
      decode_tok_per_s: tokenCount > 1 && seconds > 0 ? (tokenCount - 1) / seconds : 0,
    });
  } catch (error) {
    post("error", { requestId, phase: "generate", error: serializeError(error) });
  } finally {
    generating = false;
  }
}

self.addEventListener("message", async (event) => {
  const { type } = event.data || {};
  if (type === "load") {
    try { await loadModel(); } catch { /* detailed error was already posted */ }
  } else if (type === "generate") {
    await generate(event.data);
  } else if (type === "interrupt") {
    interrupted = true;
    stoppingCriteria.interrupt();
  } else if (type === "unload") {
    if (generating) stoppingCriteria.interrupt();
    try { await generator?.dispose?.(); } finally {
      generator = null;
      generatorPromise = null;
      post("unloaded");
    }
  }
});
