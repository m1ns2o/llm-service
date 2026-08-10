import { Wllama } from "./vendor/wllama/index.js";

const TRANSFORMERS_JS_VERSION = "4.2.0";
const WEBLLM_VERSION = "0.2.84";
const MODELS = {
  "lfm2-8b": {
    key: "lfm2-8b",
    label: "LFM2-8B-A1B",
    upstream: "LiquidAI/LFM2-8B-A1B-ONNX",
    revision: "ae708d11dfe46fc80a99d3396f65d890a35061d0",
    quantization: "q4f16",
    artifact_size_bytes: 4_784_501_794,
    runtime: `Transformers.js ${TRANSFORMERS_JS_VERSION}`,
    backend: "transformersjs-webgpu-qmoe",
  },
  "lfm25-8b": {
    key: "lfm25-8b",
    label: "LFM2.5-8B-A1B",
    upstream: "LiquidAI/LFM2.5-8B-A1B-ONNX",
    revision: "9151c307c5fb0e70fbddec06b77609db4fdd58ff",
    quantization: "q4f16",
    artifact_size_bytes: 5_045_499_372,
    gguf_artifact_size_bytes: 5_155_564_768,
    gguf_sha256: "4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0",
    preferred_browser_path: "official-q4-k-m-gguf",
    runtime: `Transformers.js ${TRANSFORMERS_JS_VERSION}`,
    backend: "transformersjs-webgpu-qmoe",
    onnx_webgpu_status: "runtime_blocked_qmoe_zero_points",
  },
  "lfm25-8b-symmetric": {
    key: "lfm25-8b-symmetric",
    label: "LFM2.5-8B-A1B Symmetric Q4F16 (experimental)",
    upstream: "LiquidAI/LFM2.5-8B-A1B-ONNX",
    revision: "9151c307c5fb0e70fbddec06b77609db4fdd58ff",
    quantization: "q4f16-symmetric-qmoe",
    artifact_size_bytes: 4_924_174_336,
    runtime: `Transformers.js ${TRANSFORMERS_JS_VERSION}`,
    backend: "transformersjs-webgpu-qmoe",
    quality_gate: "failed",
    quality_gate_reason: "Korean language compliance regression after QMoE zero-point removal",
  },
  "qwen35-9b": {
    key: "qwen35-9b",
    label: "Qwen3.5-9B",
    upstream: "mlc-ai/Qwen3.5-9B-q4f16_1-MLC",
    model_id: "Qwen3.5-9B-q4f16_1-MLC",
    revision: "c7c5d3f5a81e37b8facbb72970940a1b131314a8",
    quantization: "q4f16_1",
    artifact_size_bytes: 5_061_443_935,
    runtime: `WebLLM ${WEBLLM_VERSION}`,
    backend: "webllm-webgpu",
  },
  "qwen35-4b": {
    key: "qwen35-4b",
    label: "Qwen3.5-4B",
    upstream: "mlc-ai/Qwen3.5-4B-q4f16_1-MLC",
    model_id: "Qwen3.5-4B-q4f16_1-MLC",
    revision: "44b42469f9e192814bfd90440e3b377d89ba7a13",
    quantization: "q4f16_1",
    artifact_size_bytes: 1_543_535_108,
    runtime: `WebLLM ${WEBLLM_VERSION}`,
    backend: "webllm-webgpu",
  },
};

const $ = (id) => document.getElementById(id);
const selectedKey = new URLSearchParams(location.search).get("model") || "lfm25-8b";
const config = MODELS[selectedKey] || MODELS["lfm25-8b"];
const isLfm = config.key.startsWith("lfm");
$("model").value = config.key;

let lfmWorker = null;
let lfmRuntime = null;
let qwenEngine = null;
let loadPromise = null;
let activeRequest = null;
let elapsedTimer = null;
let stageStarted = performance.now();

const state = {
  schema_version: 1,
  measured_at: new Date().toISOString(),
  model: structuredClone(config),
  platform: "browser-webgpu-direct",
  status: "idle",
  download_verified: false,
  requests: [],
  oom_events: 0,
  worker_crashes: 0,
};

function render() {
  $("metrics").textContent = JSON.stringify(state, null, 2);
  window.__browserModelComparisonResult = structuredClone(state);
}

function log(message) {
  $("log").textContent += `[${new Date().toLocaleTimeString()}] ${message}\n`;
  $("log").scrollTop = $("log").scrollHeight;
}

function setStatus(status, message, bad = false) {
  state.status = status;
  state.status_message = message;
  $("status").textContent = message;
  $("status").className = bad ? "status bad" : "status";
  render();
}

function setBusy({ loading = false, generating = false } = {}) {
  $("model").disabled = loading || generating || state.status === "ready";
  $("detect").disabled = loading || generating;
  $("load").disabled = loading || generating || state.status === "ready";
  $("send").disabled = loading || generating || state.status !== "ready";
  $("unload").disabled = loading || generating || state.status === "idle";
  $("model-file").disabled = loading || generating || state.status === "ready";
}

function startElapsed(label) {
  stageStarted = performance.now();
  clearInterval(elapsedTimer);
  elapsedTimer = setInterval(() => {
    $("elapsed").textContent = `${label} ${((performance.now() - stageStarted) / 1000).toFixed(1)}초`;
  }, 250);
}

function stopElapsed() {
  clearInterval(elapsedTimer);
  elapsedTimer = null;
}

function classifyFailure(message) {
  if (/out of memory|oom|allocation|buffer size/i.test(message)) {
    state.oom_events += 1;
    return "memory_blocked";
  }
  if (/WebGPU|GPUDevice|adapter|device lost|shader/i.test(message)) return "runtime_blocked";
  return "failed";
}

function stripThinking(text) {
  return String(text || "").replace(/<think>[\s\S]*?(<\/think>|$)\s*/g, "").trim();
}

function languageSignals(text) {
  const hangul = (text.match(/[가-힣]/g) || []).length;
  const han = (text.match(/\p{Script=Han}/gu) || []).length;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  return { hangul_chars: hangul, han_chars: han, latin_chars: latin };
}

async function inspectEnvironment() {
  const adapter = navigator.gpu
    ? await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })
    : null;
  const info = adapter?.info || {};
  state.environment = {
    secure_context: window.isSecureContext,
    cross_origin_isolated: window.crossOriginIsolated,
    user_agent: navigator.userAgent,
    webgpu: Boolean(adapter),
    adapter: adapter ? {
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      description: info.description || null,
      max_buffer_size: Number(adapter.limits.maxBufferSize),
      max_storage_buffer_binding_size: Number(adapter.limits.maxStorageBufferBindingSize),
    } : null,
  };
  $("environment").textContent = adapter
    ? `WebGPU · ${info.description || info.architecture || info.vendor || "adapter available"}`
    : "WebGPU high-performance adapter를 찾지 못했습니다.";
  setStatus(adapter ? "preflight_passed" : "runtime_blocked", adapter ? "WebGPU 환경 확인 완료" : "WebGPU 사용 불가", !adapter);
  return Boolean(adapter);
}

async function prefillLfmShard(localUrl) {
  if (config.key !== "lfm25-8b") throw new Error("LFM cache prefill requested for another model");
  const remoteUrl = `https://huggingface.co/${config.upstream}/resolve/${config.revision}/onnx/model_q4f16.onnx_data`;
  const cache = await caches.open("transformers-cache");
  const existing = await cache.match(remoteUrl);
  if (existing) {
    state.cache_prefill = {
      status: "already_cached",
      remote_url: remoteUrl,
      content_length: Number(existing.headers.get("content-length") || 0),
    };
    render();
    return state.cache_prefill;
  }
  state.cache_prefill = { status: "copying", remote_url: remoteUrl, local_url: localUrl };
  render();
  try {
    const response = await fetch(localUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`local shard fetch failed: ${response.status}`);
    const contentLength = Number(response.headers.get("content-length") || 0);
    if (contentLength !== 2_146_754_560) {
      throw new Error(`local shard size mismatch: ${contentLength}`);
    }
    await cache.put(remoteUrl, response);
    const cached = await cache.match(remoteUrl);
    if (!cached) throw new Error("cache prefill entry is missing after put");
    state.cache_prefill = {
      status: "complete",
      remote_url: remoteUrl,
      local_url: localUrl,
      content_length: contentLength,
    };
    render();
    return state.cache_prefill;
  } catch (error) {
    state.cache_prefill = { ...state.cache_prefill, status: "failed", error: String(error?.message || error) };
    render();
    throw error;
  }
}

function ensureLfmWorker() {
  if (lfmWorker) return lfmWorker;
  lfmWorker = new Worker(`lfm25-8b-webgpu-worker.js?variant=${encodeURIComponent(config.key)}`, { type: "module" });
  lfmWorker.addEventListener("message", handleLfmMessage);
  lfmWorker.addEventListener("error", (event) => {
    state.worker_crashes += 1;
    const detail = event.message || "LFM worker crash";
    setStatus(classifyFailure(detail), detail, true);
    stopElapsed();
    setBusy();
    activeRequest?.reject(new Error(detail));
    activeRequest = null;
  });
  return lfmWorker;
}

function handleLfmMessage(event) {
  const message = event.data || {};
  state.last_worker_message_at = new Date().toISOString();
  if (message.type === "worker-ready") {
    state.worker = {
      status: "ready",
      transformers_js: message.transformers_js,
      model_id: message.model_id,
      revision: message.revision,
    };
    render();
  } else if (message.type === "stage") {
    state.load_stage = message.state || null;
    state.load_stage_message = message.message || null;
    render();
  } else if (message.type === "load-event") {
    state.last_load_event = {
      status: message.status || null,
      file: message.file || null,
      loaded: Number(message.loaded || 0),
      total: Number(message.total || 0),
      progress: Number(message.progress || 0),
      elapsed_ms: Number(message.elapsed_ms || 0),
    };
    render();
  } else if (message.type === "load-heartbeat") {
    state.load_heartbeat = {
      elapsed_ms: Number(message.elapsed_ms || 0),
      last_event: message.last_event || null,
    };
    render();
  } else if (message.type === "load-progress") {
    const pct = Math.max(0, Math.min(100, Number(message.progress || 0)));
    $("progress").value = pct;
    $("progress-label").textContent = `${message.file || "모델"} · ${pct.toFixed(1)}%`;
    render();
  } else if (message.type === "ready") {
    state.model_load_ms = message.load_ms;
    state.loaded_at = new Date().toISOString();
    state.model.external_data_shards = message.external_data_shards || null;
    state.model.local_model_id = message.local_model_id || null;
    state.model.tensor_bytes_unchanged = Boolean(message.external_data_shards);
    $("progress").value = 100;
    $("progress-label").textContent = `모델 준비 완료 · ${(message.load_ms / 1000).toFixed(1)}초`;
    stopElapsed();
    setStatus("ready", `${config.label} 모델 준비 완료`);
    setBusy();
    loadPromise?.resolve(state);
    loadPromise = null;
  } else if (message.type === "token") {
    if (!activeRequest || message.requestId !== activeRequest.id) return;
    activeRequest.text += message.text || "";
    $("answer").textContent = activeRequest.text;
  } else if (message.type === "generation-progress") {
    if (!activeRequest || message.requestId !== activeRequest.id) return;
    $("live-metrics").textContent = `첫 토큰 ${message.first_token_ms}ms · ${message.token_count} tokens · ${Number(message.decode_tok_per_s).toFixed(2)} tok/s`;
  } else if (message.type === "complete") {
    if (!activeRequest || message.requestId !== activeRequest.id) return;
    const finished = {
      text: message.text || activeRequest.text,
      first_token_ms: message.first_token_ms,
      total_ms: message.total_ms,
      completion_tokens: message.token_count,
      decode_tok_per_s: Number(Number(message.decode_tok_per_s).toFixed(2)),
      interrupted: Boolean(message.interrupted),
    };
    activeRequest.resolve(finished);
    activeRequest = null;
  } else if (message.type === "error") {
    const detail = message.error?.message || "LFM worker error";
    if (message.phase === "load") {
      stopElapsed();
      setStatus(classifyFailure(detail), `모델 로드 실패: ${detail}`, true);
      setBusy();
      loadPromise?.reject(new Error(detail));
      loadPromise = null;
    } else if (activeRequest) {
      activeRequest.reject(new Error(detail));
      activeRequest = null;
    }
    log(message.error?.stack || detail);
  } else if (message.type === "worker-runtime-error") {
    const detail = message.error?.message || "LFM worker runtime error";
    state.worker_runtime_error = message.error || { message: detail };
    setStatus(classifyFailure(detail), detail, true);
    log(message.error?.stack || detail);
  } else if (message.type === "unloaded") {
    lfmWorker?.terminate();
    lfmWorker = null;
    setStatus("idle", "모델을 해제했습니다.");
    setBusy();
  }
}

async function loadLfm() {
  const localFile = $("model-file").files[0];
  if (localFile) {
    if (localFile.size !== config.gguf_artifact_size_bytes) {
      throw new Error(`LFM GGUF size mismatch: ${localFile.size}`);
    }
    const started = performance.now();
    const wasmUrl = new URL("./vendor/wllama/wasm/wllama.wasm", location.href).href;
    lfmRuntime = new Wllama(
      { default: wasmUrl },
      {
        parallelDownloads: 1,
        logger: {
          debug: (...args) => log(`wllama debug: ${args.join(" ")}`),
          log: (...args) => log(`wllama: ${args.join(" ")}`),
          warn: (...args) => log(`wllama warning: ${args.join(" ")}`),
          error: (...args) => log(`wllama error: ${args.join(" ")}`),
        },
      },
    );
    lfmRuntime.setCompat(null);
    await lfmRuntime.loadModel([localFile], {
      n_ctx: 4096,
      n_gpu_layers: 99999,
      n_threads: Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2))),
      warmup: false,
    });
    state.model_load_ms = Math.round(performance.now() - started);
    state.loaded_at = new Date().toISOString();
    state.model.runtime = "wllama 3.5.1 / llama.cpp WebGPU";
    state.model.backend = "wllama-webgpu-gguf";
    state.model.quantization = "Q4_K_M";
    state.model.artifact_size_bytes = localFile.size;
    state.model.upstream = "LiquidAI/LFM2.5-8B-A1B-GGUF";
    state.model.revision = "dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0";
    state.model.quality_gate = "passed_four_prompt_smoke";
    delete state.model.quality_gate_reason;
    delete state.model.onnx_webgpu_status;
    state.generation_settings = { temperature: 0, top_k: 1 };
    state.model_metadata = lfmRuntime.getModelMetadata();
    state.local_file = { name: localFile.name, size_bytes: localFile.size };
    return state;
  }
  return new Promise((resolve, reject) => {
    loadPromise = { resolve, reject };
    ensureLfmWorker().postMessage({ type: "load" });
  });
}

async function loadQwen() {
  const webllm = await import(`https://esm.run/@mlc-ai/web-llm@${WEBLLM_VERSION}`);
  const modelList = webllm.prebuiltAppConfig.model_list.map((item) => {
    if (item.model_id !== config.model_id) return item;
    return {
      ...item,
      model: `https://huggingface.co/${config.upstream}/resolve/${config.revision}/`,
    };
  });
  const found = modelList.some((item) => item.model_id === config.model_id);
  if (!found) throw new Error(`${config.model_id} is absent from WebLLM ${WEBLLM_VERSION}`);
  const started = performance.now();
  qwenEngine = await webllm.CreateMLCEngine(config.model_id, {
    appConfig: { ...webllm.prebuiltAppConfig, model_list: modelList },
    initProgressCallback: (info) => {
      const fraction = Math.max(0, Math.min(1, Number(info?.progress || 0)));
      $("progress").value = fraction * 100;
      $("progress-label").textContent = info?.text || `Qwen 모델 로드 ${(fraction * 100).toFixed(1)}%`;
    },
  });
  state.model_load_ms = Math.round(performance.now() - started);
  state.generation_settings = {
    temperature: 0,
    enable_thinking: false,
    enable_thinking_transport: "extra_body",
  };
  state.loaded_at = new Date().toISOString();
  $("progress").value = 100;
  $("progress-label").textContent = `모델 준비 완료 · ${(state.model_load_ms / 1000).toFixed(1)}초`;
  return state;
}

async function loadModel() {
  if (state.status === "ready") return state;
  setBusy({ loading: true });
  if (!(await inspectEnvironment())) {
    setBusy();
    throw new Error("WebGPU adapter unavailable");
  }
  startElapsed("모델 로드");
  setStatus("loading", `${config.label} 다운로드·컴파일 중`);
  try {
    if (isLfm) await loadLfm();
    else await loadQwen();
    stopElapsed();
    setStatus("ready", `${config.label} 모델 준비 완료`);
    setBusy();
    return state;
  } catch (error) {
    stopElapsed();
    const detail = String(error?.message || error);
    setStatus(classifyFailure(detail), `모델 로드 실패: ${detail}`, true);
    setBusy();
    throw error;
  }
}

async function generateLfm(messages, maxNewTokens) {
  if (lfmRuntime) {
    const started = performance.now();
    let firstTokenAt = null;
    let text = "";
    let usage = null;
    let timings = null;
    let reasoningChunks = 0;
    const stream = await lfmRuntime.createChatCompletion({
      messages,
      max_tokens: maxNewTokens,
      temperature: 0,
      top_k: 1,
      stream: true,
    });
    for await (const chunk of stream) {
      if (chunk.usage) usage = chunk.usage;
      if (chunk.timings) timings = chunk.timings;
      const delta = chunk.choices?.[0]?.delta || {};
      const reasoningPiece = delta.reasoning_content || "";
      if (reasoningPiece) {
        reasoningChunks += 1;
        if (firstTokenAt === null) firstTokenAt = performance.now();
        if (!text) $("answer").textContent = "내부 추론 중…";
      }
      const piece = delta.content || "";
      if (!piece) continue;
      if (firstTokenAt === null) firstTokenAt = performance.now();
      text += piece;
      $("answer").textContent = text;
    }
    const finished = performance.now();
    const completionTokens = Number(usage?.completion_tokens || timings?.predicted_n || 0);
    const measuredSeconds = firstTokenAt === null ? 0 : (finished - firstTokenAt) / 1000;
    const reportedDecode = Number(timings?.predicted_per_second);
    return {
      text,
      first_token_ms: firstTokenAt === null ? null : Math.round(firstTokenAt - started),
      total_ms: Math.round(finished - started),
      completion_tokens: completionTokens || null,
      decode_tok_per_s: Number.isFinite(reportedDecode)
        ? Number(reportedDecode.toFixed(2))
        : completionTokens > 1 && measuredSeconds > 0
          ? Number(((completionTokens - 1) / measuredSeconds).toFixed(2))
          : null,
      reasoning_chunks: reasoningChunks,
      usage,
    };
  }
  return new Promise((resolve, reject) => {
    const requestId = crypto.randomUUID();
    activeRequest = { id: requestId, text: "", resolve, reject };
    ensureLfmWorker().postMessage({
      type: "generate",
      requestId,
      messages,
      maxNewTokens,
    });
  });
}

async function generateQwen(messages, maxNewTokens) {
  await qwenEngine?.resetChat?.();
  const started = performance.now();
  let firstTokenAt = null;
  let text = "";
  let usage = null;
  const stream = await qwenEngine.chat.completions.create({
    messages,
    max_tokens: maxNewTokens,
    temperature: 0,
    stream: true,
    stream_options: { include_usage: true },
    extra_body: { enable_thinking: false },
  });
  for await (const chunk of stream) {
    if (chunk.usage) usage = chunk.usage;
    const delta = chunk.choices?.[0]?.delta?.content || "";
    if (!delta) continue;
    if (firstTokenAt === null) firstTokenAt = performance.now();
    text += delta;
    $("answer").textContent = text;
  }
  const finished = performance.now();
  const completionTokens = Number(usage?.completion_tokens || 0);
  const measuredSeconds = firstTokenAt === null ? 0 : (finished - firstTokenAt) / 1000;
  const reportedDecode = Number(usage?.extra?.decode_tokens_per_s);
  return {
    text,
    first_token_ms: firstTokenAt === null ? null : Math.round(firstTokenAt - started),
    total_ms: Math.round(finished - started),
    completion_tokens: completionTokens || null,
    decode_tok_per_s: Number.isFinite(reportedDecode)
      ? Number(reportedDecode.toFixed(2))
      : completionTokens > 1 && measuredSeconds > 0
        ? Number(((completionTokens - 1) / measuredSeconds).toFixed(2))
        : null,
    usage,
  };
}

async function generate(prompt, maxNewTokens = 128, systemPrompt = null) {
  if (state.status !== "ready") await loadModel();
  const system = systemPrompt || $("system-prompt").value.trim();
  const limit = Math.max(1, Math.min(2048, Number(maxNewTokens) || 128));
  const messages = [
    { role: "system", content: system },
    { role: "user", content: String(prompt) },
  ];
  $("answer").textContent = "";
  $("live-metrics").textContent = "첫 토큰 대기 중";
  setBusy({ generating: true });
  startElapsed("생성");
  setStatus("generating", `${config.label} 생성 중`);
  try {
    const result = isLfm
      ? await generateLfm(messages, limit)
      : await generateQwen(messages, limit);
    const answer = stripThinking(result.text);
    const request = {
      language: /[가-힣]/.test(prompt) ? "ko" : "en",
      prompt: String(prompt),
      answer,
      raw_answer: result.text,
      max_new_tokens: limit,
      first_token_ms: result.first_token_ms,
      total_ms: result.total_ms,
      completion_tokens: result.completion_tokens,
      decode_tok_per_s: result.decode_tok_per_s,
      language_signals: languageSignals(answer),
      completed_at: new Date().toISOString(),
    };
    if (result.usage) request.usage = result.usage;
    if (result.reasoning_chunks) request.reasoning_chunks = result.reasoning_chunks;
    state.requests.push(request);
    state.requests_completed = state.requests.length;
    state.last_request = request;
    $("answer").textContent = answer || result.text || "(빈 응답)";
    $("live-metrics").textContent = `첫 토큰 ${result.first_token_ms ?? "-"}ms · ${result.completion_tokens ?? "-"} tokens · ${result.decode_tok_per_s ?? "-"} tok/s`;
    stopElapsed();
    setStatus("ready", `${config.label} 응답 완료`);
    setBusy();
    return request;
  } catch (error) {
    stopElapsed();
    const detail = String(error?.message || error);
    setStatus(classifyFailure(detail), `생성 실패: ${detail}`, true);
    setBusy();
    throw error;
  }
}

async function unloadModel() {
  if (isLfm) {
    if (lfmRuntime) {
      await lfmRuntime.exit();
      lfmRuntime = null;
      setStatus("idle", "모델을 해제했습니다.");
      setBusy();
    } else {
      lfmWorker?.postMessage({ type: "unload" });
    }
  } else {
    await qwenEngine?.unload?.();
    qwenEngine = null;
    setStatus("idle", "모델을 해제했습니다.");
    setBusy();
  }
}

$("model").addEventListener("change", (event) => {
  location.href = `${location.pathname}?model=${encodeURIComponent(event.target.value)}`;
});
$("detect").addEventListener("click", () => inspectEnvironment().catch((error) => log(String(error))));
$("load").addEventListener("click", () => loadModel().catch((error) => log(String(error?.stack || error))));
$("send").addEventListener("click", () => generate($("prompt").value.trim(), $("max-tokens").value).catch((error) => log(String(error?.stack || error))));
$("unload").addEventListener("click", () => unloadModel().catch((error) => log(String(error?.stack || error))));
$("export").addEventListener("click", () => {
  const blob = new Blob([`${JSON.stringify(state, null, 2)}\n`], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${config.key}-browser-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});

window.__browserModelCompareControl = { inspectEnvironment, prefillLfmShard, loadModel, generate, unloadModel };
$("lfm-gguf").hidden = config.key !== "lfm25-8b";
setBusy();
render();
