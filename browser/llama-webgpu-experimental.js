import { Wllama } from './vendor/wllama/index.js';

const $ = (id) => document.getElementById(id);
const MODEL = {
  id: 'qwen36-14b-a3b-fablevibes-q4km',
  label: 'Qwen3.6-14B-A3B FableVibes',
  architecture: 'qwen35moe',
  quantization: 'Q4_K_M',
  filename: 'Qwen3.6-14B-A3B-FableVibes-Q4_K_M.gguf',
  url: 'https://huggingface.co/tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF/resolve/main/Qwen3.6-14B-A3B-FableVibes-Q4_K_M.gguf',
  size_bytes: 8465553920,
  sha256: '21aa4b0b28090469e8a319c889451df2f1ea6aad27ac3818c8c8a86f86d5bc9e',
};
const RUNTIME = {
  name: 'wllama/llama.cpp WebGPU',
  version: '3.5.1',
  wllama_commit: '766d28e03eeac044fe055327d06b83d3f9b84544',
  llama_cpp_commit: 'dd4623a74f0c85e6b1dd9ee99a92b9c67cac3708',
  wasm_url: new URL('./vendor/wllama/wasm/wllama.wasm', location.href).href,
  wasm_sha256: '194e454efde8ab9783fa1190c89a1ebbaebbef742cad3813420d676dfb1fed98',
};
const SHARD_MANIFEST_URL = './model-shards/qwen36-14b-a3b-fablevibes-q4km.json';
const SMOKE_MODEL = 'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';
const PROMPT = '중학교 2학년 학생에게 일차함수 y=2x+3의 기울기와 y절편을 한 문단으로 설명해줘.';

let adapter = null;
let runtime = null;
let shardManifest = null;
let selectedFiles = [];
const state = {
  schema_version: 1,
  model_id: MODEL.id,
  model_architecture: MODEL.architecture,
  quantization: MODEL.quantization,
  artifact_url: MODEL.url,
  artifact_sha256: MODEL.sha256,
  artifact_size_bytes: MODEL.size_bytes,
  platform: 'browser-wllama-webgpu-experiment',
  runtime: RUNTIME.name,
  runtime_version: RUNTIME.version,
  runtime_wasm_sha256: RUNTIME.wasm_sha256,
  wllama_commit: RUNTIME.wllama_commit,
  llama_cpp_commit: RUNTIME.llama_cpp_commit,
  backend: null,
  status: 'pending',
  speed_metrics_valid: false,
  requests_completed: 0,
  oom_events: 0,
  crashes: 0,
  thermal_status: 'browser_api_unavailable',
};

const log = (message) => {
  $('log').textContent += `[${new Date().toLocaleTimeString()}] ${message}\n`;
};
const render = () => {
  $('metrics').textContent = JSON.stringify(state, null, 2);
  window.__browserLlamaBenchmarkResult = structuredClone(state);
};
const setStatus = (message, bad = false) => {
  $('status').textContent = message;
  $('status').className = bad ? 'bad' : 'ok';
};
const setBusy = (busy) => {
  for (const id of ['detect', 'smoke', 'run', 'unload']) $(id).disabled = busy;
};
const probeMemory64 = () => {
  try {
    new WebAssembly.Memory({ address: 'i64', initial: 1n });
    return true;
  } catch {
    return false;
  }
};
const classifyFailure = (error) => {
  const message = String(error?.message || error);
  if (/out of memory|oom|memory access out of bounds|allocation failed/i.test(message)) {
    state.oom_events += 1;
    return ['memory_blocked', 'browser_or_webgpu_out_of_memory'];
  }
  if (/WebAssembly|compile|JSPI|Suspending|Memory64|wasm/i.test(message)) {
    return ['compile_blocked', message];
  }
  if (/device lost|GPUDevice|WebGPU|adapter/i.test(message)) {
    return ['runtime_blocked', `webgpu_${message}`];
  }
  return ['runtime_blocked', message];
};

async function loadShardManifest() {
  const response = await fetch(SHARD_MANIFEST_URL, { cache: 'no-store' });
  if (!response.ok) throw new Error(`shard manifest HTTP ${response.status}`);
  shardManifest = await response.json();
  state.shard_manifest_status = shardManifest.status;
  state.shard_manifest_url = new URL(SHARD_MANIFEST_URL, location.href).href;
  const first = shardManifest.split?.first_shard_url;
  if (first) $('remote-url').value = first;
  $('model-note').textContent = `${MODEL.label} · ${MODEL.quantization} · ${(MODEL.size_bytes / 1e9).toFixed(2)} GB · shards: ${shardManifest.status}`;
  render();
}

async function inspectEnvironment({ ignoreModelStorage = false } = {}) {
  state.environment = {
    secure_context: window.isSecureContext,
    cross_origin_isolated: window.crossOriginIsolated,
    wasm_jspi: typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function',
    wasm_memory64: probeMemory64(),
    shared_array_buffer: typeof SharedArrayBuffer === 'function',
    navigator_gpu: !!navigator.gpu,
    reported_total_memory_gb: navigator.deviceMemory || null,
    memory_policy: 'reported total memory is informational; unused RAM is not a blocking gate',
  };
  if (!state.environment.secure_context || !state.environment.cross_origin_isolated) {
    state.status = 'compile_blocked';
    state.blocked_reason = 'cross_origin_isolation_required';
    setStatus('COOP/COEP 격리 서버가 필요합니다: compile_blocked', true);
    render();
    return false;
  }
  if (!state.environment.wasm_jspi || !state.environment.wasm_memory64) {
    state.status = 'compile_blocked';
    state.blocked_reason = 'browser_missing_jspi_or_memory64';
    setStatus('브라우저 JSPI 또는 Memory64 미지원: compile_blocked', true);
    render();
    return false;
  }

  const storage = navigator.storage?.estimate ? await navigator.storage.estimate() : {};
  const available = storage.quota != null && storage.usage != null ? storage.quota - storage.usage : null;
  state.storage = {
    quota_bytes: storage.quota ?? null,
    usage_bytes: storage.usage ?? null,
    available_bytes: available,
    persisted: navigator.storage?.persisted ? await navigator.storage.persisted() : null,
  };

  adapter = navigator.gpu ? await navigator.gpu.requestAdapter() : null;
  state.webgpu = { adapter_available: !!adapter };
  if (adapter) {
    const info = adapter.info || {};
    state.webgpu.adapter_info = {
      vendor: info.vendor || null,
      architecture: info.architecture || null,
      device: info.device || null,
      description: info.description || null,
    };
    state.webgpu.limits = {
      maxBufferSize: Number(adapter.limits.maxBufferSize),
      maxStorageBufferBindingSize: Number(adapter.limits.maxStorageBufferBindingSize),
      maxComputeBufferBindingSize: Number(adapter.limits.maxStorageBufferBindingSize),
    };
  }

  const forced = $('backend').value;
  if ((forced === 'webgpu' || forced === 'auto') && !adapter) {
    state.status = 'memory_blocked';
    state.blocked_reason = '14b_cpu_fallback_exceeds_wasm_linear_memory';
    setStatus('WebGPU 없음 · 14B CPU fallback은 WASM 메모리 한계 초과', true);
    render();
    return false;
  }
  if (forced === 'wasm-cpu') {
    state.status = 'memory_blocked';
    state.blocked_reason = '14b_cpu_fallback_exceeds_wasm_linear_memory';
    setStatus('14B 전체 CPU 로드는 WASM 메모리 한계 초과', true);
    render();
    return false;
  }

  const remoteRequested = !$('model-files').files.length && $('remote-url').value.trim();
  const required = shardManifest?.split?.total_size_bytes || MODEL.size_bytes;
  if (!ignoreModelStorage && remoteRequested && available !== null && available < required) {
    state.status = 'memory_blocked';
    state.blocked_reason = 'browser_storage_capacity_insufficient';
    setStatus('브라우저 저장공간 용량 부족: memory_blocked', true);
    render();
    return false;
  }
  state.status = 'preflight_passed';
  delete state.blocked_reason;
  state.backend = 'webgpu';
  setStatus('WebGPU·Memory64·JSPI·격리 환경 검사 완료');
  log(`adapter: ${state.webgpu.adapter_info?.description || state.webgpu.adapter_info?.architecture || 'available'}`);
  render();
  return true;
}

function createRuntime() {
  const nativeLogger = {
    debug: (...args) => log(`native debug: ${args.join(' ')}`),
    log: (...args) => log(`native: ${args.join(' ')}`),
    warn: (...args) => log(`native warning: ${args.join(' ')}`),
    error: (...args) => log(`native error: ${args.join(' ')}`),
  };
  const instance = new Wllama(
    { default: RUNTIME.wasm_url },
    { parallelDownloads: 3, logger: nativeLogger }
  );
  instance.setCompat(null);
  return instance;
}

async function sha256Blob(blob) {
  const bytes = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join('');
}

async function verifyShards(blobs, names) {
  const expected = shardManifest?.status === 'ready' ? shardManifest.shards : [];
  const records = [];
  let verified = expected.length === blobs.length && expected.length > 0;
  for (let index = 0; index < blobs.length; index += 1) {
    setStatus(`shard 해시 검증 ${index + 1}/${blobs.length}`);
    const observed = await sha256Blob(blobs[index]);
    const wanted = expected[index];
    const sizeMatches = !wanted || blobs[index].size === wanted.size_bytes;
    const hashMatches = !wanted || observed === wanted.sha256;
    verified = verified && sizeMatches && hashMatches;
    records.push({
      filename: names[index] || wanted?.filename || `shard-${index + 1}`,
      size_bytes: blobs[index].size,
      sha256: observed,
      verified: !!wanted && sizeMatches && hashMatches,
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  state.shards = records;
  state.download_verified = verified;
  state.provenance = verified
    ? 'source hash verified before split and every shard hash matched'
    : 'observed shard hashes recorded; trusted split manifest not ready';
  render();
  return verified;
}

async function obtainModelBlobs() {
  if (selectedFiles.length) {
    const files = [...selectedFiles].sort((a, b) => a.name.localeCompare(b.name));
    if (!files[0].name.match(/-00001-of-\d{5}\.gguf$/i)) {
      throw new Error('첫 파일이 llama-gguf-split 형식이 아닙니다');
    }
    state.model_source = 'local_shards';
    await verifyShards(files, files.map((file) => file.name));
    return files;
  }

  const firstShardUrl = $('remote-url').value.trim();
  if (!firstShardUrl || !firstShardUrl.match(/-00001-of-\d{5}\.gguf(?:\?.*)?$/i)) {
    state.status = 'runtime_blocked';
    state.blocked_reason = 'model_shards_not_published';
    throw new Error('분할 가중치의 첫 shard URL 또는 로컬 shard 파일이 필요합니다');
  }
  if (shardManifest?.status !== 'ready') {
    state.status = 'runtime_blocked';
    state.blocked_reason = 'model_shard_manifest_not_ready';
    throw new Error('신뢰 가능한 shard 해시 매니페스트가 아직 준비되지 않았습니다');
  }
  state.model_source = 'remote_opfs_shards';
  const model = await runtime.modelManager.getModelOrDownload(
    { url: firstShardUrl },
    {
      progressCallback: ({ loaded, total }) => {
        $('progress').value = total ? loaded / total : 0;
        setStatus(`shards 다운로드 ${(loaded / Math.max(total, 1) * 100).toFixed(1)}%`);
        state.download_bytes = loaded;
        state.download_total_bytes = total;
      },
    }
  );
  const blobs = await model.open();
  const verified = await verifyShards(blobs, shardManifest.shards.map((item) => item.filename));
  if (!verified) {
    state.status = 'runtime_blocked';
    state.blocked_reason = 'shard_sha256_mismatch';
    throw new Error('분할 가중치 SHA-256 검증 실패');
  }
  return blobs;
}

async function unload() {
  if (runtime) await runtime.exit();
  runtime = null;
  setStatus('모델·WASM worker 해제 완료');
  log('runtime unloaded');
}

async function run14B() {
  setBusy(true);
  $('answer').textContent = '';
  state.started_at = new Date().toISOString();
  state.status = 'pending';
  state.speed_metrics_valid = false;
  try {
    if (!(await inspectEnvironment())) return;
    await unload();
    runtime = createRuntime();
    state.libllama_version = Wllama.getLibllamaVersion();
    const blobs = await obtainModelBlobs();
    const loadStarted = performance.now();
    setStatus('Qwen3.6-14B-A3B 로드·WebGPU kernel 준비 중...');
    await runtime.loadModel(blobs, {
      n_ctx: Math.max(256, Math.min(8192, Number($('context-size').value) || 1024)),
      n_gpu_layers: 99999,
      n_threads: Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2))),
      warmup: false,
    });
    state.model_load_ms = Math.round(performance.now() - loadStarted);
    state.model_metadata = runtime.getModelMetadata();
    state.backend = 'webgpu';

    const generationStarted = performance.now();
    let firstTokenAt = null;
    let visible = '';
    let finalUsage = null;
    let finalTimings = null;
    setStatus('고정 프롬프트 생성 중...');
    const stream = await runtime.createChatCompletion({
      messages: [{ role: 'user', content: PROMPT }],
      max_tokens: Math.max(8, Math.min(512, Number($('token-limit').value) || 64)),
      temperature: 0,
      top_k: 1,
      stream: true,
    });
    for await (const chunk of stream) {
      const text = chunk.choices?.[0]?.delta?.content || '';
      if (text && firstTokenAt === null) firstTokenAt = performance.now();
      visible += text;
      $('answer').textContent = visible;
      if (chunk.usage) finalUsage = chunk.usage;
      if (chunk.timings) finalTimings = chunk.timings;
    }
    const completedAt = performance.now();
    state.ttft_ms = firstTokenAt === null ? null : Math.round(firstTokenAt - generationStarted);
    state.total_ms = Math.round(completedAt - generationStarted);
    state.prompt_tok_per_s = finalTimings?.prompt_per_second ?? null;
    state.decode_tok_per_s = finalTimings?.predicted_per_second ?? null;
    state.prompt_tokens = finalUsage?.prompt_tokens ?? finalTimings?.prompt_n ?? null;
    state.decode_tokens = finalUsage?.completion_tokens ?? finalTimings?.predicted_n ?? null;
    state.requests_completed = 1;
    state.completed_at = new Date().toISOString();
    if (state.download_verified) {
      state.status = 'passed';
      state.speed_metrics_valid = true;
      setStatus('14B WebGPU 추론 완료');
    } else {
      state.status = 'runtime_passed_unverified';
      state.speed_metrics_valid = false;
      state.blocked_reason = 'trusted_shard_manifest_not_ready';
      setStatus('추론 완료 · shard 출처 검증 미완료', true);
    }
    render();
  } catch (error) {
    const [status, reason] = classifyFailure(error);
    if (state.status === 'pending' || state.status === 'preflight_passed') state.status = status;
    state.blocked_reason ||= reason;
    state.speed_metrics_valid = false;
    setStatus(`${state.status}: ${error.message || error}`, true);
    log(error.stack || String(error));
    render();
  } finally {
    setBusy(false);
  }
}

async function runRuntimeSmoke() {
  setBusy(true);
  $('answer').textContent = '';
  try {
    if (!(await inspectEnvironment({ ignoreModelStorage: true }))) return;
    await unload();
    runtime = createRuntime();
    const started = performance.now();
    await runtime.loadModelFromUrl(SMOKE_MODEL, {
      n_ctx: 256,
      n_gpu_layers: 99999,
      n_threads: 2,
      progressCallback: ({ loaded, total }) => {
        $('progress').value = total ? loaded / total : 0;
        setStatus(`smoke 모델 다운로드 ${(loaded / Math.max(total, 1) * 100).toFixed(1)}%`);
      },
    });
    const result = await runtime.createCompletion({
      prompt: 'Once upon a time',
      max_tokens: 8,
      temperature: 0,
    });
    state.runtime_smoke = {
      status: 'passed',
      backend: 'webgpu',
      model_url: SMOKE_MODEL,
      elapsed_ms: Math.round(performance.now() - started),
      output: result.choices[0].text,
      libllama_version: Wllama.getLibllamaVersion(),
    };
    $('answer').textContent = result.choices[0].text;
    setStatus('WASM/WebGPU runtime smoke 통과');
    render();
  } catch (error) {
    const [status, reason] = classifyFailure(error);
    state.runtime_smoke = { status, blocked_reason: reason };
    setStatus(`smoke ${status}: ${error.message || error}`, true);
    log(error.stack || String(error));
    render();
  } finally {
    setBusy(false);
  }
}

$('model-files').addEventListener('change', (event) => {
  selectedFiles = [...event.target.files];
  if (selectedFiles.length) {
    $('remote-url').value = '';
    log(`selected ${selectedFiles.length} local shard files`);
  }
});
$('detect').addEventListener('click', () => inspectEnvironment().catch((error) => {
  state.status = 'runtime_blocked';
  state.blocked_reason = String(error.message || error);
  setStatus(state.blocked_reason, true);
  render();
}));
$('smoke').addEventListener('click', runRuntimeSmoke);
$('run').addEventListener('click', run14B);
$('unload').addEventListener('click', () => unload().catch((error) => log(String(error))));
$('save').addEventListener('click', () => {
  const blob = new Blob([`${JSON.stringify(state, null, 2)}\n`], { type: 'application/json' });
  const anchor = document.createElement('a');
  anchor.href = URL.createObjectURL(blob);
  anchor.download = `${MODEL.id}-browser.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
});

loadShardManifest().catch((error) => {
  state.shard_manifest_status = 'runtime_blocked';
  state.blocked_reason = String(error.message || error);
  log(`manifest: ${state.blocked_reason}`);
  render();
});
render();
