import { Wllama } from "./vendor/wllama/index.js";

const WLLAMA_VERSION = "3.5.1";
const MODELS = {
  "lfm25-vl16b": {
    label: "LFM2.5-VL-1.6B",
    upstream: "LiquidAI/LFM2.5-VL-1.6B-GGUF",
    revision: "0df8719db7180cedababc2bc589abfe5e8ebcd1f",
    modelFile: "LFM2.5-VL-1.6B-Q4_K_M.gguf",
    modelBytes: 730896256,
    mmprojFile: "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
    mmprojBytes: 583109888,
    parameters: "1.6B (1.2B LM + 0.4B vision)",
    imageMinTokens: 64,
    imageMaxTokens: 256,
  },
  "qwen35-4b": {
    label: "Qwen3.5-4B",
    upstream: "mradermacher/Qwen3.5-4B-GGUF",
    sourceModel: "Qwen/Qwen3.5-4B",
    revision: "1a5df2c0cba51dae8ac5888420360d8703707171",
    modelFile: "Qwen3.5-4B.Q4_K_M.gguf",
    modelBytes: 2708804800,
    mmprojFile: "Qwen3.5-4B.mmproj-Q8_0.gguf",
    mmprojBytes: 366894656,
    parameters: "4B LM + vision encoder",
    imageMinTokens: 1024,
    imageMaxTokens: 1024,
  },
};

const $ = (id) => document.getElementById(id);
let runtime = null;
let running = false;

const state = {
  schema_version: 1,
  benchmark: "browser-vlm-synthetic-v1",
  status: "idle",
  status_message: "대기 중",
  runtime: `wllama ${WLLAMA_VERSION} / llama.cpp WebGPU`,
  generation_settings: {
    temperature: 0,
    top_k: 1,
    max_tokens: 96,
    n_ctx: 4096,
    image_min_tokens: null,
    image_max_tokens: null,
  },
  requests: [],
  logs: [],
};
window.__vlmBrowserBenchmarkResult = state;

function log(message) {
  const line = `${new Date().toISOString()} ${message}`;
  state.logs.push(line);
  if (state.logs.length > 250) state.logs.shift();
  $("log").textContent = state.logs.join("\n");
  $("log").scrollTop = $("log").scrollHeight;
}

function setStatus(status, message, bad = false) {
  state.status = status;
  state.status_message = message;
  $("status").textContent = message;
  $("status").classList.toggle("bad", bad);
  renderSummary();
}

function normalize(text) {
  return String(text || "").toLowerCase().replace(/[\s,.:;$₩()\[\]-]+/g, "");
}

function hasAny(text, candidates) {
  const value = normalize(text);
  return candidates.some((candidate) => value.includes(normalize(candidate)));
}

function scoreAssertions(answer, assertions) {
  const checks = assertions.map((assertion) => ({
    id: assertion.id,
    passed: hasAny(answer, assertion.any),
  }));
  return {
    checks,
    passed: checks.every((check) => check.passed),
    score: checks.filter((check) => check.passed).length / checks.length,
  };
}

function makeCanvas(draw) {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 400;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  draw(ctx, canvas);
  return canvas;
}

function benchmarkTasks() {
  return [
    {
      id: "ko_ocr_schedule",
      title: "한국어 OCR · 일정표",
      prompt: "이미지에서 회의 날짜, 시간, 장소를 읽어 한 문장으로만 답하세요.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#10233f";
        ctx.fillRect(0, 0, 640, 82);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 38px sans-serif";
        ctx.fillText("회의 일정", 34, 55);
        ctx.fillStyle = "#10233f";
        ctx.font = "30px sans-serif";
        ctx.fillText("날짜   2026년 8월 17일", 48, 150);
        ctx.fillText("시간   오후 3시", 48, 225);
        ctx.fillText("장소   별빛도서관 2층", 48, 300);
      }),
      assertions: [
        { id: "date", any: ["2026년8월17일", "2026-08-17"] },
        { id: "time", any: ["오후3시", "15시", "3pm"] },
        { id: "place", any: ["별빛도서관"] },
        { id: "floor", any: ["2층"] },
      ],
    },
    {
      id: "en_ocr_receipt",
      title: "English OCR · receipt",
      prompt: "Read the receipt. Reply with only the total and the most expensive item.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#111827";
        ctx.font = "bold 40px monospace";
        ctx.fillText("NORTH STAR SHOP", 110, 65);
        ctx.font = "30px monospace";
        ctx.fillText("Notebook ........ $12", 70, 145);
        ctx.fillText("Pen .............. $3", 70, 205);
        ctx.fillText("Lamp ............ $25", 70, 265);
        ctx.beginPath(); ctx.moveTo(65, 295); ctx.lineTo(575, 295); ctx.stroke();
        ctx.font = "bold 34px monospace";
        ctx.fillText("TOTAL ........... $40", 70, 350);
      }),
      assertions: [
        { id: "total", any: ["$40", "total40", "40dollar"] },
        { id: "item", any: ["lamp"] },
      ],
    },
    {
      id: "ko_chart_reasoning",
      title: "한국어 도표 추론",
      prompt: "막대그래프에서 값이 가장 큰 항목과 Atlas보다 얼마나 큰지 한국어로 짧게 답하세요.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#17243a";
        ctx.font = "bold 30px sans-serif";
        ctx.fillText("Quarterly Score", 205, 42);
        const bars = [["Atlas", 42, "#3b82f6"], ["Boreal", 75, "#f97316"], ["Cygnus", 58, "#22c55e"]];
        ctx.font = "bold 24px sans-serif";
        bars.forEach(([name, value, color], index) => {
          const x = 95 + index * 180;
          const height = value * 3;
          ctx.fillStyle = color;
          ctx.fillRect(x, 320 - height, 95, height);
          ctx.fillStyle = "#17243a";
          ctx.fillText(String(value), x + 28, 310 - height);
          ctx.fillText(name, x + 4, 360);
        });
      }),
      assertions: [
        { id: "highest", any: ["boreal", "보리얼"] },
        { id: "difference", any: ["33"] },
      ],
    },
    {
      id: "en_spatial",
      title: "English spatial understanding",
      prompt: "How many shapes are shown, where is the blue circle, and what color is the center shape? Answer in one short sentence.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#ef4444";
        ctx.fillRect(65, 55, 105, 105);
        ctx.fillStyle = "#eab308";
        ctx.beginPath(); ctx.moveTo(320, 150); ctx.lineTo(250, 270); ctx.lineTo(390, 270); ctx.closePath(); ctx.fill();
        ctx.fillStyle = "#2563eb";
        ctx.beginPath(); ctx.arc(520, 315, 55, 0, Math.PI * 2); ctx.fill();
      }),
      assertions: [
        { id: "count", any: ["3shape", "three shape"] },
        { id: "position", any: ["bottomright", "lower right"] },
        { id: "center", any: ["yellow"] },
      ],
    },
    {
      id: "ko_table_reasoning",
      title: "한국어 표 OCR·계산",
      prompt: "표에서 재고가 가장 많은 항목과 물결보다 몇 개 많은지 한국어로 짧게 답하세요.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#10233f";
        ctx.fillRect(0, 0, 640, 78);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 35px sans-serif";
        ctx.fillText("재고 현황", 245, 52);
        ctx.fillStyle = "#10233f";
        ctx.font = "30px sans-serif";
        ctx.fillText("항목", 120, 125); ctx.fillText("수량", 430, 125);
        ctx.fillText("해솔", 120, 190); ctx.fillText("14", 440, 190);
        ctx.fillText("물결", 120, 255); ctx.fillText("9", 440, 255);
        ctx.fillText("소나무", 120, 320); ctx.fillText("21", 440, 320);
        ctx.strokeStyle = "#94a3b8";
        for (const y of [140, 205, 270, 335]) { ctx.beginPath(); ctx.moveTo(85, y); ctx.lineTo(555, y); ctx.stroke(); }
      }),
      assertions: [
        { id: "highest", any: ["소나무"] },
        { id: "difference", any: ["12"] },
      ],
    },
    {
      id: "en_boarding_pass",
      title: "English OCR · boarding card",
      prompt: "Read the card. Reply with only the flight, gate, and boarding time.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#0f766e";
        ctx.fillRect(0, 0, 640, 95);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 38px sans-serif";
        ctx.fillText("BOARDING CARD", 170, 62);
        ctx.fillStyle = "#134e4a";
        ctx.font = "bold 34px monospace";
        ctx.fillText("FLIGHT     AX204", 75, 170);
        ctx.fillText("GATE          C7", 75, 245);
        ctx.fillText("BOARDING   18:40", 75, 320);
      }),
      assertions: [
        { id: "flight", any: ["ax204"] },
        { id: "gate", any: ["c7"] },
        { id: "time", any: ["18:40", "1840"] },
      ],
    },
    {
      id: "ko_shape_count",
      title: "한국어 개수 세기",
      prompt: "빨간 사각형과 파란 원은 각각 몇 개인지 한국어로 한 문장으로 답하세요.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#ef4444";
        for (const [x, y] of [[80, 70], [205, 70], [330, 70], [455, 70], [145, 220]]) ctx.fillRect(x, y, 75, 75);
        ctx.fillStyle = "#2563eb";
        for (const [x, y] of [[320, 260], [500, 260]]) { ctx.beginPath(); ctx.arc(x, y, 42, 0, Math.PI * 2); ctx.fill(); }
      }),
      assertions: [
        { id: "red_squares", any: ["빨간사각형5", "사각형은5", "5개"] },
        { id: "blue_circles", any: ["파란원2", "원은2", "2개"] },
      ],
    },
    {
      id: "en_number_grid",
      title: "English visual sequence",
      prompt: "What number should replace the question mark? Reply with only the number.",
      canvas: makeCanvas((ctx) => {
        ctx.fillStyle = "#111827";
        ctx.font = "bold 54px monospace";
        const rows = [["2", "4", "6"], ["8", "10", "12"], ["14", "16", "?"]];
        rows.forEach((row, rowIndex) => row.forEach((value, colIndex) => ctx.fillText(value, 130 + colIndex * 190, 105 + rowIndex * 115)));
        ctx.strokeStyle = "#cbd5e1";
        for (const x of [95, 285, 475, 610]) { ctx.beginPath(); ctx.moveTo(x, 25); ctx.lineTo(x, 375); ctx.stroke(); }
        for (const y of [25, 140, 255, 375]) { ctx.beginPath(); ctx.moveTo(95, y); ctx.lineTo(610, y); ctx.stroke(); }
      }),
      assertions: [{ id: "next_number", any: ["18"] }],
    },
  ];
}

function warmupTask() {
  return {
    id: "warmup",
    prompt: "Reply only OK.",
    canvas: makeCanvas((ctx) => {
      ctx.fillStyle = "#111827";
      ctx.font = "bold 56px sans-serif";
      ctx.fillText("WARMUP", 195, 225);
    }),
  };
}

async function canvasBytes(canvas) {
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("PNG 변환 실패");
  return blob.arrayBuffer();
}

async function inspectEnvironment() {
  state.cross_origin_isolated = crossOriginIsolated;
  state.user_agent = navigator.userAgent;
  if (!navigator.gpu) {
    setStatus("runtime_blocked", "이 Chrome에서 WebGPU를 사용할 수 없습니다.", true);
    return false;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    setStatus("runtime_blocked", "WebGPU 어댑터를 찾지 못했습니다.", true);
    return false;
  }
  const info = adapter.info || {};
  state.webgpu = { vendor: info.vendor || null, architecture: info.architecture || null, device: info.device || null };
  $("environment").textContent = `WebGPU ${[info.vendor, info.architecture, info.device].filter(Boolean).join(" · ")} · crossOriginIsolated=${crossOriginIsolated}`;
  setStatus("detected", "WebGPU 사용 가능");
  return true;
}

function selectedFiles(config) {
  const files = [...$("model-files").files];
  if (files.length !== 2) throw new Error("모델 GGUF와 mmproj GGUF, 정확히 두 파일을 선택하세요.");
  const model = files.find((file) => file.name === config.modelFile);
  const mmproj = files.find((file) => file.name === config.mmprojFile);
  if (!model || !mmproj) throw new Error(`필요 파일: ${config.modelFile}, ${config.mmprojFile}`);
  if (model.size !== config.modelBytes || mmproj.size !== config.mmprojBytes) throw new Error("선택한 GGUF 파일 크기가 고정 manifest와 다릅니다.");
  return [model, mmproj];
}

async function loadModel(config) {
  const files = selectedFiles(config);
  const wasmUrl = new URL("./vendor/wllama/wasm/wllama.wasm", location.href).href;
  runtime = new Wllama(
    { default: wasmUrl },
    { logger: { debug: () => {}, log: (...args) => log(args.join(" ")), warn: (...args) => log(`warning: ${args.join(" ")}`), error: (...args) => log(`error: ${args.join(" ")}`) } },
  );
  runtime.setCompat(null);
  const started = performance.now();
  await runtime.loadModel(files, {
    n_ctx: 4096,
    n_gpu_layers: 99999,
    n_threads: Math.max(1, Math.min(4, Math.floor((navigator.hardwareConcurrency || 2) / 2))),
    warmup: false,
    flash_attn: true,
    image_min_tokens: config.imageMinTokens,
    image_max_tokens: config.imageMaxTokens,
    default_template_kwargs: { enable_thinking: false },
  });
  state.model_load_ms = Math.round(performance.now() - started);
  state.model_metadata = runtime.getModelMetadata();
  state.image_input_supported = runtime.supportInputModality("image");
  if (!state.image_input_supported) throw new Error("로드한 모델이 이미지 입력을 지원한다고 보고하지 않았습니다.");
  state.artifacts = files.map((file) => ({ name: file.name, size_bytes: file.size }));
}

async function infer(task, maxTokens = 96) {
  const image = await canvasBytes(task.canvas);
  const started = performance.now();
  let firstAnyAt = null;
  let firstVisibleAt = null;
  let answer = "";
  let reasoning = "";
  let usage = null;
  let timings = null;
  const stream = await runtime.createChatCompletion({
    messages: [
      { role: "system", content: "Answer directly and accurately in the language requested. Do not output chain-of-thought." },
      { role: "user", content: [{ type: "image", data: image }, { type: "text", text: task.prompt }] },
    ],
    max_tokens: maxTokens,
    temperature: 0,
    top_k: 1,
    cache_prompt: false,
    stream: true,
  });
  for await (const chunk of stream) {
    if (chunk.usage) usage = chunk.usage;
    if (chunk.timings) timings = chunk.timings;
    const delta = chunk.choices?.[0]?.delta || {};
    const reasoningPiece = delta.reasoning_content || "";
    const piece = delta.content || "";
    if ((reasoningPiece || piece) && firstAnyAt === null) firstAnyAt = performance.now();
    if (piece && firstVisibleAt === null) firstVisibleAt = performance.now();
    reasoning += reasoningPiece;
    answer += piece;
    if (task.id !== "warmup") {
      const target = document.querySelector(`[data-answer="${task.id}"]`);
      if (target) target.textContent = answer || "내부 추론 중…";
    }
  }
  const ended = performance.now();
  return {
    answer: answer.trim(),
    reasoning_chars: reasoning.length,
    first_token_ms: firstAnyAt === null ? null : Math.round(firstAnyAt - started),
    first_visible_token_ms: firstVisibleAt === null ? null : Math.round(firstVisibleAt - started),
    total_ms: Math.round(ended - started),
    completion_tokens: usage?.completion_tokens ?? timings?.predicted_n ?? null,
    prompt_tokens: usage?.prompt_tokens ?? timings?.prompt_n ?? null,
    prompt_tok_per_s: timings?.prompt_per_second ?? null,
    decode_tok_per_s: timings?.predicted_per_second ?? null,
  };
}

function renderTasks(tasks) {
  const container = $("tasks");
  container.textContent = "";
  for (const task of tasks) {
    const row = document.createElement("article");
    row.className = "task";
    row.append(task.canvas);
    const body = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = task.title;
    const prompt = document.createElement("div");
    prompt.className = "muted";
    prompt.textContent = task.prompt;
    const result = document.createElement("div");
    result.className = "metric";
    result.dataset.result = task.id;
    result.textContent = "대기 중";
    const answer = document.createElement("div");
    answer.className = "answer";
    answer.dataset.answer = task.id;
    body.append(title, prompt, result, answer);
    row.append(body);
    container.append(row);
  }
}

function renderSummary() {
  const completed = state.requests.length;
  const assertions = state.requests.flatMap((request) => request.quality?.checks || []);
  const passed = assertions.filter((check) => check.passed).length;
  const decode = state.requests.map((request) => request.decode_tok_per_s).filter(Number.isFinite);
  const visible = state.requests.map((request) => request.first_visible_token_ms).filter(Number.isFinite);
  $("summary").textContent = JSON.stringify({
    model: state.model?.label || null,
    status: state.status,
    completed: `${completed}/${state.task_count || 8}`,
    quality_assertions: assertions.length ? `${passed}/${assertions.length}` : null,
    quality_percent: assertions.length ? Number((passed / assertions.length * 100).toFixed(1)) : null,
    mean_decode_tok_per_s: decode.length ? Number((decode.reduce((a, b) => a + b, 0) / decode.length).toFixed(2)) : null,
    mean_first_visible_token_ms: visible.length ? Math.round(visible.reduce((a, b) => a + b, 0) / visible.length) : null,
    model_load_ms: state.model_load_ms || null,
  }, null, 2);
}

async function runBenchmark() {
  if (running) return state;
  running = true;
  $("run").disabled = true;
  const config = MODELS[$("model").value];
  state.model = { id: $("model").value, ...config };
  state.generation_settings.image_min_tokens = config.imageMinTokens;
  state.generation_settings.image_max_tokens = config.imageMaxTokens;
  state.requests = [];
  state.started_at = new Date().toISOString();
  const tasks = benchmarkTasks();
  state.task_count = tasks.length;
  let heartbeat = null;
  renderTasks(tasks);
  $("progress").value = 0;
  try {
    if (!(await inspectEnvironment())) return state;
    setStatus("loading", `${config.label} 로드·컴파일 중`);
    heartbeat = window.setInterval(() => {
      $("live").textContent = `${state.status_message} · ${Math.round((Date.now() - Date.parse(state.started_at)) / 1000)}초 경과`;
    }, 1000);
    await loadModel(config);
    setStatus("warming_up", `${config.label} 이미지 경로 워밍업 중`);
    state.warmup = await infer(warmupTask(), 12);
    $("progress").value = 1;
    for (let index = 0; index < tasks.length; index += 1) {
      const task = tasks[index];
      setStatus("running", `${index + 1}/${tasks.length} · ${task.title}`);
      const result = await infer(task);
      result.id = task.id;
      result.prompt = task.prompt;
      result.quality = scoreAssertions(result.answer, task.assertions);
      state.requests.push(result);
      $("progress").value = index + 2;
      const metric = document.querySelector(`[data-result="${task.id}"]`);
      metric.className = result.quality.passed ? "metric pass" : "metric fail";
      metric.textContent = `${result.quality.passed ? "통과" : "실패"} · 첫 표시 ${result.first_visible_token_ms ?? "-"}ms · ${Number(result.decode_tok_per_s || 0).toFixed(2)} tok/s`;
      renderSummary();
    }
    window.clearInterval(heartbeat);
    heartbeat = null;
    state.finished_at = new Date().toISOString();
    setStatus("complete", `${config.label} 벤치마크 완료`);
    $("live").textContent = "결과가 메모리에 보관되었습니다. JSON 저장 버튼으로 내보낼 수 있습니다.";
    return state;
  } catch (error) {
    state.error = { name: error?.name || "Error", message: String(error?.message || error), stack: error?.stack || null };
    setStatus("failed", `실행 실패: ${state.error.message}`, true);
    log(state.error.stack || state.error.message);
    return state;
  } finally {
    if (heartbeat !== null) window.clearInterval(heartbeat);
    running = false;
    $("run").disabled = false;
    renderSummary();
  }
}

function exportResult() {
  const blob = new Blob([`${JSON.stringify(state, null, 2)}\n`], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${state.model?.id || "vlm"}-browser-benchmark.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

const queryModel = new URLSearchParams(location.search).get("model");
if (MODELS[queryModel]) $("model").value = queryModel;
renderTasks(benchmarkTasks());
renderSummary();
$("detect").addEventListener("click", () => inspectEnvironment().catch((error) => setStatus("failed", String(error), true)));
$("run").addEventListener("click", () => runBenchmark());
$("export").addEventListener("click", exportResult);
window.__vlmBrowserBenchmarkControl = { runBenchmark, inspectEnvironment, getState: () => state };
