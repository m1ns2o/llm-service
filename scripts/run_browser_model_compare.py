#!/usr/bin/env python3
"""Run one 8B/9B model comparison case in Chrome through CDP."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

import requests
import websockets


ROOT = Path(__file__).resolve().parents[1]
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
LOAD_FAILURES = {"failed", "memory_blocked", "runtime_blocked"}
LFM_PREFILL_SIZE = 2_146_754_560
LFM_PREFILL_SHA256 = "2915866e7a0c45308dabb0034532c67534e8dbbac56b55d006ca19dd82bc9843"
LFM_GGUF_SIZE = 5_155_564_768
LFM_GGUF_SHA256 = "4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0"

PROMPTS = [
    {
        "id": "ko_general",
        "text": "오프라인 모바일 AI 비서를 만들 때 개인정보 보호, 응답 속도, 배터리를 함께 개선하는 3단계 계획을 작성해줘. 각 단계에 장점과 한계를 하나씩 포함해.",
        "max_new_tokens": 160,
    },
    {
        "id": "en_general",
        "text": "Write a three-step plan for an offline mobile AI assistant that improves privacy, response speed, and battery life. Include one benefit and one limitation per step.",
        "max_new_tokens": 160,
    },
    {
        "id": "ko_reasoning",
        "text": "어떤 기기의 배터리가 시간당 12%씩 줄고 현재 76%다. 최소 25%를 남겨야 한다면 최대 몇 시간 사용할 수 있는지 계산 과정과 결론을 한국어로 간단히 설명해줘.",
        "max_new_tokens": 128,
    },
    {
        "id": "ko_instruction",
        "text": "정확히 세 문장으로 답해줘. 각 문장은 1., 2., 3.으로 시작하고, 온디바이스 AI의 장점 하나와 단점 하나를 모두 포함해야 해.",
        "max_new_tokens": 128,
    },
]


class CdpClient:
    def __init__(self, socket: websockets.ClientConnection) -> None:
        self.socket = socket
        self.next_id = 1

    async def call(self, method: str, params: dict | None = None) -> dict:
        request_id = self.next_id
        self.next_id += 1
        await self.socket.send(
            json.dumps({"id": request_id, "method": method, "params": params or {}})
        )
        while True:
            message = json.loads(await self.socket.recv())
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"CDP {method}: {message['error']}")
                return message.get("result", {})

    async def evaluate(self, expression: str, *, await_promise: bool = True) -> object:
        result = await self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
            },
        )
        value = result["result"]
        if value.get("subtype") == "error":
            raise RuntimeError(value.get("description", "browser evaluation failed"))
        return value.get("value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(
            "lfm2-8b",
            "lfm25-8b",
            "lfm25-8b-symmetric",
            "qwen35-4b",
            "qwen35-9b",
        ),
        required=True,
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/browser-model-compare.html")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument(
        "--max-tokens-floor",
        type=int,
        default=0,
        help="Raise every prompt token budget to at least this value (maximum 2048).",
    )
    parser.add_argument("--prefill-lfm-shard", type=Path)
    parser.add_argument("--local-gguf", type=Path)
    return parser.parse_args()


def start_chrome(args: argparse.Namespace) -> subprocess.Popen:
    if not CHROME.is_file():
        raise SystemExit(f"Chrome not found: {CHROME}")
    command = [
        str(CHROME),
        f"--remote-debugging-port={args.port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={args.profile.resolve()}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-mode",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
        "--enable-unsafe-webgpu",
        "--disable-gpu-sandbox",
        "--window-position=-32000,-32000",
        "about:blank",
    ]
    if not args.headed:
        command.insert(-1, "--headless=new")
    return subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)


def wait_for_debugger(port: int, timeout: int = 30) -> dict:
    deadline = time.monotonic() + timeout
    endpoint = f"http://127.0.0.1:{port}/json/list"
    while time.monotonic() < deadline:
        try:
            pages = requests.get(endpoint, timeout=2).json()
            return next(item for item in pages if item["type"] == "page")
        except (requests.RequestException, ValueError, StopIteration):
            time.sleep(0.25)
    raise TimeoutError("Chrome DevTools endpoint did not become ready")


async def wait_for_page(client: CdpClient, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready = await client.evaluate(
            "document.readyState === 'complete' && !!window.__browserModelCompareControl"
        )
        if ready:
            return
        await asyncio.sleep(0.25)
    raise TimeoutError("comparison page did not initialize")


async def poll_load(client: CdpClient, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last_report = 0.0
    result: dict = {}
    while time.monotonic() < deadline:
        result = await client.evaluate("window.__browserModelComparisonResult")
        status = result.get("status")
        now = time.monotonic()
        if now - last_report >= 10:
            load_event = result.get("last_load_event") or {}
            heartbeat = result.get("load_heartbeat") or {}
            print(
                f"load status={status} model_load_ms={result.get('model_load_ms')} "
                f"message={result.get('status_message')} "
                f"event={load_event.get('status')} file={load_event.get('file')} "
                f"event_elapsed_ms={load_event.get('elapsed_ms')} "
                f"heartbeat_ms={heartbeat.get('elapsed_ms')}",
                flush=True,
            )
            last_report = now
        if status == "ready" or status in LOAD_FAILURES:
            return result
        await asyncio.sleep(2)
    result["status"] = "runtime_blocked"
    result["status_message"] = f"model load did not finish within {timeout} seconds"
    result["blocked_reason"] = "browser_model_load_timeout"
    result["load_timeout_seconds"] = timeout
    return result


async def prefill_lfm_cache(client: CdpClient, timeout: int) -> dict:
    expression = (
        "window.__browserModelCompareControl.prefillLfmShard("
        "'/__model_cache__/lfm25-8b/model_q4f16.onnx_data'"
        ").catch(error => console.error(error))"
    )
    await client.evaluate(expression, await_promise=False)
    deadline = time.monotonic() + timeout
    last_report = 0.0
    while time.monotonic() < deadline:
        result = await client.evaluate("window.__browserModelComparisonResult")
        prefill = result.get("cache_prefill", {})
        now = time.monotonic()
        if now - last_report >= 10:
            print(f"cache prefill status={prefill.get('status')}", flush=True)
            last_report = now
        if prefill.get("status") in {"complete", "already_cached"}:
            return prefill
        if prefill.get("status") == "failed":
            raise RuntimeError(prefill.get("error", "cache prefill failed"))
        await asyncio.sleep(1)
    raise TimeoutError("LFM cache prefill timed out")


async def run_prompt(client: CdpClient, prompt: dict, timeout: int) -> dict:
    before = await client.evaluate("window.__browserModelComparisonResult.requests.length")
    expression = (
        "window.__browserModelCompareControl.generate("
        f"{json.dumps(prompt['text'], ensure_ascii=False)},"
        f"{prompt['max_new_tokens']}"
        ").catch(error => console.error(error))"
    )
    await client.evaluate(expression, await_promise=False)
    deadline = time.monotonic() + timeout
    last_report = 0.0
    while time.monotonic() < deadline:
        result = await client.evaluate("window.__browserModelComparisonResult")
        now = time.monotonic()
        if now - last_report >= 5:
            print(
                f"prompt={prompt['id']} status={result.get('status')} "
                f"requests={len(result.get('requests', []))}",
                flush=True,
            )
            last_report = now
        if len(result.get("requests", [])) > before:
            request = result["requests"][-1]
            request["id"] = prompt["id"]
            return request
        if result.get("status") in LOAD_FAILURES:
            raise RuntimeError(result.get("status_message", "generation failed"))
        await asyncio.sleep(1)
    raise TimeoutError(f"generation timed out: {prompt['id']}")


async def run(args: argparse.Namespace, page: dict) -> tuple[dict, bytes | None]:
    async with websockets.connect(
        page["webSocketDebuggerUrl"], origin="http://127.0.0.1"
    ) as socket:
        client = CdpClient(socket)
        await client.call("Page.enable")
        await client.call("Runtime.enable")
        url = f"{args.url}?model={args.model}"
        await client.call("Page.navigate", {"url": url})
        await wait_for_page(client, 30)
        if args.local_gguf:
            document = await client.call("DOM.getDocument")
            selected = await client.call(
                "DOM.querySelector",
                {"nodeId": document["root"]["nodeId"], "selector": "#model-file"},
            )
            await client.call(
                "DOM.setFileInputFiles",
                {"nodeId": selected["nodeId"], "files": [str(args.local_gguf.resolve())]},
            )
        if args.prefill_lfm_shard:
            await prefill_lfm_cache(client, min(args.timeout, 900))
        await client.evaluate("document.getElementById('load').click(); true")
        result = await poll_load(client, args.timeout)
        if result.get("status") != "ready":
            return result, None
        if not args.load_only:
            try:
                for prompt in PROMPTS:
                    effective_prompt = dict(prompt)
                    effective_prompt["max_new_tokens"] = min(
                        2048,
                        max(prompt["max_new_tokens"], args.max_tokens_floor),
                    )
                    request = await run_prompt(
                        client, effective_prompt, min(args.timeout, 900)
                    )
                    print(
                        f"completed {prompt['id']}: ttft={request.get('first_token_ms')}ms "
                        f"decode={request.get('decode_tok_per_s')} tok/s",
                        flush=True,
                    )
            except Exception as error:
                result = await client.evaluate("window.__browserModelComparisonResult")
                result["automation_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                return result, None
            result = await client.evaluate("window.__browserModelComparisonResult")
            for request, prompt in zip(result.get("requests", []), PROMPTS):
                request["id"] = prompt["id"]
        screenshot = None
        if args.screenshot:
            capture = await client.call(
                "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True}
            )
            screenshot = base64.b64decode(capture["data"])
        return result, screenshot


def main() -> int:
    args = parse_args()
    prefill_hash = None
    gguf_hash = None
    if args.local_gguf:
        if args.model != "lfm25-8b":
            raise SystemExit("--local-gguf is only valid with --model lfm25-8b")
        if not args.local_gguf.is_file():
            raise SystemExit(f"LFM GGUF not found: {args.local_gguf}")
        if args.local_gguf.stat().st_size != LFM_GGUF_SIZE:
            raise SystemExit("LFM GGUF size mismatch")
        digest = hashlib.sha256()
        with args.local_gguf.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        gguf_hash = digest.hexdigest()
        if gguf_hash != LFM_GGUF_SHA256:
            raise SystemExit(f"LFM GGUF SHA-256 mismatch: {gguf_hash}")
    if args.prefill_lfm_shard:
        if args.model != "lfm25-8b":
            raise SystemExit("--prefill-lfm-shard is only valid with --model lfm25-8b")
        if not args.prefill_lfm_shard.is_file():
            raise SystemExit(f"LFM prefill shard not found: {args.prefill_lfm_shard}")
        if args.prefill_lfm_shard.stat().st_size != LFM_PREFILL_SIZE:
            raise SystemExit("LFM prefill shard size mismatch")
        digest = hashlib.sha256()
        with args.prefill_lfm_shard.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        prefill_hash = digest.hexdigest()
        if prefill_hash != LFM_PREFILL_SHA256:
            raise SystemExit(f"LFM prefill shard SHA-256 mismatch: {prefill_hash}")
    args.profile.mkdir(parents=True, exist_ok=True)
    process = start_chrome(args)
    try:
        page = wait_for_debugger(args.port)
        result, screenshot = asyncio.run(run(args, page))
        if prefill_hash:
            result["host_cache_prefill"] = {
                "path": str(args.prefill_lfm_shard.resolve()),
                "size_bytes": LFM_PREFILL_SIZE,
                "sha256": prefill_hash,
                "verified": True,
            }
        if gguf_hash:
            result["host_source"] = {
                "path": str(args.local_gguf.resolve()),
                "size_bytes": LFM_GGUF_SIZE,
                "sha256": gguf_hash,
                "verified": True,
            }
            result["download_verified"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if screenshot is not None and args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            args.screenshot.write_bytes(screenshot)
        print(f"result: {args.output}", flush=True)
        return 0 if result.get("status") == "ready" else 2
    finally:
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
