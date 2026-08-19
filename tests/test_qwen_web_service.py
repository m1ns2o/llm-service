import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class QwenWebServiceTests(unittest.TestCase):
    def setUp(self):
        self.page = (ROOT / "web/app/pages/index.vue").read_text(encoding="utf-8")
        self.runtime = (ROOT / "web/app/composables/useLocalQwen.ts").read_text(
            encoding="utf-8"
        )

    def test_service_offers_current_qwen_tiers_with_balanced_default(self):
        self.assertIn("QWEN_MODEL_ID = 'Qwen3.5-2B-q4f16_1-MLC'", self.runtime)
        self.assertIn("Qwen3.5-0.8B-q4f16_1-MLC", self.runtime)
        self.assertIn("Qwen3.5-2B-q4f16_1-MLC", self.runtime)
        self.assertIn("Qwen3.5-4B-q4f16_1-MLC", self.runtime)
        self.assertIn("QWEN_CONTEXT_WINDOW_SIZE = 4096", self.runtime)
        self.assertIn(":items=\"modelOptions\"", self.page)
        self.assertIn("aria-label=\"모델 선택\"", self.page)

    def test_first_run_selects_low_memory_model_conservatively(self):
        self.assertIn("ramMb < 6144", self.runtime)
        self.assertIn("navigator as Navigator & { deviceMemory?: number }", self.runtime)
        self.assertIn("'Qwen3.5-0.8B-q4f16_1-MLC'", self.runtime)
        self.assertIn("localStorage.setItem('qwen-local-model-v1'", self.runtime)

    def test_native_android_bridge_prepares_and_streams_locally(self):
        self.assertIn("AndroidLLM?", self.runtime)
        self.assertIn("bridge.prepareModel(requestId, targetModelId)", self.runtime)
        self.assertIn("bridge.generate(requestId", self.runtime)
        self.assertIn("event.type === 'model-ready'", self.runtime)
        self.assertIn("event.type === 'token'", self.runtime)
        self.assertIn("nativeBridge()?.setColorScheme(scheme)", self.runtime)

    def test_browser_path_uses_webgpu_worker_and_streaming_markdown(self):
        worker = (ROOT / "web/app/workers/qwen.worker.ts").read_text(
            encoding="utf-8"
        )
        package = json.loads((ROOT / "web/package.json").read_text(encoding="utf-8"))

        self.assertEqual(package["dependencies"]["@mlc-ai/web-llm"], "0.2.84")
        self.assertIn("WebWorkerMLCEngineHandler", worker)
        self.assertIn("CreateWebWorkerMLCEngine", self.runtime)
        self.assertIn("stream: true", self.page)
        self.assertIn("for await (const chunk of chunks)", self.page)
        self.assertIn("<MDC", self.page)

    def test_model_switch_reuses_engine_without_parallel_loads(self):
        self.assertIn("if (engine && loadedModelId === targetModelId) return engine", self.runtime)
        self.assertIn("if (enginePromise) return enginePromise", self.runtime)
        self.assertIn("await (engine as any).reload(targetModelId", self.runtime)
        self.assertIn("loadedModelId = targetModelId", self.runtime)
        self.assertIn("modelWorker?.terminate()", self.runtime)

    def test_nuxt_ui_chat_controls_and_mobile_layout_are_present(self):
        self.assertIn("<UChatPrompt", self.page)
        self.assertIn("<UChatPromptSubmit", self.page)
        self.assertIn("<UDashboardSidebarCollapse", self.page)
        self.assertIn("class=\"chat-submit-button justify-center\"", self.page)
        self.assertIn("최대 · 2,048 토큰", self.page)
        self.assertNotIn("AI는 실수할 수 있습니다", self.page)

    def test_runtime_diagnostic_records_webgpu_and_warm_runs(self):
        result = json.loads(
            (ROOT / "benchmarks/results/qwen35-2b-4b-webllm-runtime-diagnostic.json")
            .read_text(encoding="utf-8")
        )

        self.assertTrue(result["environment"]["webgpu"])
        self.assertEqual(result["environment"]["adapter"]["architecture"], "rdna-3")
        self.assertEqual(result["runtime"]["backend"], "WebGPU")
        self.assertEqual(result["runtime"]["quantization"], "q4f16_1")
        self.assertGreater(
            result["models"]["qwen35-2b"]["warm_decode_tokens_per_s"],
            result["models"]["qwen35-4b"]["warm_decode_tokens_per_s"],
        )


if __name__ == "__main__":
    unittest.main()
