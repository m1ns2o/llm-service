import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class QwenWebServiceTests(unittest.TestCase):
    def test_service_offers_qwen_2b_default_and_4b_quality_mode(self):
        page = (ROOT / "web/app/pages/index.vue").read_text(encoding="utf-8")
        runtime = (ROOT / "web/app/composables/useLocalQwen.ts").read_text(
            encoding="utf-8"
        )

        self.assertIn("ref<ModelChoice>('qwen35-2b')", page)
        self.assertIn("Qwen3.5-2B · 빠른 기본", page)
        self.assertIn("Qwen3.5-4B · 품질 우선", page)
        self.assertIn("Qwen3.5-2B-q4f16_1-MLC", runtime)
        self.assertIn("Qwen3.5-4B-q4f16_1-MLC", runtime)
        self.assertIn("QWEN_CONTEXT_WINDOW_SIZE = 4096", runtime)
        self.assertEqual(runtime.count("quantization: 'q4f16_1'"), 3)

    def test_model_switch_releases_previous_gpu_runtime(self):
        runtime = (ROOT / "web/app/composables/useLocalQwen.ts").read_text(
            encoding="utf-8"
        )

        self.assertIn("await activeEngine.unload()", runtime)
        self.assertIn("modelWorker?.terminate()", runtime)
        self.assertIn("if (engine && loadedModel === model) return engine", runtime)
        self.assertIn("if (engine || enginePromise || modelWorker)", runtime)

    def test_chat_streams_in_worker_without_server_inference(self):
        page = (ROOT / "web/app/pages/index.vue").read_text(encoding="utf-8")
        worker = (ROOT / "web/app/workers/qwen.worker.ts").read_text(
            encoding="utf-8"
        )
        package = json.loads((ROOT / "web/package.json").read_text(encoding="utf-8"))

        self.assertEqual(package["dependencies"]["@mlc-ai/web-llm"], "0.2.84")
        self.assertIn("WebWorkerMLCEngineHandler", worker)
        self.assertIn("stream: true", page)
        self.assertIn("extra_body: { enable_thinking: false }", page)
        self.assertIn("for await (const chunk of chunks)", page)
        self.assertIn("STREAM_RENDER_INTERVAL_MS = 50", page)
        self.assertIn("performance.now() - lastRenderedAt", page)
        self.assertNotIn("await allowBrowserPaint()", page)
        self.assertIn("webgpuAdapterLabel", page)
        self.assertIn("MLC ${modelDefinition.value.quantization}", page)
        self.assertIn("model: selectedModel.value", page)
        self.assertIn("modelLabelByMessageId", page)
        self.assertNotIn("selectedModel.value === 'lfm25-8b'", page)

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
