import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class QwenWebServiceTests(unittest.TestCase):
    def test_service_offers_adaptive_qwen_tiers_and_optional_lfm(self):
        page = (ROOT / "web/app/pages/index.vue").read_text(encoding="utf-8")
        runtime = (ROOT / "web/app/composables/useLocalQwen.ts").read_text(
            encoding="utf-8"
        )
        catalog = (ROOT / "web/app/composables/localModelCatalog.ts").read_text(
            encoding="utf-8"
        )

        self.assertIn("ref<ModelChoice>('qwen35-2b')", page)
        self.assertIn("Qwen3.5-0.8B-q4f16_1-MLC", catalog)
        self.assertIn("Qwen3.5-2B-q4f16_1-MLC", runtime)
        self.assertIn("Qwen3.5-4B-q4f16_1-MLC", runtime)
        self.assertIn("Qwen3.5-9B-q4f16_1-MLC", catalog)
        self.assertIn("LiquidAI/LFM2-8B-A1B-ONNX", catalog)
        self.assertIn("autoEligible: false", catalog)
        self.assertIn("LOCAL_MODEL_CHOICES.map", page)
        self.assertIn("QWEN_CONTEXT_WINDOW_SIZE = 4096", runtime)
        self.assertEqual(runtime.count("quantization: 'q4f16_1'"), 5)

    def test_hardware_profile_uses_conservative_browser_signals(self):
        profile = (ROOT / "web/app/composables/useHardwareProfile.ts").read_text(
            encoding="utf-8"
        )

        self.assertIn("navigator.hardwareConcurrency", profile)
        self.assertIn("deviceMemory", profile)
        self.assertIn("adapter.limits.maxBufferSize", profile)
        self.assertIn("maxStorageBufferBindingSize", profile)
        self.assertIn("adapter?.features.has('shader-f16')", profile)
        self.assertIn("model: 'qwen35-08b'", profile)
        self.assertIn("model: 'qwen35-2b'", profile)
        self.assertIn("model: 'qwen35-4b'", profile)
        self.assertNotIn("model: 'qwen35-9b'", profile)
        self.assertNotIn("model: 'lfm2-8b'", profile)

    def test_model_manager_reuses_and_deletes_browser_cache(self):
        page = (ROOT / "web/app/pages/index.vue").read_text(encoding="utf-8")
        qwen = (ROOT / "web/app/composables/useLocalQwen.ts").read_text(
            encoding="utf-8"
        )
        lfm = (ROOT / "web/app/composables/useLocalLfmWebGpu.ts").read_text(
            encoding="utf-8"
        )

        self.assertIn("hasModelInCache", qwen)
        self.assertIn("deleteModelAllInfoInCache", qwen)
        self.assertIn("for (const cacheName of await caches.keys())", lfm)
        self.assertIn("lfmCacheKeyMatches", lfm)
        self.assertIn("다운로드하고 실행", page)
        self.assertIn("캐시에서 실행", page)
        self.assertIn("removeCachedModel", page)

    def test_lfm_speed_mode_is_pinned_and_streamed_in_webgpu_worker(self):
        worker = (ROOT / "web/app/workers/lfm.worker.ts").read_text(
            encoding="utf-8"
        )
        package = json.loads((ROOT / "web/package.json").read_text(encoding="utf-8"))

        self.assertEqual(
            package["dependencies"]["@huggingface/transformers"],
            "4.0.0-next.9",
        )
        self.assertIn("LiquidAI/LFM2-8B-A1B-ONNX", worker)
        self.assertIn("ae708d11dfe46fc80a99d3396f65d890a35061d0", worker)
        self.assertIn("dtype: 'q4f16'", worker)
        self.assertIn("device: 'webgpu'", worker)
        self.assertIn("TextStreamer", worker)
        self.assertIn("post('token'", worker)

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
        self.assertIn("modelDefinition.value.runtime", page)
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
