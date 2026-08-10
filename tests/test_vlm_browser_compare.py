import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class VlmBrowserCompareTests(unittest.TestCase):
    def test_page_exposes_two_model_multimodal_harness(self):
        page = (ROOT / "browser/vlm-browser-compare.html").read_text(encoding="utf-8")
        client = (ROOT / "browser/vlm-browser-compare.js").read_text(encoding="utf-8")

        self.assertIn('value="lfm25-vl16b"', page)
        self.assertIn('value="qwen35-4b"', page)
        self.assertIn('id="model-files"', page)
        self.assertIn('type: "image", data: image', client)
        self.assertIn('runtime.supportInputModality("image")', client)
        self.assertIn("ko_ocr_schedule", client)
        self.assertIn("en_ocr_receipt", client)
        self.assertIn("ko_chart_reasoning", client)
        self.assertIn("en_spatial", client)
        self.assertIn("ko_table_reasoning", client)
        self.assertIn("en_boarding_pass", client)
        self.assertIn("ko_shape_count", client)
        self.assertIn("en_number_grid", client)

    def test_comparison_uses_matching_quantization_and_runtime_settings(self):
        client = (ROOT / "browser/vlm-browser-compare.js").read_text(encoding="utf-8")

        self.assertEqual(client.count("Q4_K_M.gguf"), 2)
        self.assertEqual(client.count("Q8_0.gguf"), 2)
        self.assertIn("n_gpu_layers: 99999", client)
        self.assertIn("temperature: 0", client)
        self.assertIn("top_k: 1", client)
        self.assertIn("imageMinTokens: 64", client)
        self.assertIn("imageMaxTokens: 256", client)
        self.assertIn("imageMinTokens: 1024", client)
        self.assertIn("imageMaxTokens: 1024", client)
        self.assertIn("cache_prompt: false", client)

    def test_automation_script_parses_and_pins_all_artifacts(self):
        path = ROOT / "scripts/run_vlm_browser_compare.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)

        self.assertIn("verify_artifact", source)
        self.assertIn("51eafbc127f35598c8f1d2ec58b2520d6126c7d1195c4eca26832e63a2939d39", source)
        self.assertIn("aefc3c97c9eb30d9c0dd6af4c38250f5f5106b57c8cf92de7914c7d0a9c94da2", source)
        self.assertIn("window.__vlmBrowserBenchmarkResult", source)

    def test_combined_result_uses_complete_image_runs(self):
        result = json.loads(
            (
                ROOT
                / "benchmarks/results/lfm25-vl16b-vs-qwen35-4b-browser-webgpu.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(result["benchmark"], "browser-vlm-synthetic-v1")
        self.assertEqual(result["models"]["lfm25-vl16b"]["tasks_total"], 8)
        self.assertEqual(result["models"]["qwen35-4b"]["tasks_total"], 8)
        self.assertGreater(result["ratios"]["decode_lfm_over_qwen"], 2.0)
        self.assertGreater(
            result["models"]["qwen35-4b"]["quality_percent"],
            result["models"]["lfm25-vl16b"]["quality_percent"],
        )
        self.assertEqual(
            result["decision"]["speed_and_footprint_winner"], "lfm25-vl16b"
        )
        self.assertEqual(
            result["decision"]["multimodal_quality_winner"], "qwen35-4b"
        )


if __name__ == "__main__":
    unittest.main()
