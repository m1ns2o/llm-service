import json
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from llm_bench.gates import evaluate_gate
from llm_bench.manifest import load_manifest
from llm_bench.runtime import load_runtime_config
from llm_bench.schema import RunEvidence
from llm_bench.scoring import CandidateMetrics, pareto_frontier


ROOT = Path(__file__).parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_manifest_contains_all_requested_combinations(self):
        models, combinations = load_manifest(ROOT / "configs/models.json")
        self.assertEqual(len(models), 18)
        self.assertEqual(
            models["qwen36-14b-a3b-fablevibes-q4km"].artifacts[0]["quantization"],
            "Q4_K_M",
        )
        self.assertEqual(
            models["qwen36-28b-a3b-reap20-iq3xxs"].platform_status["browser-windows"],
            "blocked",
        )
        self.assertEqual(set(combinations), {*(f"S{i}" for i in range(1, 5)), *(f"L{i}" for i in range(1, 9)), *(f"P{i}" for i in range(1, 9)), *(f"X{i}" for i in range(1, 4))})

    def test_runtime_config_enforces_platform_order_and_fallback(self):
        config = load_runtime_config(ROOT / "configs/runtime.json")
        self.assertEqual(config["android"]["backend_priority"], ["opencl", "vulkan", "cpu-arm64"])
        self.assertFalse(config["fixed_benchmark"]["student_input_allowed"])
        self.assertEqual(config["browser"]["native_build_script"], "scripts/build_wllama_webgpu.sh")
        self.assertTrue(config["browser"]["wasm_memory64_required"])
        self.assertIn("do not gate on free RAM", config["browser"]["memory_preflight_policy"])

    def test_browser_14b_runtime_artifact_and_provenance(self):
        models, _ = load_manifest(ROOT / "configs/models.json")
        model = models["qwen36-14b-a3b-fablevibes-q4km"]
        self.assertEqual(
            model.browser_artifact_manifest,
            "browser/model-shards/qwen36-14b-a3b-fablevibes-q4km.json",
        )
        model_artifact = model.artifacts[0]
        shard_manifest = json.loads(
            (ROOT / "browser/model-shards/qwen36-14b-a3b-fablevibes-q4km.json").read_text(encoding="utf-8")
        )
        self.assertEqual(shard_manifest["source"]["size_bytes"], model_artifact["size_bytes"])
        self.assertEqual(shard_manifest["source"]["sha256"], model_artifact["sha256"])
        self.assertIn(shard_manifest["status"], {"pending_split", "ready"})

        metadata = json.loads((ROOT / "browser/vendor/wllama/build-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "built")
        self.assertIn("qwen35moe", metadata["features"])
        for artifact in metadata["artifacts"]:
            path = ROOT / "browser/vendor/wllama" / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])

    def test_browser_14b_page_uses_shards_without_free_ram_gate(self):
        source = (ROOT / "browser/llama-webgpu-experimental.js").read_text(encoding="utf-8")
        self.assertIn("import { Wllama }", source)
        self.assertIn("model_shards_not_published", source)
        self.assertNotIn("window.LlamaCppWebRuntime", source)
        self.assertNotIn("device_memory_headroom_insufficient", source)

    def test_browser_14b_result_is_run_evidence_compatible(self):
        value = json.loads(
            (ROOT / "benchmarks/results/qwen36-14b-a3b-browser-webgpu-port.json").read_text(encoding="utf-8")
        )
        evidence = RunEvidence.from_dict(value)
        self.assertEqual(evidence.status, "runtime_blocked")
        self.assertEqual(evidence.blocked_reason, "model_shards_not_published")
        self.assertFalse(evidence.metadata["speed_metrics_valid"])
        self.assertEqual(evidence.metadata["runtime_smoke"]["status"], "passed")

    def test_gate_requires_complete_evidence(self):
        result = evaluate_gate(RunEvidence(model_id="qwen3-8b", platform="android", quantization="q4"))
        self.assertEqual(result.status, "incomplete")

    def test_gate_fails_on_oom_even_when_other_metrics_pass(self):
        evidence = RunEvidence(model_id="qwen3-8b", platform="android", quantization="q4", download_verified=True, cold_loads=3, requests_completed=20, thermal_minutes=20, initial_decode_tps=10, final_decode_tps=9, text_ttft_p50_seconds=2, oom_events=1)
        self.assertEqual(evaluate_gate(evidence).status, "fail")

    def test_pareto_preserves_tradeoff_candidates(self):
        small = CandidateMetrics("small", 70, 95, 14, 2, 2500, 5000, "android")
        quality = CandidateMetrics("quality", 95, 96, 8, 4, 6000, 7000, "android")
        dominated = CandidateMetrics("dominated", 80, 94, 7, 5, 6500, 7500, "android")
        self.assertEqual([c.id for c in pareto_frontier([small, quality, dominated])], ["small", "quality"])


if __name__ == "__main__":
    unittest.main()
