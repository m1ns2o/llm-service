import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class BrowserModelCompareTests(unittest.TestCase):
    def test_lfm25_gguf_path_hides_reasoning_and_uses_4096_context(self):
        source = (ROOT / "browser/browser-model-compare.js").read_text(encoding="utf-8")

        self.assertIn("n_ctx: 4096", source)
        self.assertIn('const reasoningPiece = delta.reasoning_content || ""', source)
        self.assertIn('const piece = delta.content || ""', source)
        self.assertNotIn("delta.content || delta.reasoning_content", source)
        self.assertIn('quality_gate = "passed_four_prompt_smoke"', source)

    def test_symmetric_qmoe_candidate_is_explicitly_experimental(self):
        page = (ROOT / "browser/browser-model-compare.html").read_text(encoding="utf-8")
        client = (ROOT / "browser/browser-model-compare.js").read_text(encoding="utf-8")
        worker = (ROOT / "browser/lfm25-8b-webgpu-worker.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("Symmetric Q4F16 · experimental", page)
        self.assertIn('quality_gate: "failed"', client)
        self.assertIn("Korean language compliance regression", client)
        self.assertIn("transformersjs-onnx-symmetric-shift", worker)
        self.assertIn("@huggingface/transformers@4.2.0", worker)

    def test_final_gguf_evidence_has_only_final_answers(self):
        evidence = json.loads(
            (
                ROOT
                / "benchmarks/results/lfm25-8b-browser-webgpu-gguf-context4096.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(evidence["status"], "ready")
        self.assertTrue(evidence["download_verified"])
        self.assertEqual(len(evidence["requests"]), 4)
        self.assertTrue(all(request["reasoning_chunks"] > 0 for request in evidence["requests"]))
        answers = {request["id"]: request["answer"] for request in evidence["requests"]}
        self.assertIn("4.25시간", answers["ko_reasoning"])
        self.assertIn("1.", answers["ko_instruction"])
        self.assertIn("2.", answers["ko_instruction"])
        self.assertIn("3.", answers["ko_instruction"])
        for answer in answers.values():
            self.assertNotIn("The user asks:", answer)
            self.assertNotIn("We need to", answer)

    def test_build_summary_keeps_failed_transforms_out_of_recommendation(self):
        summary = json.loads(
            (
                ROOT / "benchmarks/results/lfm25-8b-browser-build-summary.json"
            ).read_text(encoding="utf-8")
        )
        candidates = {item["id"]: item for item in summary["candidates"]}

        self.assertEqual(
            summary["decision"]["quality_first"], "lfm25-official-gguf-q4-k-m"
        )
        self.assertEqual(
            candidates["lfm25-official-gguf-q4-k-m"]["quality_smoke_passed"], 4
        )
        self.assertEqual(
            candidates["lfm25-symmetric-q4f16-shift-only"]["status"],
            "quality_failed_experimental",
        )
        self.assertEqual(
            candidates["lfm25-official-onnx-q4f16"]["status"], "runtime_blocked"
        )


if __name__ == "__main__":
    unittest.main()
