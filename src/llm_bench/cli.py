from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gates import evaluate_gate
from .manifest import load_manifest
from .runtime import load_runtime_config
from .schema import RunEvidence
from .scoring import CandidateMetrics, pareto_frontier


def _manifest_path(value: str | None) -> Path:
    return Path(value or "configs/models.json")


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest")

    runtime = sub.add_parser("validate-runtime")
    runtime.add_argument("--config", default="configs/runtime.json")

    gate = sub.add_parser("gate")
    gate.add_argument("evidence", type=Path)

    frontier = sub.add_parser("pareto")
    frontier.add_argument("metrics", type=Path, help="JSON array of CandidateMetrics")

    args = parser.parse_args()
    if args.command == "validate-manifest":
        models, combinations = load_manifest(_manifest_path(args.manifest))
        print(json.dumps({"status": "ok", "models": len(models), "combinations": len(combinations)}, ensure_ascii=False))
    elif args.command == "validate-runtime":
        config = load_runtime_config(args.config)
        print(json.dumps({"status": "ok", "sections": sorted(config)}, ensure_ascii=False))
    elif args.command == "gate":
        evidence = RunEvidence.from_dict(json.loads(args.evidence.read_text(encoding="utf-8")))
        result = evaluate_gate(evidence)
        print(json.dumps({"status": result.status, "reasons": result.reasons}, ensure_ascii=False))
        if result.status != "pass":
            raise SystemExit(1)
    elif args.command == "pareto":
        raw = json.loads(args.metrics.read_text(encoding="utf-8"))
        candidates = [CandidateMetrics(**item) for item in raw]
        print(json.dumps([candidate.__dict__ for candidate in pareto_frontier(candidates)], ensure_ascii=False))
