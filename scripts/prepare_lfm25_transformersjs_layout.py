#!/usr/bin/env python3
"""Patch a local LFM2.5 ONNX layout for Transformers.js browser loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GENERATION_TAGS = ("{%- generation -%}", "{%- endgeneration -%}")


def strip_unsupported_generation_tags(template: str) -> str:
    for tag in GENERATION_TAGS:
        template = template.replace(tag, "")
    return template


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--external-data-shards", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--transform",
        choices=("repack-only", "symmetric-qmoe"),
        default="repack-only",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    config_path = model_dir / "config.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    template_path = model_dir / "chat_template.jinja"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    custom = config.setdefault("transformers.js_config", {})
    external = custom.setdefault("use_external_data_format", {})
    external["model_q4f16.onnx"] = args.external_data_shards
    custom["local_repack"] = {
        "source_revision": args.source_revision,
        "transform": args.transform,
        "tensor_bytes_unchanged": args.transform == "repack-only",
        "max_external_shard_bytes": 512 * 1024 * 1024,
    }
    if args.transform == "symmetric-qmoe":
        custom["local_repack"].update(
            {
                "qmoe_zero_point_mode": "implicit_uint4_midpoint_8",
                "source_quantization": "asymmetric_uint4_block32",
            }
        )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["chat_template"] = strip_unsupported_generation_tags(
        tokenizer_config["chat_template"]
    )
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    template = strip_unsupported_generation_tags(
        template_path.read_text(encoding="utf-8")
    )
    template_path.write_text(template, encoding="utf-8")
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "external_data_shards": args.external_data_shards,
                "transform": args.transform,
                "unsupported_generation_tags_remaining": any(
                    tag in template for tag in GENERATION_TAGS
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
