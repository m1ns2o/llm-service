#!/usr/bin/env python3
"""Repack an ONNX model's external tensors into browser-sized shards.

Tensor bytes and tensor metadata are preserved. Only each initializer's external
file location and offset are rewritten. This avoids multi-gigabyte ArrayBuffer
allocations in browser ONNX loaders without requantizing the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

import onnx


COPY_CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument("--alignment", type=int, default=64)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def external_fields(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def set_external_fields(
    tensor: onnx.TensorProto,
    *,
    location: str,
    offset: int,
    length: int,
) -> None:
    del tensor.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value
    tensor.data_location = onnx.TensorProto.EXTERNAL


def shard_name(model_name: str, index: int) -> str:
    suffix = "" if index == 0 else f"_{index}"
    return f"{model_name}_data{suffix}"


def copy_exact(source, destination, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not block:
            raise EOFError(f"external tensor ended {remaining} bytes early")
        destination.write(block)
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def hash_exact(source, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        block = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not block:
            raise EOFError(f"repacked tensor ended {remaining} bytes early")
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source_model = args.source_model.resolve()
    source_dir = source_model.parent
    output_dir = args.output_dir.resolve()
    if args.max_shard_bytes <= 0:
        raise SystemExit("--max-shard-bytes must be positive")
    if args.alignment <= 0:
        raise SystemExit("--alignment must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / "INCOMPLETE"
    incomplete.write_text("repacking\n", encoding="utf-8")

    model = onnx.load_model(source_model, load_external_data=False)
    external_tensors = [
        tensor for tensor in model.graph.initializer if tensor.external_data
    ]
    if not external_tensors:
        raise SystemExit("model has no external initializers")

    records: list[dict[str, object]] = []
    output_paths: list[Path] = []
    shard_index = -1
    shard_offset = 0
    destination = None

    with ExitStack() as stack:
        sources: dict[Path, object] = {}

        def source_stream(path: Path):
            if path not in sources:
                if not path.is_file() or path.parent != source_dir:
                    raise FileNotFoundError(f"unsafe or missing external data: {path}")
                sources[path] = stack.enter_context(path.open("rb"))
            return sources[path]

        for tensor in external_tensors:
            fields = external_fields(tensor)
            length = int(fields["length"])
            source_offset = int(fields.get("offset", "0"))
            source_path = (source_dir / fields["location"]).resolve()
            if source_path.parent != source_dir:
                raise ValueError(f"external path escapes source directory: {source_path}")
            if length > args.max_shard_bytes:
                raise ValueError(
                    f"tensor {tensor.name!r} is {length} bytes, larger than the shard limit"
                )

            aligned = (
                (shard_offset + args.alignment - 1) // args.alignment
            ) * args.alignment
            if destination is None or (
                shard_offset > 0 and aligned + length > args.max_shard_bytes
            ):
                if destination is not None:
                    destination.close()
                shard_index += 1
                output_path = output_dir / shard_name(source_model.name, shard_index)
                destination = output_path.open("wb")
                output_paths.append(output_path)
                shard_offset = 0
                aligned = 0

            if aligned > shard_offset:
                destination.write(b"\0" * (aligned - shard_offset))
            destination.seek(aligned)
            source = source_stream(source_path)
            source.seek(source_offset)
            digest = copy_exact(source, destination, length)
            destination_offset = aligned
            shard_offset = destination_offset + length
            output_name = output_paths[-1].name
            set_external_fields(
                tensor,
                location=output_name,
                offset=destination_offset,
                length=length,
            )
            records.append(
                {
                    "tensor": tensor.name,
                    "length": length,
                    "sha256": digest,
                    "source": {
                        "file": source_path.name,
                        "offset": source_offset,
                    },
                    "output": {
                        "file": output_name,
                        "offset": destination_offset,
                    },
                }
            )
        if destination is not None:
            destination.close()

    output_model = output_dir / source_model.name
    onnx.save_model(model, output_model)

    if args.verify:
        output_streams = {
            path.name: path.open("rb") for path in output_paths
        }
        try:
            for record in records:
                output = record["output"]
                stream = output_streams[output["file"]]
                stream.seek(output["offset"])
                actual = hash_exact(stream, record["length"])
                if actual != record["sha256"]:
                    raise ValueError(f"repacked tensor mismatch: {record['tensor']}")
        finally:
            for stream in output_streams.values():
                stream.close()

    manifest = {
        "schema_version": 1,
        "source_model": str(source_model),
        "output_model": str(output_model),
        "max_shard_bytes": args.max_shard_bytes,
        "alignment": args.alignment,
        "external_tensor_count": len(records),
        "external_tensor_bytes": sum(record["length"] for record in records),
        "shard_count": len(output_paths),
        "shards": [
            {
                "file": path.name,
                "size": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in output_paths
        ],
        "tensor_bytes_verified": bool(args.verify),
    }
    (output_dir / "repack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    incomplete.unlink()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
