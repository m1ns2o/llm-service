#!/usr/bin/env python3
"""Convert asymmetric QMoE INT4 weights to WebGPU-compatible symmetric INT4.

ONNX Runtime WebGPU currently rejects QMoE nodes that provide per-block zero
points.  This tool preserves every non-QMoE tensor byte-for-byte, reconstructs
the already-quantized expert weights, requantizes those blocks around the
implicit WebGPU zero point (8), and removes only the QMoE zero-point inputs.
The source model and its external data files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import onnx


COPY_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ExpertTensorBundle:
    weight_name: str
    scale_name: str
    zero_point_name: str
    block_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES,
    )
    parser.add_argument("--alignment", type=int, default=64)
    parser.add_argument("--row-chunk", type=int, default=2048)
    parser.add_argument(
        "--strategy",
        choices=("best-mse", "shift", "rescale"),
        default="best-mse",
    )
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


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_exact(source: BinaryIO, destination: BinaryIO, length: int) -> str:
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


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Unpack low-nibble-first uint4 data along the final dimension."""
    unpacked = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    unpacked[..., 0::2] = packed & 0x0F
    unpacked[..., 1::2] = packed >> 4
    return unpacked


def pack_int4(unpacked: np.ndarray) -> np.ndarray:
    """Pack low-nibble-first uint4 data along the final dimension."""
    if unpacked.shape[-1] % 2:
        raise ValueError("uint4 input length must be even")
    return (
        unpacked[..., 0::2] | (unpacked[..., 1::2] << 4)
    ).astype(np.uint8, copy=False)


def symmetrize_q4_blocks(
    packed: np.ndarray,
    scales: np.ndarray,
    packed_zero_points: np.ndarray,
    strategy: str = "best-mse",
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Requantize packed asymmetric uint4 blocks around implicit zero point 8."""
    if packed.ndim != 3 or scales.ndim != 2 or packed_zero_points.ndim != 2:
        raise ValueError("expected packed[rows, blocks, bytes], scales/zp[rows, blocks]")
    rows, blocks, _ = packed.shape
    if scales.shape != (rows, blocks):
        raise ValueError("scale shape does not match packed blocks")

    zero_points = unpack_int4(packed_zero_points)[..., :blocks]
    codes = unpack_int4(packed)
    signed = codes.astype(np.int16) - zero_points[..., None].astype(np.int16)
    positive_max = np.max(np.maximum(signed, 0), axis=-1).astype(np.float32)
    negative_max = np.max(np.maximum(-signed, 0), axis=-1).astype(np.float32)
    scale_ratios = np.maximum(positive_max / 7.0, negative_max / 8.0)
    nonzero = scale_ratios > 0
    ratios = np.zeros_like(signed, dtype=np.float32)
    np.divide(
        signed.astype(np.float32),
        scale_ratios[..., None],
        out=ratios,
        where=nonzero[..., None],
    )
    source_scales = scales.astype(np.float32)
    rescaled_codes = np.clip(np.rint(ratios) + 8.0, 0, 15).astype(np.uint8)
    rescaled_codes[~nonzero] = 8
    rescaled_scales = np.where(
        nonzero,
        source_scales * scale_ratios,
        source_scales,
    ).astype(np.float16)
    original = signed.astype(np.float32) * source_scales[..., None]
    rescaled_reconstruction = (
        rescaled_codes.astype(np.float32) - 8.0
    ) * rescaled_scales.astype(np.float32)[..., None]

    shifted_codes = np.clip(signed + 8, 0, 15).astype(np.uint8)
    shifted_scales = scales.astype(np.float16, copy=False)
    shifted_reconstruction = (
        shifted_codes.astype(np.float32) - 8.0
    ) * shifted_scales.astype(np.float32)[..., None]

    if strategy == "rescale":
        choose_rescaled = np.ones(scales.shape, dtype=bool)
    elif strategy == "shift":
        choose_rescaled = np.zeros(scales.shape, dtype=bool)
    elif strategy == "best-mse":
        rescaled_error = np.sum(
            (rescaled_reconstruction - original) ** 2,
            axis=-1,
        )
        shifted_error = np.sum(
            (shifted_reconstruction - original) ** 2,
            axis=-1,
        )
        choose_rescaled = rescaled_error < shifted_error
    else:
        raise ValueError(f"unknown symmetrization strategy: {strategy}")

    symmetric_codes = np.where(
        choose_rescaled[..., None],
        rescaled_codes,
        shifted_codes,
    )
    symmetric_scales = np.where(
        choose_rescaled,
        rescaled_scales,
        shifted_scales,
    ).astype(np.float16)
    reconstructed = (
        symmetric_codes.astype(np.float32) - 8.0
    ) * symmetric_scales.astype(np.float32)[..., None]
    error = reconstructed - original
    metrics = {
        "value_count": float(error.size),
        "squared_error_sum": float(np.sum(error * error, dtype=np.float64)),
        "original_squared_sum": float(
            np.sum(original * original, dtype=np.float64)
        ),
        "absolute_error_sum": float(np.sum(np.abs(error), dtype=np.float64)),
        "max_absolute_error": float(np.max(np.abs(error), initial=0.0)),
        "rescaled_block_count": float(np.count_nonzero(choose_rescaled)),
        "shifted_block_count": float(choose_rescaled.size - np.count_nonzero(choose_rescaled)),
    }
    count = metrics["value_count"]
    metrics["mean_absolute_error"] = metrics["absolute_error_sum"] / count
    metrics["rmse"] = math.sqrt(metrics["squared_error_sum"] / count)
    metrics["relative_rmse"] = (
        math.sqrt(metrics["squared_error_sum"] / metrics["original_squared_sum"])
        if metrics["original_squared_sum"]
        else 0.0
    )
    return pack_int4(symmetric_codes), symmetric_scales, metrics


def _attribute_int(node: onnx.NodeProto, name: str) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(onnx.helper.get_attribute_value(attribute))
    raise ValueError(f"QMoE node {node.name!r} is missing {name!r}")


def collect_qmoe_bundles(model: onnx.ModelProto) -> list[ExpertTensorBundle]:
    bundles: list[ExpertTensorBundle] = []
    for node in model.graph.node:
        if node.op_type != "QMoE":
            continue
        if len(node.input) < 14:
            raise ValueError(f"QMoE node {node.name!r} has only {len(node.input)} inputs")
        if _attribute_int(node, "expert_weight_bits") != 4:
            raise ValueError(f"QMoE node {node.name!r} is not INT4")
        block_size = _attribute_int(node, "block_size")
        for weight_index, scale_index, zero_point_index in ((2, 3, 11), (5, 6, 12)):
            names = (
                node.input[weight_index],
                node.input[scale_index],
                node.input[zero_point_index],
            )
            if not all(names):
                raise ValueError(f"QMoE node {node.name!r} has incomplete quantization inputs")
            bundles.append(ExpertTensorBundle(*names, block_size=block_size))
            node.input[zero_point_index] = ""
    if not bundles:
        raise ValueError("model has no QMoE nodes")
    if len({bundle.weight_name for bundle in bundles}) != len(bundles):
        raise ValueError("QMoE expert weight initializer is reused unexpectedly")
    return bundles


def _tensor_source(
    tensor: onnx.TensorProto,
    source_dir: Path,
) -> tuple[Path, int, int]:
    fields = external_fields(tensor)
    if not fields:
        raise ValueError(f"tensor {tensor.name!r} is not external")
    source_path = (source_dir / fields["location"]).resolve()
    if source_path.parent != source_dir:
        raise ValueError(f"external path escapes source directory: {source_path}")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    return source_path, int(fields.get("offset", "0")), int(fields["length"])


def _memmap_tensor(
    tensor: onnx.TensorProto,
    source_dir: Path,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.memmap:
    path, offset, length = _tensor_source(tensor, source_dir)
    expected = math.prod(shape) * np.dtype(dtype).itemsize
    if length != expected:
        raise ValueError(
            f"tensor {tensor.name!r} length {length} does not match {expected}"
        )
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)


def _merge_metrics(target: dict[str, float], update: dict[str, float]) -> None:
    for key in (
        "value_count",
        "squared_error_sum",
        "original_squared_sum",
        "absolute_error_sum",
        "rescaled_block_count",
        "shifted_block_count",
    ):
        target[key] = target.get(key, 0.0) + update[key]
    target["max_absolute_error"] = max(
        target.get("max_absolute_error", 0.0),
        update["max_absolute_error"],
    )


def validate_webgpu_qmoe_graph(
    model_path: Path,
    *,
    expected_qmoe_nodes: int,
    removed_zero_points: set[str],
) -> dict[str, object]:
    """Validate graph invariants without rejecting ONNX Runtime custom operators."""
    model = onnx.load_model(model_path, load_external_data=False)
    qmoe_nodes = [node for node in model.graph.node if node.op_type == "QMoE"]
    if len(qmoe_nodes) != expected_qmoe_nodes:
        raise ValueError(
            f"expected {expected_qmoe_nodes} QMoE nodes, found {len(qmoe_nodes)}"
        )
    for node in qmoe_nodes:
        if len(node.input) < 14 or node.input[11] or node.input[12] or node.input[13]:
            raise ValueError(f"QMoE zero-point input remains in {node.name!r}")
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    remaining = removed_zero_points & initializer_names
    if remaining:
        raise ValueError(f"removed zero-point initializers remain: {sorted(remaining)}")
    for tensor in model.graph.initializer:
        if not tensor.external_data:
            continue
        fields = external_fields(tensor)
        path = (model_path.parent / fields["location"]).resolve()
        if path.parent != model_path.parent or not path.is_file():
            raise ValueError(f"invalid external tensor path for {tensor.name!r}: {path}")
        offset = int(fields.get("offset", "0"))
        length = int(fields["length"])
        if offset < 0 or length < 0 or offset + length > path.stat().st_size:
            raise ValueError(f"external tensor range is invalid for {tensor.name!r}")
    return {
        "qmoe_nodes": len(qmoe_nodes),
        "qmoe_zero_point_inputs_removed": True,
        "qmoe_zero_point_initializers_removed": len(removed_zero_points),
        "external_ranges_valid": True,
        "onnx_standard_checker": (
            "not_applicable: source graph uses ONNX Runtime custom operators"
        ),
    }


def symmetrize_weight(
    bundle: ExpertTensorBundle,
    initializers: dict[str, onnx.TensorProto],
    source_dir: Path,
    destination: BinaryIO,
    row_chunk: int,
    strategy: str,
) -> tuple[str, bytes, dict[str, float]]:
    weight = initializers[bundle.weight_name]
    scale = initializers[bundle.scale_name]
    zero_point = initializers[bundle.zero_point_name]
    if weight.data_type != onnx.TensorProto.UINT8:
        raise ValueError(f"{weight.name!r} is not uint8")
    if scale.data_type != onnx.TensorProto.FLOAT16:
        raise ValueError(f"{scale.name!r} is not float16")
    if zero_point.data_type != onnx.TensorProto.UINT8:
        raise ValueError(f"{zero_point.name!r} is not uint8")
    if len(weight.dims) != 3 or len(scale.dims) != 3 or len(zero_point.dims) != 3:
        raise ValueError(f"unexpected QMoE tensor rank for {weight.name!r}")

    experts, outputs, packed_bytes_per_row = map(int, weight.dims)
    rows = experts * outputs
    blocks = int(scale.dims[-1])
    packed_bytes_per_block = bundle.block_size // 2
    if packed_bytes_per_row != blocks * packed_bytes_per_block:
        raise ValueError(f"QMoE packed shape mismatch for {weight.name!r}")
    if tuple(map(int, scale.dims[:2])) != (experts, outputs):
        raise ValueError(f"QMoE scale shape mismatch for {weight.name!r}")
    expected_zp_bytes = (blocks + 1) // 2
    if tuple(map(int, zero_point.dims)) != (experts, outputs, expected_zp_bytes):
        raise ValueError(f"QMoE zero-point shape mismatch for {weight.name!r}")

    packed_map = _memmap_tensor(
        weight,
        source_dir,
        np.uint8,
        (rows, blocks, packed_bytes_per_block),
    )
    scale_map = _memmap_tensor(scale, source_dir, np.dtype("<f2"), (rows, blocks))
    zp_map = _memmap_tensor(
        zero_point,
        source_dir,
        np.uint8,
        (rows, expected_zp_bytes),
    )
    new_scales = np.empty((rows, blocks), dtype=np.float16)
    digest = hashlib.sha256()
    metrics: dict[str, float] = {}
    for start in range(0, rows, row_chunk):
        stop = min(rows, start + row_chunk)
        new_packed, chunk_scales, chunk_metrics = symmetrize_q4_blocks(
            np.asarray(packed_map[start:stop]),
            np.asarray(scale_map[start:stop]),
            np.asarray(zp_map[start:stop]),
            strategy,
        )
        block = new_packed.tobytes(order="C")
        destination.write(block)
        digest.update(block)
        new_scales[start:stop] = chunk_scales
        _merge_metrics(metrics, chunk_metrics)

    count = metrics["value_count"]
    metrics["mean_absolute_error"] = metrics["absolute_error_sum"] / count
    metrics["rmse"] = math.sqrt(metrics["squared_error_sum"] / count)
    original_squared_sum = metrics["original_squared_sum"]
    metrics["relative_rmse"] = (
        math.sqrt(metrics["squared_error_sum"] / original_squared_sum)
        if original_squared_sum
        else 0.0
    )
    return digest.hexdigest(), new_scales.tobytes(order="C"), metrics


def main() -> int:
    args = parse_args()
    source_model = args.source_model.resolve()
    source_dir = source_model.parent
    output_dir = args.output_dir.resolve()
    if not source_model.is_file():
        raise SystemExit(f"source model not found: {source_model}")
    if args.max_shard_bytes <= 0 or args.alignment <= 0 or args.row_chunk <= 0:
        raise SystemExit("shard size, alignment, and row chunk must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    incomplete = output_dir / "INCOMPLETE"
    incomplete.write_text("symmetrizing QMoE INT4 weights\n", encoding="utf-8")

    model = onnx.load_model(source_model, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    bundles = collect_qmoe_bundles(model)
    for bundle in bundles:
        for name in (bundle.weight_name, bundle.scale_name, bundle.zero_point_name):
            if name not in initializers:
                raise ValueError(f"missing QMoE initializer {name!r}")

    removed_zero_points = {bundle.zero_point_name for bundle in bundles}
    retained = [
        tensor
        for tensor in model.graph.initializer
        if tensor.name not in removed_zero_points
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)

    bundle_by_weight = {bundle.weight_name: bundle for bundle in bundles}
    weight_by_scale = {bundle.scale_name: bundle.weight_name for bundle in bundles}
    scale_cache: dict[str, bytes] = {}
    transform_metrics: dict[str, dict[str, float]] = {}
    records: list[dict[str, object]] = []
    output_paths: list[Path] = []
    shard_index = -1
    shard_offset = 0
    destination: BinaryIO | None = None

    with ExitStack() as stack:
        sources: dict[Path, BinaryIO] = {}

        def source_stream(path: Path) -> BinaryIO:
            if path not in sources:
                sources[path] = stack.enter_context(path.open("rb"))
            return sources[path]

        for tensor in model.graph.initializer:
            if not tensor.external_data:
                continue
            source_path, source_offset, source_length = _tensor_source(tensor, source_dir)
            length = source_length
            if tensor.name in scale_cache:
                length = len(scale_cache[tensor.name])
            if length > args.max_shard_bytes:
                raise ValueError(
                    f"tensor {tensor.name!r} is {length} bytes, larger than shard limit"
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

            transform = "byte_copy"
            if tensor.name in bundle_by_weight:
                bundle = bundle_by_weight[tensor.name]
                digest, new_scale_bytes, metrics = symmetrize_weight(
                    bundle,
                    initializers,
                    source_dir,
                    destination,
                    args.row_chunk,
                    args.strategy,
                )
                scale_cache[bundle.scale_name] = new_scale_bytes
                transform_metrics[tensor.name] = metrics
                transform = "asymmetric_q4_to_symmetric_q4_zp8"
            elif tensor.name in weight_by_scale:
                data = scale_cache.pop(tensor.name)
                destination.write(data)
                digest = hashlib.sha256(data).hexdigest()
                transform = "qmoe_symmetric_scale"
            else:
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
                    "transform": transform,
                    "source": {
                        "file": source_path.name,
                        "offset": source_offset,
                        "length": source_length,
                    },
                    "output": {
                        "file": output_name,
                        "offset": destination_offset,
                    },
                }
            )
        if destination is not None:
            destination.close()
    if scale_cache:
        raise ValueError(f"unwritten transformed scales: {sorted(scale_cache)}")

    output_model = output_dir / source_model.name
    onnx.save_model(model, output_model)
    graph_validation = validate_webgpu_qmoe_graph(
        output_model,
        expected_qmoe_nodes=len(bundles) // 2,
        removed_zero_points=removed_zero_points,
    )

    if args.verify:
        output_streams = {path.name: path.open("rb") for path in output_paths}
        try:
            for record in records:
                stream = output_streams[record["output"]["file"]]
                stream.seek(record["output"]["offset"])
                digest = hashlib.sha256()
                remaining = int(record["length"])
                while remaining:
                    block = stream.read(min(COPY_CHUNK_BYTES, remaining))
                    if not block:
                        raise EOFError(f"repacked tensor {record['tensor']!r} ended early")
                    digest.update(block)
                    remaining -= len(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"repacked tensor mismatch: {record['tensor']}")
        finally:
            for stream in output_streams.values():
                stream.close()

    total_error = {
        key: 0.0
        for key in (
            "value_count",
            "squared_error_sum",
            "original_squared_sum",
            "absolute_error_sum",
            "rescaled_block_count",
            "shifted_block_count",
            "max_absolute_error",
        )
    }
    for metrics in transform_metrics.values():
        _merge_metrics(total_error, metrics)
    count = total_error["value_count"]
    total_error["mean_absolute_error"] = total_error["absolute_error_sum"] / count
    total_error["rmse"] = math.sqrt(total_error["squared_error_sum"] / count)
    total_error["relative_rmse"] = math.sqrt(
        total_error["squared_error_sum"] / total_error["original_squared_sum"]
    )

    manifest = {
        "schema_version": 1,
        "operation": "QMoE asymmetric INT4 to symmetric INT4 with implicit zero point 8",
        "strategy": args.strategy,
        "source_model": str(source_model),
        "output_model": str(output_model),
        "max_shard_bytes": args.max_shard_bytes,
        "alignment": args.alignment,
        "qmoe_tensor_count": len(bundles),
        "removed_zero_point_tensor_count": len(removed_zero_points),
        "external_tensor_count": len(records),
        "external_tensor_bytes": sum(int(record["length"]) for record in records),
        "non_qmoe_tensor_bytes_preserved": True,
        "graph_validation": graph_validation,
        "qmoe_error": total_error,
        "qmoe_tensors": transform_metrics,
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
    (output_dir / "symmetrize_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    incomplete.unlink()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
