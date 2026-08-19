from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/symmetrize_qmoe_onnx.py"
SPEC = importlib.util.spec_from_file_location("symmetrize_qmoe_onnx", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Lfm25SymmetricQmoeTests(unittest.TestCase):
    def test_symmetrize_reconstructs_blocks_around_implicit_zero_point(self):
        codes = np.array(
            [
                [0, 3, 5, 7, 9, 12, 15, 6],
                [4, 5, 6, 7, 8, 9, 10, 11],
            ],
            dtype=np.uint8,
        ).reshape(1, 2, 8)
        packed = MODULE.pack_int4(codes)
        scales = np.array([[0.25, 0.5]], dtype=np.float16)
        zero_points = MODULE.pack_int4(np.array([[5, 9]], dtype=np.uint8))

        output, output_scales, metrics = MODULE.symmetrize_q4_blocks(
            packed,
            scales,
            zero_points,
        )

        output_codes = MODULE.unpack_int4(output)
        original = (codes.astype(np.float32) - np.array([[[5], [9]]])) * scales[..., None]
        reconstructed = (
            output_codes.astype(np.float32) - 8
        ) * output_scales.astype(np.float32)[..., None]
        self.assertTrue(np.all((0 <= output_codes) & (output_codes <= 15)))
        self.assertLess(np.max(np.abs(reconstructed - original)), 0.5)
        self.assertGreater(metrics["value_count"], 0)
        self.assertGreaterEqual(metrics["relative_rmse"], 0)

    def test_zero_block_remains_exactly_zero(self):
        codes = np.full((1, 1, 32), 11, dtype=np.uint8)
        packed = MODULE.pack_int4(codes)
        scales = np.array([[0.125]], dtype=np.float16)
        zero_points = MODULE.pack_int4(np.array([[11, 0]], dtype=np.uint8))

        output, output_scales, metrics = MODULE.symmetrize_q4_blocks(
            packed,
            scales,
            zero_points,
        )

        np.testing.assert_array_equal(MODULE.unpack_int4(output), 8)
        np.testing.assert_array_equal(output_scales, scales)
        self.assertEqual(metrics["max_absolute_error"], 0.0)

    def test_best_mse_never_exceeds_shift_or_rescale_error(self):
        rng = np.random.default_rng(7)
        codes = rng.integers(0, 16, size=(3, 5, 32), dtype=np.uint8)
        zero_values = rng.integers(3, 13, size=(3, 6), dtype=np.uint8)
        packed = MODULE.pack_int4(codes)
        packed_zero_points = MODULE.pack_int4(zero_values)
        scales = rng.uniform(0.001, 0.2, size=(3, 5)).astype(np.float16)

        _, _, best = MODULE.symmetrize_q4_blocks(
            packed, scales, packed_zero_points, "best-mse"
        )
        _, _, shift = MODULE.symmetrize_q4_blocks(
            packed, scales, packed_zero_points, "shift"
        )
        _, _, rescale = MODULE.symmetrize_q4_blocks(
            packed, scales, packed_zero_points, "rescale"
        )

        self.assertLessEqual(best["squared_error_sum"], shift["squared_error_sum"])
        self.assertLessEqual(best["squared_error_sum"], rescale["squared_error_sum"])


if __name__ == "__main__":
    unittest.main()
