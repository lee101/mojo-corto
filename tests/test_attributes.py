import numpy as np
import pytest

import mojo_corto as corto
import reference


def test_quantization_matches_generic_attr():
    values = np.array(
        [[1.99, -1.99, 0.0], [2.01, -2.01, 9.5], [-0.1, 0.1, -9.5]],
        dtype=np.float32,
    )
    offset = np.array([1.0, -1.0, 0.25], dtype=np.float32)
    actual = corto.quantize(values, 0.5, offset=offset)
    expected = reference.quantize(values, 0.5, offset)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(
        corto.dequantize(actual, 0.5, offset=offset),
        actual * np.float32(0.5) + offset,
    )


def test_quantization_empty_and_validation():
    assert corto.quantize(np.empty((0, 3), np.float32), 0.1).shape == (0, 3)
    with pytest.raises(ValueError):
        corto.quantize([[1, 2, 3]], 0)
    with pytest.raises(TypeError, match="safely convert"):
        corto.quantize(np.ones((1, 3), dtype=np.float64), 0.1)
    with pytest.raises(ValueError, match="finite"):
        corto.quantize(np.array([[np.nan, 0, 0]], np.float32), 0.1)
    with pytest.raises(OverflowError):
        corto.quantize(np.array([[3e38]], np.float32), 1e-20)


def test_quantization_simd_tail_matches_generic_attr():
    values = np.linspace(-7.25, 8.5, 21, dtype=np.float32).reshape(7, 3)
    offset = np.array([0.125, -0.75, 1.5], dtype=np.float32)
    np.testing.assert_array_equal(
        corto.quantize(values, 0.0625, offset=offset),
        reference.quantize(values, 0.0625, offset),
    )


def test_quantization_parallel_path_matches_generic_attr():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(400_000, 3)).astype(np.float32)
    offset = np.array([0.25, -0.5, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(
        corto.quantize(values, 0.0025, offset=offset),
        reference.quantize(values, 0.0025, offset),
    )


def test_quantization_parallel_path_reports_validation_errors():
    values = np.zeros((400_000, 3), dtype=np.float32)
    values[-1, 0] = np.float32(3e38)
    with pytest.raises(OverflowError):
        corto.quantize(values, 1e-20)

    values[-1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        corto.quantize(values, 1e-20)


def test_quantization_step_matches_encoder():
    points = np.array([[-2, 1, 7], [6, 3, -1], [2, -5, 3]], np.float32)
    assert corto.quantization_step(points, 4) == pytest.approx(8 / 16)
    assert corto.quantization_step(np.empty((0, 3), np.float32)) == 0


def test_point_cloud_delta_roundtrip():
    rng = np.random.default_rng(4)
    values = rng.integers(-10000, 10000, size=(1000, 5), dtype=np.int32)
    diffs = corto.point_cloud_delta_encode(values)
    np.testing.assert_array_equal(corto.point_cloud_delta_decode(diffs), values)


@pytest.mark.parametrize("parallelogram", [False, True])
def test_mesh_delta_matches_generic_attr(parallelogram):
    values = np.array(
        [[10, 3], [12, 8], [20, 9], [25, 18], [40, 7]], dtype=np.int32
    )
    quads = np.array(
        [[0, 0, 0, 0], [1, 0, 0, 0], [2, 1, 0, 0], [3, 2, 1, 0]],
        dtype=np.int32,
    )
    diffs = corto.delta_encode(values, quads, parallelogram=parallelogram)
    expected = np.empty_like(diffs)
    expected[0] = values[0]
    for i, (target, a, b, c) in enumerate(quads[1:], 1):
        predictor = values[a]
        if parallelogram and a != b:
            predictor = values[a] + values[b] - values[c]
        expected[i] = values[target] - predictor
    np.testing.assert_array_equal(diffs, expected)
    decode_context = quads[:, 1:].copy()
    np.testing.assert_array_equal(
        corto.delta_decode(diffs, decode_context, parallelogram=parallelogram),
        values[: len(quads)],
    )


def test_correlated_bitstream_exact_and_roundtrip():
    values = np.array(
        [[0, -1, 0], [3, -4, 1], [100, 2, -128], [32767, -32768, 7]],
        dtype=np.int32,
    )
    packed = corto.pack_correlated(values)
    words, logs = reference.pack_correlated(values)
    np.testing.assert_array_equal(packed.logs, logs)
    np.testing.assert_array_equal(packed.words, words)
    np.testing.assert_array_equal(corto.unpack_correlated(packed), values)


def test_correlated_bitstream_empty_and_zero():
    empty = corto.pack_correlated(np.empty((0, 3), np.int32))
    assert empty.words.size == empty.logs.size == 0
    zeros = np.zeros((17, 4), dtype=np.int32)
    np.testing.assert_array_equal(corto.unpack_correlated(corto.pack_correlated(zeros)), zeros)


def test_correlated_unpack_rejects_invalid_or_truncated_streams():
    with pytest.raises(ValueError, match="bit widths"):
        corto.unpack_correlated(
            corto.PackedArray(np.zeros(1, np.uint32), np.array([33], np.uint8), 1)
        )
    with pytest.raises(ValueError, match="end before"):
        corto.unpack_correlated(
            corto.PackedArray(np.empty(0, np.uint32), np.array([1], np.uint8), 1)
        )
    with pytest.raises(ValueError, match="trailing"):
        corto.unpack_correlated(
            corto.PackedArray(np.array([0, 0], np.uint32), np.array([1], np.uint8), 1)
        )
