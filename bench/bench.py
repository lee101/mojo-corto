"""Locked benchmarks against source-derived NumPy/Python references."""

from __future__ import annotations

import os
import platform
import struct
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mojo_corto as corto  # noqa: E402
import reference  # noqa: E402
from mojo_corto.codec import _dictionary  # noqa: E402


def best(function, repetitions=3):
    result = None
    elapsed = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        elapsed = min(elapsed, time.perf_counter() - start)
    return elapsed, result


def vector_octa(normals, bits=10):
    length = np.abs(normals).sum(axis=1)
    safe = np.where(length == 0, 1, length)
    projected = normals[:, :2] / safe[:, None]
    lower = normals[:, 2] < 0
    folded = 1 - np.abs(projected[lower, ::-1])
    folded *= np.where(normals[lower, :2] < 0, -1, 1)
    projected[lower] = folded
    projected[length == 0] = 0
    return np.trunc(projected * (1 << (bits - 1))).astype(np.int32)


def python_tunstall_decode(payload):
    symbol_count = payload[0]
    pair_end = 1 + 2 * symbol_count
    pairs = np.frombuffer(payload[1:pair_end], dtype=np.uint8).reshape(-1, 2)
    size, encoded_size = struct.unpack_from("<II", payload, pair_end)
    if symbol_count == 1:
        return bytes((int(pairs[0, 0]),)) * size
    index, lengths, table = _dictionary(pairs[:, 0], pairs[:, 1])
    encoded = payload[pair_end + 8 : pair_end + 8 + encoded_size]
    target = bytearray()
    for code in encoded:
        start = int(index[code])
        target.extend(table[start : start + int(lengths[code])])
    return bytes(target[:size])


def grid_faces(side):
    y, x = np.mgrid[: side - 1, : side - 1]
    a = (y * side + x).ravel()
    b = a + 1
    c = a + side
    d = c + 1
    return np.column_stack(
        (
            np.concatenate((a, c)),
            np.concatenate((b, b)),
            np.concatenate((c, d)),
        )
    ).astype(np.uint32)


def numpy_edge_incidence(faces):
    edges = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    rng = np.random.default_rng(42)
    rows = []

    values = rng.normal(size=(2_000_000, 3)).astype(np.float32)
    offset = np.array([0.25, -0.5, 1.0], np.float32)
    mojo_time, mojo_quantized = best(
        lambda: corto.quantize(values, 0.0025, offset=offset)
    )
    ref_time, ref_quantized = best(
        lambda: np.trunc((values - offset) / np.float32(0.0025)).astype(np.int32)
    )
    np.testing.assert_array_equal(mojo_quantized, ref_quantized)
    rows.append(("quantize f32, 2M x 3", mojo_time, ref_time, "NumPy"))

    normals = rng.normal(size=(1_500_000, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    mojo_time, mojo_octa = best(lambda: corto.normals_to_octa(normals, 10))
    ref_time, ref_octa = best(lambda: vector_octa(normals, 10))
    np.testing.assert_array_equal(mojo_octa, ref_octa)
    rows.append(("octa encode, 1.5M", mojo_time, ref_time, "NumPy"))

    integers = rng.integers(-31, 32, size=(250_000, 3), dtype=np.int32)
    mojo_time, packed = best(lambda: corto.pack_correlated(integers))
    ref_time, reference_packed = best(
        lambda: reference.pack_correlated(integers), repetitions=1
    )
    np.testing.assert_array_equal(packed.words, reference_packed[0])
    rows.append(("correlated bit-pack, 250K x 3", mojo_time, ref_time, "Python port"))

    entropy_data = np.tile(
        np.array([0, 0, 0, 0, 1, 0, 0, 2, 0, 3], np.uint8), 200_000
    )
    entropy_block = corto.tunstall_compress(entropy_data)
    mojo_time, mojo_decoded = best(lambda: corto.tunstall_decompress(entropy_block))
    ref_time, ref_decoded = best(
        lambda: python_tunstall_decode(entropy_block.payload), repetitions=2
    )
    assert mojo_decoded == ref_decoded == entropy_data.tobytes()
    rows.append(("Tunstall decode, 2MB", mojo_time, ref_time, "Python port"))

    faces = grid_faces(300)
    mojo_time, connectivity = best(
        lambda: corto.encode_connectivity(faces, 300 * 300)
    )
    ref_time, incidence = best(lambda: numpy_edge_incidence(faces))
    assert connectivity.face_count == faces.shape[0]
    assert incidence[0].shape[0] > 0
    rows.append(
        ("connectivity, 178802 faces", mojo_time, ref_time, "NumPy edge incidence")
    )

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| Kernel | Mojo | Reference | Speedup | Reference |")
    print("|---|---:|---:|---:|---|")
    for name, mojo_time, ref_time, reference_name in rows:
        speedup = ref_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1000:.3f} ms | "
            f"{ref_time * 1000:.3f} ms | {speedup:.2f}x | {reference_name} |"
        )


if __name__ == "__main__":
    main()
