"""NumPy/Python references transcribed from the cited corto source functions."""

from __future__ import annotations

import numpy as np


def quantize(values, q, offset):
    # corto: include/corto/vertex_attribute.h GenericAttr::quantize
    return np.trunc(
        (np.asarray(values, dtype=np.float32) - np.asarray(offset, dtype=np.float32))
        / np.float32(q)
    ).astype(np.int32)


def needed(value):
    # corto: include/corto/cstream.h OutStream::needed
    value = int(value)
    if value == 0:
        return 0
    if value == -1:
        return 1
    if value < 0:
        value = -value - 1
    result = 2
    while value >> 1:
        value >>= 1
        result += 1
    return result


def pack_correlated(values):
    # corto: include/corto/cstream.h OutStream::encodeArray
    values = np.asarray(values, dtype=np.int32)
    logs = np.asarray(
        [max(needed(value) for value in row) for row in values], dtype=np.uint8
    )
    words = []
    accumulator = 0
    free = 32
    for row, width in zip(values, logs):
        width = int(width)
        if width == 0:
            continue
        middle = 1 << (width - 1)
        for signed in row:
            value = int(signed) + middle
            remaining = width
            if remaining >= free:
                accumulator = (
                    (accumulator << free) | (value >> (remaining - free))
                ) & 0xFFFFFFFF
                words.append(accumulator)
                remaining -= free
                value &= (1 << remaining) - 1 if remaining else 0
                free = 32
                accumulator = 0
            if remaining:
                accumulator = ((accumulator << remaining) | value) & 0xFFFFFFFF
                free -= remaining
    if free != 32:
        words.append((accumulator << free) & 0xFFFFFFFF)
    return np.asarray(words, dtype=np.uint32), logs


def normals_to_octa(normals, bits):
    # corto: include/corto/normal_attribute.h NormalAttr::toOcta
    normals = np.asarray(normals, dtype=np.float32)
    target = np.empty((len(normals), 2), dtype=np.int32)
    unit = np.float32(1 << (bits - 1))
    for i, (x, y, z) in enumerate(normals):
        length = np.float32(abs(x) + abs(y) + abs(z))
        if length == 0:
            target[i] = 0
            continue
        px, py = np.float32(x / length), np.float32(y / length)
        if z < 0:
            px, py = np.float32(1 - abs(py)), np.float32(1 - abs(px))
            if x < 0:
                px = -px
            if y < 0:
                py = -py
        target[i] = np.trunc([px * unit, py * unit]).astype(np.int32)
    return target


def octa_to_normals(octa, bits):
    # corto: include/corto/normal_attribute.h NormalAttr::toSphere
    octa = np.asarray(octa, dtype=np.int32)
    target = np.empty((len(octa), 3), dtype=np.float32)
    unit = np.float32(1 << (bits - 1))
    for i, (ivx, ivy) in enumerate(octa):
        vx, vy = np.float32(ivx), np.float32(ivy)
        x, y = vx, vy
        z = np.float32(unit - abs(vx) - abs(vy))
        if z < 0:
            x = np.float32((1 if vx > 0 else -1) * (unit - abs(vy)))
            y = np.float32((1 if vy > 0 else -1) * (unit - abs(vx)))
        length = np.float32(np.sqrt(x * x + y * y + z * z))
        target[i] = [x / length, y / length, z / length]
    return target


def colors_to_ycc(colors, steps):
    # corto: src/color_attribute.cpp ColorAttr::quantize
    colors = np.asarray(colors, dtype=np.uint8)
    steps = np.asarray(steps, dtype=np.uint8)
    result = np.empty_like(colors)
    for i, row in enumerate(colors):
        quantized = row.astype(np.uint16) // steps
        r, g, b = map(int, quantized[:3])
        result[i, 0] = g
        result[i, 1] = (b - g) & 255
        result[i, 2] = (r - g) & 255
        if colors.shape[1] == 4:
            result[i, 3] = quantized[3]
    return result


def estimate_normals(positions, faces, normalize=True):
    # corto: src/normal_attribute.cpp estimateNormals
    positions = np.asarray(positions, dtype=np.float32)
    result = np.zeros_like(positions)
    for face in np.asarray(faces, dtype=np.uint32):
        a, b, c = map(int, face)
        normal = np.cross(positions[b] - positions[a], positions[c] - positions[a])
        result[a] += normal
        result[b] += normal
        result[c] += normal
    if normalize:
        for i, normal in enumerate(result):
            length = np.float32(np.linalg.norm(normal))
            result[i] = (0, 0, 1) if length < 1e-5 else normal / length
    return result


def face_multiset(faces):
    return sorted(tuple(sorted(map(int, face))) for face in np.asarray(faces))


def edge_multiplicities(faces):
    counts = {}
    for face in np.asarray(faces):
        a, b, c = map(int, face)
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            counts[key] = counts.get(key, 0) + 1
    return sorted(counts.values())
