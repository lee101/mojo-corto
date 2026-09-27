"""Public NumPy API for the ported corto kernels."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np

from ._lib import address, lib


def _array(value, dtype, *, ndim: int | None = None) -> np.ndarray:
    requested = np.dtype(dtype)
    if isinstance(value, np.ndarray) and not np.can_cast(
        value.dtype, requested, casting="safe"
    ):
        raise TypeError(
            f"cannot safely convert ndarray from {value.dtype} to {requested}"
        )
    array = np.ascontiguousarray(value, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected a {ndim}-D array, got shape {array.shape}")
    return array


def quantization_step(positions, bits: int = 14) -> float:
    """Return corto's bounding-box quantization step for a bit budget."""
    points = _array(positions, np.float32, ndim=2)
    if points.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    if bits < 1 or bits > 30:
        raise ValueError("bits must be between 1 and 30")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    if points.shape[0] == 0:
        return 0.0
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.float32(extent.max() / np.float32(2.0**bits)))


def quantize(
    values,
    q: float,
    *,
    offset=None,
) -> np.ndarray:
    """Quantize float32 attributes using corto's truncation toward zero."""
    source = _array(values, np.float32, ndim=2)
    q32 = np.float32(q)
    if not np.isfinite(q32) or q32 == 0:
        raise ValueError("q must be finite and non-zero")
    components = source.shape[1]
    offsets = (
        np.zeros(components, dtype=np.float32)
        if offset is None
        else _array(offset, np.float32).reshape(-1)
    )
    if offsets.size != components:
        raise ValueError("offset must have one value per component")
    if not np.all(np.isfinite(offsets)):
        raise ValueError("values and offset must be finite")
    target = np.empty(source.shape, dtype=np.int32)
    if source.shape[0]:
        statuses = np.empty(16, dtype=np.int32)
        status = lib().corto_quantize_f32(
            address(source),
            address(target),
            source.shape[0],
            components,
            q32,
            address(offsets),
            address(statuses),
        )
        if status == 1:
            raise ValueError("values and offset must be finite")
        if status == 2:
            raise OverflowError("quantized values do not fit in int32")
    return target


def dequantize(values, q: float, *, offset=None) -> np.ndarray:
    source = _array(values, np.int32, ndim=2)
    components = source.shape[1]
    offsets = (
        np.zeros(components, dtype=np.float32)
        if offset is None
        else _array(offset, np.float32).reshape(-1)
    )
    if offsets.size != components:
        raise ValueError("offset must have one value per component")
    q32 = np.float32(q)
    if not np.isfinite(q32) or not np.all(np.isfinite(offsets)):
        raise ValueError("q and offset must be finite")
    target = np.empty(source.shape, dtype=np.float32)
    if source.shape[0]:
        lib().corto_dequantize_f32(
            address(source),
            address(target),
            source.shape[0],
            components,
            q32,
            address(offsets),
        )
    return target


def point_cloud_delta_encode(values) -> np.ndarray:
    source = _array(values, np.int32, ndim=2)
    target = source.copy()
    if source.shape[0] > 1:
        target[1:] = source[1:] - source[:-1]
    return target


def point_cloud_delta_decode(diffs) -> np.ndarray:
    values = _array(diffs, np.int32, ndim=2).copy()
    if values.shape[0]:
        dummy = np.empty(1, dtype=np.int32)
        lib().corto_delta_decode(
            address(values),
            address(dummy),
            values.shape[0],
            values.shape[1],
            0,
            1,
        )
    return values


def delta_encode(values, context, *, parallelogram: bool = True) -> np.ndarray:
    """Reorder and predict an integer attribute with corto Quad contexts."""
    source = _array(values, np.int32, ndim=2)
    quads = _array(context, np.int32, ndim=2)
    if quads.shape[1] != 4:
        raise ValueError("context must have columns (target, a, b, c)")
    if quads.size and (
        np.any(quads < 0) or np.any(quads >= source.shape[0])
    ):
        raise ValueError("context contains an out-of-range vertex index")
    target = np.empty((quads.shape[0], source.shape[1]), dtype=np.int32)
    if quads.shape[0]:
        lib().corto_delta_encode(
            address(source),
            address(quads),
            address(target),
            quads.shape[0],
            source.shape[1],
            int(parallelogram),
        )
    return target


def delta_decode(diffs, context, *, parallelogram: bool = True) -> np.ndarray:
    """Undo prediction using corto Face contexts in encoded vertex order."""
    values = _array(diffs, np.int32, ndim=2).copy()
    faces = _array(context, np.int32, ndim=2)
    if faces.shape != (values.shape[0], 3):
        raise ValueError("context must have shape (len(diffs), 3)")
    if faces.shape[0] > 1 and (
        np.any(faces[1:] < 0) or np.any(faces[1:] >= values.shape[0])
    ):
        raise ValueError("context contains an out-of-range prediction index")
    if values.shape[0]:
        lib().corto_delta_decode(
            address(values),
            address(faces),
            values.shape[0],
            values.shape[1],
            int(parallelogram),
            0,
        )
    return values


@dataclass(frozen=True)
class PackedArray:
    """Corto's correlated signed-integer bitstream before entropy coding logs."""

    words: np.ndarray
    logs: np.ndarray
    components: int


def pack_correlated(values) -> PackedArray:
    source = _array(values, np.int32, ndim=2)
    count, components = source.shape
    if components <= 0:
        raise ValueError("values must have at least one component")
    logs = np.empty(count, dtype=np.uint8)
    words = np.empty(max(1, count * components), dtype=np.uint32)
    if count:
        used = int(
            lib().corto_pack_correlated(
                address(source),
                count,
                components,
                address(logs),
                address(words),
            )
        )
    else:
        used = 0
    return PackedArray(words[:used].copy(), logs, components)


def unpack_correlated(packed: PackedArray) -> np.ndarray:
    logs = _array(packed.logs, np.uint8).reshape(-1)
    words = _array(packed.words, np.uint32).reshape(-1)
    if not isinstance(packed.components, (int, np.integer)) or packed.components <= 0:
        raise ValueError("components must be a positive integer")
    if np.any(logs > 32):
        raise ValueError("correlated bit widths must be at most 32")
    target = np.empty((logs.size, int(packed.components)), dtype=np.int32)
    if logs.size:
        if not words.size and np.any(logs):
            raise ValueError("packed words end before the declared bit widths")
        safe_words = words if words.size else np.zeros(1, dtype=np.uint32)
        consumed = int(lib().corto_unpack_correlated(
            address(safe_words),
            words.size,
            address(logs),
            address(target),
            logs.size,
            packed.components,
        ))
        if consumed < 0:
            raise ValueError("packed words end before the declared bit widths")
        if consumed != words.size:
            raise ValueError("packed words contain trailing data")
    return target


def normals_to_octa(normals, bits: int = 10) -> np.ndarray:
    source = _array(normals, np.float32, ndim=2)
    if source.shape[1] != 3:
        raise ValueError("normals must have shape (n, 3)")
    if bits < 2 or bits > 30:
        raise ValueError("bits must be between 2 and 30")
    target = np.empty((source.shape[0], 2), dtype=np.int32)
    if source.shape[0]:
        lib().corto_normals_to_octa(
            address(source), address(target), source.shape[0], bits
        )
    return target


def octa_to_normals(octa, bits: int = 10) -> np.ndarray:
    source = _array(octa, np.int32, ndim=2)
    if source.shape[1] != 2:
        raise ValueError("octa values must have shape (n, 2)")
    if bits < 2 or bits > 30:
        raise ValueError("bits must be between 2 and 30")
    target = np.empty((source.shape[0], 3), dtype=np.float32)
    if source.shape[0]:
        lib().corto_octa_to_normals(
            address(source), address(target), source.shape[0], bits
        )
    return target


def estimate_normals(positions, faces, *, normalize: bool = True) -> np.ndarray:
    points = _array(positions, np.float32, ndim=2)
    triangles = _array(faces, np.uint32, ndim=2)
    if points.shape[1] != 3 or triangles.shape[1] != 3:
        raise ValueError("positions and faces must have shapes (n, 3), (m, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions must be finite")
    if triangles.size and np.any(triangles >= points.shape[0]):
        raise ValueError("faces contain an out-of-range vertex index")
    target = np.empty_like(points)
    if points.shape[0]:
        lib().corto_estimate_normals(
            address(points),
            address(triangles),
            address(target),
            points.shape[0],
            triangles.shape[0],
            int(normalize),
        )
    return target


def color_steps(bits=(6, 7, 6, 5)) -> np.ndarray:
    values = np.asarray(bits, dtype=np.int32).reshape(-1)
    if values.size not in (3, 4) or np.any((values < 1) | (values > 8)):
        raise ValueError("bits must contain three or four values in [1, 8]")
    return np.asarray(1 << (8 - values), dtype=np.uint8)


def colors_to_ycc(colors, bits=(6, 7, 6, 5)) -> tuple[np.ndarray, np.ndarray]:
    source = _array(colors, np.uint8, ndim=2)
    if source.shape[1] not in (3, 4):
        raise ValueError("colors must have three or four components")
    steps = color_steps(bits)
    if steps.size != source.shape[1]:
        raise ValueError("bits must match the color component count")
    target = np.empty_like(source)
    if source.shape[0]:
        lib().corto_colors_to_ycc(
            address(source),
            address(target),
            source.shape[0],
            source.shape[1],
            address(steps),
        )
    return target, steps


def ycc_to_colors(ycc, steps, *, components: int | None = None) -> np.ndarray:
    source = _array(ycc, np.uint8, ndim=2)
    quantization = _array(steps, np.uint8).reshape(-1)
    target_components = source.shape[1] if components is None else components
    if target_components not in (3, 4) or quantization.size < target_components:
        raise ValueError("components and steps must describe RGB or RGBA")
    if np.any(quantization[:target_components] == 0):
        raise ValueError("steps must be non-zero")
    target = np.empty((source.shape[0], target_components), dtype=np.uint8)
    if source.shape[0]:
        lib().corto_ycc_to_colors(
            address(source),
            address(target),
            source.shape[0],
            source.shape[1],
            target_components,
            address(quantization),
        )
    return target


@dataclass(frozen=True)
class TunstallBlock:
    """A serialized corto Tunstall stream and its uncompressed size."""

    payload: bytes
    size: int

    def to_bytes(self) -> bytes:
        return self.payload


def _probabilities(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.empty(256, dtype=np.uint32)
    if data.size:
        lib().corto_histogram_u8(address(data), data.size, address(counts))
    else:
        counts.fill(0)
    symbols = np.flatnonzero(counts).astype(np.uint8)
    if not symbols.size:
        return symbols, symbols.copy()
    probabilities = (
        counts[symbols.astype(np.int64)].astype(np.uint64) * 255 // data.size
    ).astype(np.uint8)
    order = np.argsort(-probabilities.astype(np.int16), kind="stable")
    return symbols[order], probabilities[order]


def _dictionary(symbols, probabilities):
    symbols = _array(symbols, np.uint8).reshape(-1)
    probabilities = _array(probabilities, np.uint8).reshape(-1)
    # createDecodingTables2 uses 2*dictionary_size as construction scratch,
    # then compacts the first 256 entries in place.
    index = np.empty(512, dtype=np.int32)
    lengths = np.empty(512, dtype=np.int32)
    table = np.empty(8192, dtype=np.uint8)
    queues = np.empty(512, dtype=np.uint32)
    starts = np.empty(max(1, symbols.size), dtype=np.int32)
    word_count = int(
        lib().corto_tunstall_build(
            address(symbols),
            address(probabilities),
            symbols.size,
            address(index),
            address(lengths),
            address(table),
            address(queues),
            address(starts),
            table.size,
        )
    )
    if word_count < 0:
        raise RuntimeError("Tunstall dictionary exceeded corto's 8192-byte table")
    used = int(np.max(index[:word_count] + lengths[:word_count])) if word_count else 0
    return index[:word_count].copy(), lengths[:word_count].copy(), table[:used].copy()


def tunstall_compress(data) -> TunstallBlock:
    """Compress bytes in corto's OutStream::tunstall_compress wire format."""
    source = _array(
        np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data,
        np.uint8,
    ).reshape(-1)
    if source.size == 0:
        return TunstallBlock(b"\x00" + struct.pack("<II", 0, 0), 0)
    symbols, probabilities = _probabilities(source)
    if symbols.size == 256:
        raise ValueError(
            "corto's one-byte symbol-count field cannot represent all 256 symbols"
        )
    pairs = np.column_stack((symbols, probabilities)).astype(np.uint8).tobytes()
    if symbols.size == 1:
        encoded = b""
    else:
        index, lengths, table = _dictionary(symbols, probabilities)
        target = np.empty(source.size * 2, dtype=np.uint8)
        encoded_size = int(
            lib().corto_tunstall_encode(
                address(source),
                source.size,
                address(index),
                address(lengths),
                address(table),
                index.size,
                address(target),
            )
        )
        if encoded_size < 0:
            raise RuntimeError("Tunstall encoding failed")
        encoded = target[:encoded_size].tobytes()
    payload = (
        bytes((symbols.size & 0xFF,))
        + pairs
        + struct.pack("<II", source.size, len(encoded))
        + encoded
    )
    return TunstallBlock(payload, source.size)


def tunstall_decompress(block: TunstallBlock | bytes | bytearray) -> bytes:
    payload = block.payload if isinstance(block, TunstallBlock) else bytes(block)
    if len(payload) < 9:
        raise ValueError("truncated Tunstall stream")
    symbol_count = payload[0]
    pair_end = 1 + 2 * symbol_count
    if len(payload) < pair_end + 8:
        raise ValueError("truncated Tunstall probability table")
    pairs = np.frombuffer(payload[1:pair_end], dtype=np.uint8).reshape(-1, 2)
    size, encoded_size = struct.unpack_from("<II", payload, pair_end)
    encoded_start = pair_end + 8
    if encoded_start + encoded_size != len(payload):
        raise ValueError("Tunstall payload is truncated or has trailing data")
    if size == 0:
        return b""
    if symbol_count == 0:
        raise ValueError("non-empty stream has no symbols")
    if symbol_count == 1:
        return bytes((int(pairs[0, 0]),)) * size
    if encoded_size == 0:
        raise ValueError("non-empty multi-symbol stream has no encoded payload")
    index, lengths, table = _dictionary(pairs[:, 0], pairs[:, 1])
    encoded = np.frombuffer(
        payload[encoded_start : encoded_start + encoded_size], dtype=np.uint8
    )
    target = np.empty(size, dtype=np.uint8)
    written = int(
        lib().corto_tunstall_decode(
            address(encoded),
            encoded.size,
            address(index),
            address(lengths),
            address(table),
            index.size,
            address(target),
            size,
        )
    )
    if written != size:
        raise ValueError("Tunstall stream ended before the declared output size")
    return target.tobytes()


@dataclass(frozen=True)
class Connectivity:
    """Corto CLERS symbols, split operands, and attribute prediction contexts."""

    clers: np.ndarray
    splits: np.ndarray
    prediction: np.ndarray
    face_count: int
    original_vertex_count: int
    removed_degenerate_faces: int

    @property
    def vertex_count(self) -> int:
        return int(self.prediction.shape[0])

    @property
    def vertex_order(self) -> np.ndarray:
        return self.prediction[:, 0]


@dataclass(frozen=True)
class DecodedConnectivity:
    faces: np.ndarray
    prediction: np.ndarray


def encode_connectivity(faces, vertex_count: int | None = None) -> Connectivity:
    """Encode triangle connectivity using corto's CLERS front traversal.

    Repeated-index faces are removed exactly as in ``Encoder::encodeMesh``.
    Unreferenced vertices are intentionally omitted, matching the default
    upstream build without ``PRESERVED_UNREFERENCED``.
    """
    source = _array(faces, np.uint32, ndim=2)
    if source.shape[1] != 3:
        raise ValueError("faces must have shape (n, 3)")
    inferred = int(source.max()) + 1 if source.size else 0
    total_vertices = inferred if vertex_count is None else int(vertex_count)
    if total_vertices < inferred or total_vertices < 0:
        raise ValueError("vertex_count does not cover every face index")
    keep = (
        (source[:, 0] != source[:, 1])
        & (source[:, 0] != source[:, 2])
        & (source[:, 1] != source[:, 2])
    )
    clean = np.ascontiguousarray(source[keep])
    removed = int(source.shape[0] - clean.shape[0])
    face_count = clean.shape[0]
    if face_count == 0:
        return Connectivity(
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.int32),
            np.empty((0, 4), dtype=np.int32),
            0,
            total_vertices,
            removed,
        )

    edges = face_count * 3
    opposite_face = np.empty(edges, dtype=np.int32)
    opposite_side = np.empty(edges, dtype=np.int32)
    edge_records = np.empty(2 * edges, dtype=np.uint64)
    lib().corto_build_topology(
        address(clean),
        face_count,
        address(opposite_face),
        address(opposite_side),
        address(edge_records),
    )

    front_capacity = edges + 3
    encoded = np.empty(max(1, total_vertices), dtype=np.int32)
    prediction = np.empty((max(1, total_vertices), 4), dtype=np.int32)
    clers = np.empty(8 * face_count + 16, dtype=np.uint8)
    splits = np.empty(4 * face_count + 16, dtype=np.int32)
    visited = np.empty(face_count, dtype=np.uint8)
    faceorder = np.empty(front_capacity, dtype=np.int32)
    delayed = np.empty(front_capacity, dtype=np.int32)
    front_face = np.empty(front_capacity, dtype=np.int32)
    front_side = np.empty(front_capacity, dtype=np.uint8)
    front_prev = np.empty(front_capacity, dtype=np.int32)
    front_next = np.empty(front_capacity, dtype=np.int32)
    front_deleted = np.empty(front_capacity, dtype=np.uint8)
    metadata = np.zeros(5, dtype=np.int64)
    status = int(
        lib().corto_connectivity_encode(
            address(clean),
            face_count,
            total_vertices,
            address(opposite_face),
            address(opposite_side),
            address(encoded),
            address(prediction),
            address(clers),
            address(splits),
            address(visited),
            address(faceorder),
            address(delayed),
            address(front_face),
            address(front_side),
            address(front_prev),
            address(front_next),
            address(front_deleted),
            address(metadata),
        )
    )
    if status != 0:
        raise RuntimeError(f"corto connectivity encoding failed with status {status}")
    cler_count, split_count, encoded_vertices = map(int, metadata[:3])
    return Connectivity(
        clers[:cler_count].copy(),
        splits[:split_count].copy(),
        prediction[:encoded_vertices].copy(),
        face_count,
        total_vertices,
        removed,
    )


def decode_connectivity(connectivity: Connectivity) -> DecodedConnectivity:
    """Decode CLERS symbols to faces in corto's encoded vertex order."""
    face_count = int(connectivity.face_count)
    vertex_count = connectivity.vertex_count
    if face_count == 0:
        return DecodedConnectivity(
            np.empty((0, 3), dtype=np.uint32),
            np.empty((0, 3), dtype=np.int32),
        )
    clers = _array(connectivity.clers, np.uint8).reshape(-1)
    splits = _array(connectivity.splits, np.int32).reshape(-1)
    if clers.size == 0:
        raise ValueError("non-empty connectivity has no CLERS symbols")
    safe_splits = splits if splits.size else np.zeros(1, dtype=np.int32)
    faces = np.empty((face_count, 3), dtype=np.uint32)
    prediction = np.empty((max(1, vertex_count), 3), dtype=np.int32)
    capacity = face_count * 3 + 3
    faceorder = np.empty(capacity, dtype=np.int32)
    delayed = np.empty(capacity, dtype=np.int32)
    front_v0 = np.empty(capacity, dtype=np.int32)
    front_v1 = np.empty(capacity, dtype=np.int32)
    front_v2 = np.empty(capacity, dtype=np.int32)
    front_prev = np.empty(capacity, dtype=np.int32)
    front_next = np.empty(capacity, dtype=np.int32)
    front_deleted = np.empty(capacity, dtype=np.uint8)
    metadata = np.zeros(4, dtype=np.int64)
    status = int(
        lib().corto_connectivity_decode(
            address(clers),
            clers.size,
            address(safe_splits),
            splits.size,
            vertex_count,
            face_count,
            address(faces),
            address(prediction),
            address(faceorder),
            address(delayed),
            address(front_v0),
            address(front_v1),
            address(front_v2),
            address(front_prev),
            address(front_next),
            address(front_deleted),
            address(metadata),
        )
    )
    if status != 0:
        raise RuntimeError(f"corto connectivity decoding failed with status {status}")
    consumed_clers, consumed_splits, decoded_vertices = map(int, metadata[:3])
    if (
        consumed_clers != clers.size
        or consumed_splits != splits.size
        or decoded_vertices != vertex_count
    ):
        raise RuntimeError("connectivity stream has trailing or missing data")
    return DecodedConnectivity(faces, prediction[:vertex_count].copy())


def reorder_vertices(values, connectivity: Connectivity) -> np.ndarray:
    """Gather vertex attributes into the order produced by connectivity coding."""
    source = np.asarray(values)
    if source.shape[0] < connectivity.original_vertex_count:
        raise ValueError("values do not cover the original vertex range")
    return np.ascontiguousarray(source[connectivity.vertex_order])


@dataclass(frozen=True)
class CompressedMesh:
    """End-to-end compressed position and triangle-connectivity payload."""

    connectivity: Connectivity
    clers: TunstallBlock
    position_words: np.ndarray
    position_logs: TunstallBlock
    q: float
    offset: np.ndarray

    @property
    def compressed_bytes(self) -> int:
        # Prediction contexts are regenerated by connectivity decoding.
        return (
            len(self.clers.payload)
            + self.connectivity.splits.nbytes
            + self.position_words.nbytes
            + len(self.position_logs.payload)
            + self.offset.nbytes
            + 16
        )


def compress_mesh(
    positions,
    faces,
    *,
    q: float | None = None,
    bits: int = 14,
    offset=None,
) -> CompressedMesh:
    """Compress mesh positions through corto's full hot-kernel pipeline."""
    points = _array(positions, np.float32, ndim=2)
    if points.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")
    origin = (
        np.zeros(3, dtype=np.float32)
        if offset is None
        else _array(offset, np.float32).reshape(-1)
    )
    if origin.size != 3:
        raise ValueError("offset must have three values")
    step = quantization_step(points, bits) if q is None else float(q)
    if step == 0:
        step = 1.0
        if points.size:
            origin = points[0].copy()
    connectivity = encode_connectivity(faces, points.shape[0])
    quantized = quantize(points, step, offset=origin)
    diffs = delta_encode(quantized, connectivity.prediction, parallelogram=True)
    packed = pack_correlated(diffs)
    return CompressedMesh(
        connectivity,
        tunstall_compress(connectivity.clers),
        packed.words,
        tunstall_compress(packed.logs),
        step,
        origin.copy(),
    )


def decompress_mesh(block: CompressedMesh) -> tuple[np.ndarray, np.ndarray]:
    """Return reordered positions and faces from :func:`compress_mesh`."""
    clers = np.frombuffer(tunstall_decompress(block.clers), dtype=np.uint8).copy()
    connectivity = Connectivity(
        clers,
        block.connectivity.splits,
        block.connectivity.prediction,
        block.connectivity.face_count,
        block.connectivity.original_vertex_count,
        block.connectivity.removed_degenerate_faces,
    )
    decoded = decode_connectivity(connectivity)
    logs = np.frombuffer(
        tunstall_decompress(block.position_logs), dtype=np.uint8
    ).copy()
    diffs = unpack_correlated(PackedArray(block.position_words, logs, 3))
    quantized = delta_decode(diffs, decoded.prediction, parallelogram=True)
    positions = dequantize(quantized, block.q, offset=block.offset)
    return positions, decoded.faces
