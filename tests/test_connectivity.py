import numpy as np
import pytest

import mojo_corto as corto
import reference


def roundtrip_faces(faces, vertex_count=None):
    encoded = corto.encode_connectivity(faces, vertex_count)
    decoded = corto.decode_connectivity(encoded)
    mapping = {int(old): new for new, old in enumerate(encoded.vertex_order)}
    clean = [
        [mapping[int(v)] for v in face]
        for face in np.asarray(faces)
        if len(set(map(int, face))) == 3
    ]
    assert reference.face_multiset(decoded.faces) == reference.face_multiset(clean)
    assert reference.edge_multiplicities(decoded.faces) == reference.edge_multiplicities(clean)
    return encoded, decoded


def test_empty_mesh_and_unreferenced_vertices():
    encoded, decoded = roundtrip_faces(np.empty((0, 3), np.uint32), 7)
    assert encoded.vertex_count == 0
    assert encoded.original_vertex_count == 7
    assert decoded.faces.shape == (0, 3)


def test_single_triangle_exact_symbols():
    encoded, decoded = roundtrip_faces(np.array([[0, 1, 2]], np.uint32), 5)
    np.testing.assert_array_equal(encoded.clers, [0])
    np.testing.assert_array_equal(encoded.vertex_order, [0, 1, 2])
    np.testing.assert_array_equal(decoded.faces, [[0, 1, 2]])


def test_open_strip_parallelogram_context():
    faces = np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4]], np.uint32)
    encoded, decoded = roundtrip_faces(faces)
    assert encoded.vertex_count == 5
    np.testing.assert_array_equal(decoded.prediction[3], [2, 1, 0])


def test_watertight_tetrahedron_euler_characteristic():
    faces = np.array(
        [[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]], dtype=np.uint32
    )
    encoded, decoded = roundtrip_faces(faces)
    unique_edges = len(reference.edge_multiplicities(decoded.faces))
    assert encoded.vertex_count - unique_edges + decoded.faces.shape[0] == 2
    assert all(value == 2 for value in reference.edge_multiplicities(decoded.faces))
    assert set(encoded.clers).issubset(set(range(7)))


def test_repeated_index_degenerate_removed_zero_area_preserved():
    faces = np.array([[0, 0, 1], [0, 1, 2], [2, 3, 4]], np.uint32)
    encoded, decoded = roundtrip_faces(faces, 6)
    assert encoded.removed_degenerate_faces == 1
    assert decoded.faces.shape[0] == 2


def test_duplicate_vertices_and_zero_area_geometry():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0, 0], [0, 1, 0]],
        np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 3, 4]], np.uint32)
    encoded, decoded = roundtrip_faces(faces)
    reordered = corto.reorder_vertices(vertices, encoded)
    areas = []
    for a, b, c in decoded.faces:
        areas.append(np.linalg.norm(np.cross(reordered[b] - reordered[a], reordered[c] - reordered[a])) / 2)
    np.testing.assert_array_equal(areas, [0, 0])


def test_non_manifold_edge_roundtrip():
    faces = np.array(
        [[0, 1, 2], [1, 0, 3], [0, 1, 4], [1, 0, 5]], dtype=np.uint32
    )
    encoded, decoded = roundtrip_faces(faces)
    assert max(reference.edge_multiplicities(decoded.faces)) == 4
    assert encoded.vertex_count == 6


def test_disconnected_components_use_split_when_reusing_vertex():
    faces = np.array([[0, 1, 2], [0, 3, 4]], np.uint32)
    encoded, _ = roundtrip_faces(faces)
    assert 6 in encoded.clers
    assert encoded.splits[0] == 1


def test_connectivity_attribute_prediction_roundtrip():
    faces = np.array(
        [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]], np.uint32
    )
    values = np.array(
        [[10, 0, 3], [11, 2, 5], [20, 4, 7], [24, 8, 8], [31, 9, 2], [40, 3, 1]],
        np.int32,
    )
    encoded = corto.encode_connectivity(faces)
    decoded = corto.decode_connectivity(encoded)
    diffs = corto.delta_encode(values, encoded.prediction, parallelogram=True)
    restored = corto.delta_decode(diffs, decoded.prediction, parallelogram=True)
    np.testing.assert_array_equal(restored, corto.reorder_vertices(values, encoded))


def test_connectivity_rejects_out_of_range_vertex_count():
    with pytest.raises(ValueError):
        corto.encode_connectivity([[0, 1, 3]], vertex_count=3)


def test_connectivity_decode_rejects_truncated_native_streams():
    encoded = corto.encode_connectivity(
        np.array([[0, 1, 2], [2, 1, 3]], np.uint32), vertex_count=4
    )
    truncated_clers = corto.Connectivity(
        encoded.clers[:-1],
        encoded.splits,
        encoded.prediction,
        encoded.face_count,
        encoded.original_vertex_count,
        encoded.removed_degenerate_faces,
    )
    with pytest.raises(RuntimeError, match="status"):
        corto.decode_connectivity(truncated_clers)

    split_mesh = corto.encode_connectivity(
        np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]], np.uint32),
        vertex_count=5,
    )
    if split_mesh.splits.size:
        truncated_splits = corto.Connectivity(
            split_mesh.clers,
            split_mesh.splits[:-1],
            split_mesh.prediction,
            split_mesh.face_count,
            split_mesh.original_vertex_count,
            split_mesh.removed_degenerate_faces,
        )
        with pytest.raises(RuntimeError, match="status"):
            corto.decode_connectivity(truncated_splits)


def test_end_to_end_mesh_compression_preserves_geometry():
    vertices = np.array(
        [[0, 0, 0], [1.125, 0, 0], [1, 1, 0], [0, 1.25, 0], [9, 9, 9]],
        np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)
    block = corto.compress_mesh(vertices, faces, bits=12)
    decoded_vertices, decoded_faces = corto.decompress_mesh(block)
    expected_vertices = corto.reorder_vertices(vertices, block.connectivity)
    assert np.max(np.abs(decoded_vertices - expected_vertices)) <= block.q
    assert reference.face_multiset(decoded_faces) == reference.face_multiset(
        [[0, 1, 2], [0, 2, 3]]
    )
    assert block.compressed_bytes > 0
