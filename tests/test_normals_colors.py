import numpy as np
import pytest

import mojo_corto as corto
import reference


def test_octa_quantization_matches_normal_attr():
    normals = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [-0.2, 0.7, 0.3],
            [0.2, -0.7, -0.3],
            [0, 0, -1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    actual = corto.normals_to_octa(normals, 10)
    np.testing.assert_array_equal(actual, reference.normals_to_octa(normals, 10))
    np.testing.assert_allclose(
        corto.octa_to_normals(actual, 10),
        reference.octa_to_normals(actual, 10),
        rtol=2e-6,
        atol=2e-6,
    )


def test_octa_roundtrip_angle_bound():
    rng = np.random.default_rng(9)
    normals = rng.normal(size=(5000, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    decoded = corto.octa_to_normals(corto.normals_to_octa(normals, 10), 10)
    cosines = np.einsum("ij,ij->i", normals, decoded)
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    assert angles.max() < 0.5
    np.testing.assert_allclose(np.linalg.norm(decoded, axis=1), 1, atol=2e-6)


def test_estimated_normals_match_upstream_loop():
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [3, 3, 3]],
        np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 0, 1]], np.uint32)
    expected = reference.estimate_normals(positions, faces)
    np.testing.assert_allclose(
        corto.estimate_normals(positions, faces), expected, atol=1e-7
    )
    np.testing.assert_array_equal(expected[4], [0, 0, 1])


def test_estimated_normals_empty_mesh():
    positions = np.empty((0, 3), np.float32)
    faces = np.empty((0, 3), np.uint32)
    assert corto.estimate_normals(positions, faces).shape == (0, 3)


@pytest.mark.parametrize("components,bits", [(3, (6, 7, 6)), (4, (6, 7, 6, 5))])
def test_color_ycc_matches_color_attr(components, bits):
    colors = np.array(
        [[0, 0, 0, 0], [255, 128, 7, 254], [1, 2, 3, 4]], dtype=np.uint8
    )[:, :components]
    ycc, steps = corto.colors_to_ycc(colors, bits)
    np.testing.assert_array_equal(ycc, reference.colors_to_ycc(colors, steps))
    decoded = corto.ycc_to_colors(ycc, steps)
    expected = (colors.astype(np.uint16) // steps * steps).astype(np.uint8)
    np.testing.assert_array_equal(decoded, expected)


def test_rgb_can_decode_to_rgba():
    colors = np.array([[255, 127, 63], [0, 1, 2]], np.uint8)
    ycc, steps3 = corto.colors_to_ycc(colors, (8, 8, 8))
    steps4 = np.append(steps3, np.uint8(1))
    rgba = corto.ycc_to_colors(ycc, steps4, components=4)
    np.testing.assert_array_equal(rgba[:, :3], colors)
    np.testing.assert_array_equal(rgba[:, 3], 255)
