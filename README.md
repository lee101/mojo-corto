# mojo-corto

`mojo-corto` is a standalone Mojo port of the compute-heavy geometry
compression kernels in [corto](https://github.com/cnr-isti-vclab/corto), the
Visual Computing Lab mesh and point-cloud codec. It provides a NumPy/ctypes API
and an end-to-end in-memory compressed mesh object without linking the original
C++ library. Exact-dtype contiguous inputs cross the FFI without a copy;
wrappers allocate output and scratch arrays and make contiguous copies when
needed.

This is a source-derived port. The individual corto C++ files used here state
LGPL-3.0-or-later, while corto's top-level license file says MIT and its README
calls the C++ code GPL. This repository follows the stricter source-file terms:
[LGPL-3.0-or-later](LICENSE). See [NOTICE](NOTICE) for attribution and the
upstream licensing discrepancy.

## Coverage

Implemented from the real upstream source:

- Tunstall probability histograms, the optimized 256-word dictionary builder,
  compression, decompression, and corto's entropy-block wire layout
- correlated signed-integer bit packing and unpacking
- float attribute quantization, dequantization, point-cloud deltas, mesh
  first-order prediction, and parallelogram prediction
- octahedral normal quantization, sphere reconstruction, and area-weighted
  vertex-normal estimation with corto's `1e-5` degenerate fallback
- RGB/RGBA quantization and byte-wrapping YCC transforms
- edge topology construction and the full CLERS front traversal:
  `VERTEX`, `LEFT`, `RIGHT`, `END`, `BOUNDARY`, `DELAY`, and `SPLIT`
- end-to-end position and connectivity compression through `compress_mesh`

Not yet covered:

- reading or writing complete `.crt` files, EXIF fields, material groups, and
  the C++ class API
- integrated UV/custom-attribute streams and `ESTIMATED`/`BORDER` normal
  residual streams
- corto's Morton sorting pass for unordered point clouds
- upstream's optional 16-bit output-index path

The connectivity API exposes split operands as `int32` arrays instead of
corto's final packed split bitstream. CLERS coding and prediction behavior are
ported, but a `Connectivity` object is not itself a `.crt` stream. Tunstall
blocks are wire-compatible for one through 255 distinct symbols. The upstream
one-byte symbol-count field cannot represent 256, so this port rejects that
case instead of reproducing the upstream overflow.

## Install

```bash
pixi install
pixi run build
```

The pinned Mojo nightly builds `dist/libmojo-corto.so`. Python imports use the
repository's `python/` directory through the Pixi activation environment.

## Usage

```python
import numpy as np
from mojo_corto import compress_mesh, decompress_mesh

vertices = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    dtype=np.float32,
)
faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

block = compress_mesh(vertices, faces, bits=14)
decoded_vertices, decoded_faces = decompress_mesh(block)

print(block.compressed_bytes)
print(decoded_vertices.shape, decoded_faces.shape)
```

Corto reorders referenced vertices during connectivity traversal and omits
unreferenced vertices by default. `decoded_faces` therefore indexes
`decoded_vertices`, not the original input array. Repeated-index triangles are
removed; geometrically zero-area triangles with three distinct indices are
preserved.

Individual APIs such as `tunstall_compress`, `pack_correlated`,
`normals_to_octa`, `colors_to_ycc`, `encode_connectivity`, and
`point_cloud_delta_encode` are also public.

## Correctness

There is no maintained Python package binding the same corto C++ kernels, and
the upstream checkout contains no expected-value unit suite for these
functions. Tests therefore use source-derived NumPy/Python references plus
mesh invariants.

The 41 tests assert exact quantized values, packed words, probability tables,
entropy round trips, CLERS reconstruction, triangle and edge multiplicities,
Euler characteristic, watertightness, quantization error, normal angle bounds,
end-to-end attribute prediction, malformed native stream rejection, and unsafe
dtype rejection. SIMD remainder handling and the quantization parallel path
have dedicated parity cases. Inputs include empty meshes, a single triangle,
repeated and duplicate vertices, zero-area faces, non-manifold edges,
disconnected components, and unreferenced vertices.

```bash
pixi run test
```

## How it works

All hot kernels live in one Mojo compilation unit. Exported functions use a C
ABI and receive NumPy buffer addresses and explicit sizes as machine integers;
each wrapper rebuilds an `UnsafePointer[..., AnyOrigin[mut=True]]`. Python owns
every input, output, and scratch allocation, keeps each array alive for the
synchronous call, and validates shapes, dtypes, ranges, and encoded-stream
lengths before or during native access. No allocator crosses the FFI boundary.

Positions and attributes use contiguous array-of-structures buffers. Mesh
topology is flat index storage: opposite-face arrays, front `prev`/`next`
indices, CLERS bytes, and prediction contexts. There are no pointer graphs or
virtual objects. The end-to-end path is:

```text
topology -> CLERS/Tunstall
positions -> quantize -> parallelogram delta -> correlated bits
                                      logs -> Tunstall
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86_64. These are best-of-three wall-clock results except the deliberately
slow Python bit-pack reference. The connectivity baseline performs NumPy edge
incidence construction rather than the complete CLERS traversal, so that row
is a practical baseline, not equal work.

| Kernel | Mojo | Reference | Speedup | Reference |
|---|---:|---:|---:|---|
| quantize f32, 2M x 3 | 25.639 ms | 31.847 ms | 1.24x | NumPy |
| octa encode, 1.5M | 8.537 ms | 180.209 ms | 21.11x | NumPy |
| correlated bit-pack, 250K x 3 | 10.875 ms | 1278.399 ms | 117.55x | Python port |
| Tunstall decode, 2MB | 2.459 ms | 224.629 ms | 91.35x | Python port |
| connectivity, 178802 faces | 107.011 ms | 695.659 ms | 6.50x | NumPy edge incidence |

The three-component quantizer uses contiguous SIMD blocks with a scalar
remainder and uses thresholded CPU parallelism for large inputs. Finite-value
and integer-range validation is fused into the same native SIMD pass, avoiding
a full-size temporary; the table includes that validation cost.

There is no GPU path. No kernel has the roughly greater-than-two-flops-per-byte
arithmetic intensity needed to justify device transfer and launch overhead;
the kernels are memory-bound, branch-heavy, or serially dependent.

Run the locked benchmark on your machine with:

```bash
pixi run bench
```
