"""ctypes loader for the Mojo corto kernels."""

from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_CORTO_LIB") or os.path.join(
    ROOT, "dist", "libmojo-corto.so"
)

I = ctypes.c_int64
F32 = ctypes.c_float

_SIGNATURES = {
    "corto_parallel_init": ([], I),
    "corto_quantize_f32": ([I, I, I, I, F32, I, I], I),
    "corto_dequantize_f32": ([I, I, I, I, F32, I], None),
    "corto_delta_encode": ([I, I, I, I, I, I], None),
    "corto_delta_decode": ([I, I, I, I, I, I], None),
    "corto_pack_correlated": ([I, I, I, I, I], I),
    "corto_unpack_correlated": ([I, I, I, I, I, I], I),
    "corto_normals_to_octa": ([I, I, I, I], None),
    "corto_octa_to_normals": ([I, I, I, I], None),
    "corto_estimate_normals": ([I, I, I, I, I, I], None),
    "corto_colors_to_ycc": ([I, I, I, I, I], None),
    "corto_ycc_to_colors": ([I, I, I, I, I, I], None),
    "corto_histogram_u8": ([I, I, I], None),
    "corto_tunstall_build": ([I, I, I, I, I, I, I, I, I], I),
    "corto_tunstall_encode": ([I, I, I, I, I, I, I], I),
    "corto_tunstall_decode": ([I, I, I, I, I, I, I, I], I),
    "corto_build_topology": ([I, I, I, I, I, I, I, I, I], None),
    "corto_connectivity_encode": ([I] * 18, I),
    "corto_connectivity_decode": ([I] * 17, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "corto.mojo")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if not os.path.exists(pixi):
        raise BuildError("pixi is required to build libmojo-corto.so")
    proc = subprocess.run(
        [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
        parallel_device = _LIBRARY.corto_parallel_init()
        release_device = _LIBRARY.KGEN_CompilerRT_AsyncRT_ReleaseCPUDevice
        release_device.argtypes = [I]
        release_device.restype = None
        atexit.register(release_device, parallel_device)
    return _LIBRARY


def address(array) -> int:
    return int(array.ctypes.data)
