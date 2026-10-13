"""
ZeroCompute Python Ctypes Binding and Verification Sample.

Provides a lightweight, zero-dependency Python bridge to the sovereign
ZeroCompute native dynamic library (zero_compute.dll / libzero_compute.so).
"""

import ctypes
import os
import sys
from pathlib import Path


class ZeroCompute:
    """Python wrapper for ZeroCompute C-ABI."""

    def __init__(self, dll_path: str = None):
        if dll_path is None:
            # Look in standard locations relative to this script
            base_dir = Path(__file__).resolve().parent
            candidates = [
                base_dir / "zero_compute.dll",
                base_dir.parent.parent / "target" / "release" / "zero_compute.dll",
                base_dir.parent.parent / "target" / "debug" / "zero_compute.dll",
            ]
            for c in candidates:
                if c.exists():
                    dll_path = str(c)
                    break

        if not dll_path or not os.path.exists(dll_path):
            raise FileNotFoundError(f"Cannot find zero_compute dynamic library: {dll_path}")

        self._lib = ctypes.CDLL(dll_path)
        self._setup_bindings()

    def _setup_bindings(self):
        # uint32_t zc_version(void)
        self._lib.zc_version.argtypes = []
        self._lib.zc_version.restype = ctypes.c_uint32

        # int32_t zc_is_simd_supported(void)
        self._lib.zc_is_simd_supported.argtypes = []
        self._lib.zc_is_simd_supported.restype = ctypes.c_int32

        # int32_t zc_gemm_f32(...)
        self._lib.zc_gemm_f32.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int32
        ]
        self._lib.zc_gemm_f32.restype = ctypes.c_int32

        # int32_t zc_rmsnorm_f32(size_t n, const float* x, const float* weight, float* dst, float eps)
        self._lib.zc_rmsnorm_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_float
        ]
        self._lib.zc_rmsnorm_f32.restype = ctypes.c_int32

        # int32_t zc_swiglu_f32(size_t n, const float* gate, const float* up, float* dst)
        self._lib.zc_swiglu_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self._lib.zc_swiglu_f32.restype = ctypes.c_int32

        # int32_t zc_rope_f32(size_t num_heads, size_t seq_len, size_t head_dim, size_t pos_offset, float theta_base, float* x)
        self._lib.zc_rope_f32.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float)
        ]
        self._lib.zc_rope_f32.restype = ctypes.c_int32

    @property
    def version(self) -> int:
        return self._lib.zc_version()

    @property
    def is_simd_supported(self) -> bool:
        return bool(self._lib.zc_is_simd_supported())

    def gemm_f32(self, m: int, n: int, k: int, a: list[float], b: list[float], alpha: float = 1.0, beta: float = 0.0) -> list[float]:
        c = [0.0] * (m * n)
        c_a = (ctypes.c_float * len(a))(*a)
        c_b = (ctypes.c_float * len(b))(*b)
        c_c = (ctypes.c_float * len(c))(*c)

        status = self._lib.zc_gemm_f32(m, n, k, alpha, c_a, k, 1, c_b, n, 1, beta, c_c, n, 1, 0)
        if status != 0:
            raise RuntimeError(f"zc_gemm_f32 failed: {status}")
        return list(c_c)

    def rmsnorm_f32(self, x: list[float], weight: list[float], eps: float = 1e-5) -> list[float]:
        n = len(x)
        dst = [0.0] * n
        c_x = (ctypes.c_float * n)(*x)
        c_w = (ctypes.c_float * n)(*weight)
        c_dst = (ctypes.c_float * n)(*dst)

        status = self._lib.zc_rmsnorm_f32(n, c_x, c_w, c_dst, eps)
        if status != 0:
            raise RuntimeError(f"zc_rmsnorm_f32 failed: {status}")
        return list(c_dst)

    def swiglu_f32(self, gate: list[float], up: list[float]) -> list[float]:
        assert len(gate) == len(up)
        n = len(gate)
        dst = [0.0] * n
        c_g = (ctypes.c_float * n)(*gate)
        c_u = (ctypes.c_float * n)(*up)
        c_dst = (ctypes.c_float * n)(*dst)

        status = self._lib.zc_swiglu_f32(n, c_g, c_u, c_dst)
        if status != 0:
            raise RuntimeError(f"zc_swiglu_f32 failed: {status}")
        return list(c_dst)

    def rope_f32(self, num_heads: int, seq_len: int, head_dim: int, pos_offset: int, theta_base: float, x: list[float]) -> list[float]:
        n = len(x)
        assert n == num_heads * seq_len * head_dim
        c_x = (ctypes.c_float * n)(*x)

        status = self._lib.zc_rope_f32(num_heads, seq_len, head_dim, pos_offset, theta_base, c_x)
        if status != 0:
            raise RuntimeError(f"zc_rope_f32 failed: {status}")
        return list(c_x)


def run_tests():
    print("=" * 60)
    print(" ZeroCompute Python Ctypes Sovereign Verification")
    print("=" * 60)

    zc = ZeroCompute()
    print(f"[1] ZeroCompute C-ABI Version: {zc.version}")
    print(f"    Hardware AVX2+FMA SIMD Supported: {zc.is_simd_supported}")

    # GEMM 2x2
    # A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    # C = [[19, 22], [43, 50]]
    c = zc.gemm_f32(2, 2, 2, [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    print(f"[2] GEMM F32 Result: {c}")
    assert abs(c[0] - 19.0) < 1e-4 and abs(c[1] - 22.0) < 1e-4
    assert abs(c[2] - 43.0) < 1e-4 and abs(c[3] - 50.0) < 1e-4
    print("    GEMM PASS")

    # RMSNorm
    x = [2.0, 2.0, 2.0, 2.0]
    w = [1.0, 1.0, 1.0, 1.0]
    norm = zc.rmsnorm_f32(x, w)
    print(f"[3] RMSNorm Result: {[round(v, 4) for v in norm]}")
    for val in norm:
        assert abs(val - 1.0) < 1e-3
    print("    RMSNorm PASS")

    # SwiGLU
    gate = [0.0, 0.0]
    up = [10.0, 20.0]
    sw = zc.swiglu_f32(gate, up)
    print(f"[4] SwiGLU Result: {sw}")
    assert abs(sw[0] - 0.0) < 1e-4 and abs(sw[1] - 0.0) < 1e-4
    print("    SwiGLU PASS")

    # RoPE
    rope_vec = [1.0, 2.0, 3.0, 4.0]
    out_rope = zc.rope_f32(1, 1, 4, 0, 10000.0, rope_vec)
    print(f"[5] RoPE Result (pos 0): {out_rope}")
    for orig, actual in zip(rope_vec, out_rope):
        assert abs(orig - actual) < 1e-4
    print("    RoPE PASS")

    print("=" * 60)
    print(">>> ALL PYTHON CTYPES VERIFICATION TESTS PASSED! <<<")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
