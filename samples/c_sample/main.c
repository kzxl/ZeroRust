#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../include/zero_compute.h"

static int assert_close(float actual, float expected, float eps, const char* name) {
    float diff = fabsf(actual - expected);
    if (diff > eps) {
        printf("FAIL: %s: expected %f, got %f (diff %f > eps %f)\n", name, expected, actual, diff, eps);
        return 0;
    }
    return 1;
}

int main(void) {
    printf("============================================================\n");
    printf(" ZeroCompute C-ABI Sovereign Verification Sample\n");
    printf("============================================================\n");

    // 1. Version & Hardware SIMD Capability
    uint32_t version = zc_version();
    int32_t simd = zc_is_simd_supported();
    printf("[1] ZeroCompute C-ABI Version: %u\n", version);
    printf("    Hardware AVX2+FMA SIMD Supported: %s\n", simd ? "YES" : "NO");

    if (version < 100) {
        printf("ERROR: Unexpected version\n");
        return 1;
    }

    // 2. GEMM F32 Test: C = 1.0 * (A @ B) + 0.0 * C
    // A: 2x2 [[1, 2], [3, 4]]
    // B: 2x2 [[5, 6], [7, 8]]
    // C expected: [[19, 22], [43, 50]]
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int32_t status = zc_gemm_f32(2, 2, 2, 1.0f, a, 2, 1, b, 2, 1, 0.0f, c, 2, 1, 0);
    if (status != ZC_SUCCESS) {
        printf("ERROR: zc_gemm_f32 failed with code %d\n", status);
        return 1;
    }

    int gemm_ok = assert_close(c[0], 19.0f, 1e-4f, "GEMM C[0,0]") &&
                  assert_close(c[1], 22.0f, 1e-4f, "GEMM C[0,1]") &&
                  assert_close(c[2], 43.0f, 1e-4f, "GEMM C[1,0]") &&
                  assert_close(c[3], 50.0f, 1e-4f, "GEMM C[1,1]");
    printf("[2] GEMM F32 (2x2): %s -> [[%.1f, %.1f], [%.1f, %.1f]]\n",
           gemm_ok ? "PASS" : "FAIL", c[0], c[1], c[2], c[3]);

    // 3. RMSNorm F32 Test
    // Vector: [2, 2, 2, 2], gamma: [1, 1, 1, 1], eps: 1e-5
    // RMS = sqrt(mean(4, 4, 4, 4) + 1e-5) = sqrt(4 + 1e-5) ≈ 2.0
    // Normalized = 2.0 / 2.0 = 1.0
    float x_norm[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out_norm[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    status = zc_rmsnorm_f32(4, x_norm, gamma, out_norm, 1e-5f);
    if (status != ZC_SUCCESS) {
        printf("ERROR: zc_rmsnorm_f32 failed with code %d\n", status);
        return 1;
    }

    int rms_ok = assert_close(out_norm[0], 1.0f, 1e-3f, "RMSNorm[0]") &&
                 assert_close(out_norm[1], 1.0f, 1e-3f, "RMSNorm[1]");
    printf("[3] RMSNorm F32: %s -> [%.4f, %.4f, %.4f, %.4f]\n",
           rms_ok ? "PASS" : "FAIL", out_norm[0], out_norm[1], out_norm[2], out_norm[3]);

    // 4. SwiGLU F32 Test
    // gate = [0, 0], up = [10, 20] -> silu(0) * up = 0.5 * 0 * up = 0
    float gate[2] = {0.0f, 0.0f};
    float up[2] = {10.0f, 20.0f};
    float out_swiglu[2] = {-1.0f, -1.0f};

    status = zc_swiglu_f32(2, gate, up, out_swiglu);
    if (status != ZC_SUCCESS) {
        printf("ERROR: zc_swiglu_f32 failed with code %d\n", status);
        return 1;
    }

    int swiglu_ok = assert_close(out_swiglu[0], 0.0f, 1e-4f, "SwiGLU[0]") &&
                    assert_close(out_swiglu[1], 0.0f, 1e-4f, "SwiGLU[1]");
    printf("[4] SwiGLU F32: %s -> [%.4f, %.4f]\n",
           swiglu_ok ? "PASS" : "FAIL", out_swiglu[0], out_swiglu[1]);

    // 5. RoPE F32 Test (Rotary Position Embedding)
    // 1 head, 1 seq, dim = 4, pos = 0 -> cos(0) = 1, sin(0) = 0 -> vector unchanged
    float rope_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    status = zc_rope_f32(1, 1, 4, 0, 10000.0f, rope_data);
    if (status != ZC_SUCCESS) {
        printf("ERROR: zc_rope_f32 failed with code %d\n", status);
        return 1;
    }

    int rope_ok = assert_close(rope_data[0], 1.0f, 1e-4f, "RoPE[0]") &&
                  assert_close(rope_data[1], 2.0f, 1e-4f, "RoPE[1]") &&
                  assert_close(rope_data[2], 3.0f, 1e-4f, "RoPE[2]") &&
                  assert_close(rope_data[3], 4.0f, 1e-4f, "RoPE[3]");
    printf("[5] RoPE F32 (pos 0): %s -> [%.2f, %.2f, %.2f, %.2f]\n",
           rope_ok ? "PASS" : "FAIL", rope_data[0], rope_data[1], rope_data[2], rope_data[3]);

    printf("============================================================\n");
    if (gemm_ok && rms_ok && swiglu_ok && rope_ok) {
        printf(">>> ALL ZEROCOMPUTE C-ABI TESTS PASSED! <<<\n");
        printf("============================================================\n");
        return 0;
    } else {
        printf(">>> SOME VERIFICATION TESTS FAILED! <<<\n");
        printf("============================================================\n");
        return 1;
    }
}
