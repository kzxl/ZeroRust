//! C-FFI exports for ZeroPlatform (.NET C# P/Invoke) integration.
//!
//! Provides ABI-stable, high-performance compute kernels for GEMM, FlashAttention,
//! and Quantized Matrix-Vector multiplications.

use crate::gemm;
use crate::attention;
use crate::quant;

/// Returns the C-ABI version of ZeroRust Compute Engine (e.g. 100 for 1.0.0).
#[no_mangle]
pub extern "C" fn zerorust_version() -> u32 {
    100
}

/// Checks whether AVX2 and FMA hardware SIMD acceleration is supported on the running CPU.
/// Returns 1 if supported, 0 otherwise.
#[no_mangle]
pub extern "C" fn zerorust_is_simd_supported() -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return 1;
        }
    }
    0
}

/// Evaluates C = alpha * (A @ B) + beta * C with arbitrary row/col strides.
///
/// # Safety
/// Pointers `a`, `b`, and `c` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zerorust_gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
    b: *const f32,
    rsb: isize,
    csb: isize,
    beta: f32,
    c: *mut f32,
    rsc: isize,
    csc: isize,
    parallel: i32,
) -> i32 {
    if a.is_null() || b.is_null() || c.is_null() {
        return -1;
    }

    gemm::gemm_f32(
        m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, parallel != 0,
    );
    0
}

/// Evaluates FlashAttention on CPU with 2D block-tiling and online softmax.
///
/// # Safety
/// Pointers `q`, `k`, `v`, and `out` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zerorust_flash_attention_f32(
    batch_count: usize,
    sq: usize,
    sk: usize,
    d: usize,
    dv: usize,
    scale_factor: f32,
    is_gqa: i32,
    hq: usize,
    hkv: usize,
    gqa_group: usize,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    mask: *const f32,
    out: *mut f32,
) -> i32 {
    if q.is_null() || k.is_null() || v.is_null() || out.is_null() {
        return -1;
    }

    attention::flash_attention_f32(
        batch_count,
        sq,
        sk,
        d,
        dv,
        scale_factor,
        is_gqa != 0,
        hq,
        hkv,
        gqa_group,
        q,
        k,
        v,
        mask,
        out,
    );
    0
}

/// Evaluates streaming on-the-fly Q4_0 Matrix-Vector multiplication.
///
/// # Safety
/// Pointers `ptr_weight`, `p_act`, and `p_dst` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zerorust_gemv_q4_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    if ptr_weight.is_null() || p_act.is_null() || p_dst.is_null() {
        return -1;
    }

    quant::gemv_q4_0(ptr_weight, p_act, p_dst, n_rows, k_cols);
    0
}

/// Evaluates streaming on-the-fly Q8_0 Matrix-Vector multiplication.
///
/// # Safety
/// Pointers `ptr_weight`, `p_act`, and `p_dst` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zerorust_gemv_q8_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    if ptr_weight.is_null() || p_act.is_null() || p_dst.is_null() {
        return -1;
    }

    quant::gemv_q8_0(ptr_weight, p_act, p_dst, n_rows, k_cols);
    0
}
