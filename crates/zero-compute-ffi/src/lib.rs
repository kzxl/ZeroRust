//! # ZeroCompute FFI Gateway
//!
//! ABI-stable, panic-shielded C-ABI export interface for ZeroUniverse Compute Substrate.
//! Produces zero_compute.dll / libzero_compute.so with automated C header emission.

use std::panic::catch_unwind;

pub const ZC_SUCCESS: i32 = 0;
pub const ZC_ERR_NULL_POINTER: i32 = -1;
pub const ZC_ERR_INVALID_DIMENSION: i32 = -2;
pub const ZC_ERR_UNSUPPORTED: i32 = -3;
pub const ZC_ERR_PANIC: i32 = -99;

// ============================================================================
// Version & Hardware Capability
// ============================================================================

/// Returns the C-ABI version of ZeroCompute (e.g. 100 for 1.0.0).
#[no_mangle]
pub extern "C" fn zc_version() -> u32 {
    100
}

/// Backward compatibility alias for zerorust_version.
#[no_mangle]
pub extern "C" fn zerorust_version() -> u32 {
    zc_version()
}

/// Returns 1 if AVX2 and FMA hardware SIMD acceleration is supported, 0 otherwise.
#[no_mangle]
pub extern "C" fn zc_is_simd_supported() -> i32 {
    if zero_simd::is_avx2_fma_supported() {
        1
    } else {
        0
    }
}

/// Backward compatibility alias for zerorust_is_simd_supported.
#[no_mangle]
pub extern "C" fn zerorust_is_simd_supported() -> i32 {
    zc_is_simd_supported()
}

// ============================================================================
// General Matrix Multiplication (GEMM)
// ============================================================================

/// Evaluates C = alpha * (A @ B) + beta * C with arbitrary row/col strides.
///
/// # Safety
/// Pointers `a`, `b`, and `c` must point to valid allocations satisfying the specified dimensions and strides.
#[no_mangle]
pub unsafe extern "C" fn zc_gemm_f32(
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
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::gemm_f32(
            m,
            n,
            k,
            alpha,
            a,
            rsa,
            csa,
            b,
            rsb,
            csb,
            beta,
            c,
            rsc,
            csc,
            parallel != 0,
        );
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Backward compatibility alias for zerorust_gemm_f32.
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
    zc_gemm_f32(
        m, n, k, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc, parallel,
    )
}

// ============================================================================
// FlashAttention-2
// ============================================================================

/// Evaluates FlashAttention on CPU with 2D block-tiling and online softmax.
///
/// # Safety
/// Pointers `q`, `k`, `v`, and `out` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zc_flash_attention_f32(
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
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::flash_attention_f32(
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
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Backward compatibility alias for zerorust_flash_attention_f32.
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
    zc_flash_attention_f32(
        batch_count,
        sq,
        sk,
        d,
        dv,
        scale_factor,
        is_gqa,
        hq,
        hkv,
        gqa_group,
        q,
        k,
        v,
        mask,
        out,
    )
}

// ============================================================================
// Streaming Quantized GEMV (Q4_0, Q8_0)
// ============================================================================

/// Evaluates streaming on-the-fly Q4_0 Matrix-Vector multiplication.
///
/// # Safety
/// Pointers `ptr_weight`, `p_act`, and `p_dst` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zc_gemv_q4_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    if ptr_weight.is_null() || p_act.is_null() || p_dst.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::gemv_q4_0(ptr_weight, p_act, p_dst, n_rows, k_cols);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Backward compatibility alias for zerorust_gemv_q4_0.
#[no_mangle]
pub unsafe extern "C" fn zerorust_gemv_q4_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    zc_gemv_q4_0(ptr_weight, p_act, p_dst, n_rows, k_cols)
}

/// Evaluates streaming on-the-fly Q8_0 Matrix-Vector multiplication.
///
/// # Safety
/// Pointers `ptr_weight`, `p_act`, and `p_dst` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zc_gemv_q8_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    if ptr_weight.is_null() || p_act.is_null() || p_dst.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::gemv_q8_0(ptr_weight, p_act, p_dst, n_rows, k_cols);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Backward compatibility alias for zerorust_gemv_q8_0.
#[no_mangle]
pub unsafe extern "C" fn zerorust_gemv_q8_0(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    zc_gemv_q8_0(ptr_weight, p_act, p_dst, n_rows, k_cols)
}

// ============================================================================
// Modern Neural Activations & Normalizations
// ============================================================================

/// Evaluates Root Mean Square Normalization (RMSNorm).
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
#[no_mangle]
pub unsafe extern "C" fn zc_rmsnorm_f32(
    n: usize,
    x: *const f32,
    weight: *const f32,
    dst: *mut f32,
    eps: f32,
) -> i32 {
    if x.is_null() || weight.is_null() || dst.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::rmsnorm_f32(n, x, weight, dst, eps);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates Vectorized SiLU: dst[i] = x[i] / (1 + exp(-x[i])).
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
#[no_mangle]
pub unsafe extern "C" fn zc_silu_f32(n: usize, src: *const f32, dst: *mut f32) -> i32 {
    if src.is_null() || dst.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::silu_f32(n, src, dst);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates Vectorized SwiGLU: dst[i] = silu(gate[i]) * up[i].
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
#[no_mangle]
pub unsafe extern "C" fn zc_swiglu_f32(
    n: usize,
    gate: *const f32,
    up: *const f32,
    dst: *mut f32,
) -> i32 {
    if gate.is_null() || up.is_null() || dst.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::swiglu_f32(n, gate, up, dst);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates streaming on-the-fly FP8 (E4M3) Matrix-Vector multiplication.
///
/// # Safety
/// Pointers must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zc_gemv_fp8(
    ptr_weight: *const u8,
    scale: f32,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    if ptr_weight.is_null() || p_act.is_null() || p_dst.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::gemv_fp8(ptr_weight, scale, p_act, p_dst, n_rows, k_cols);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates streaming on-the-fly Q4_K Matrix-Vector multiplication.
///
/// # Safety
/// Pointers must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zc_gemv_q4_k(
    ptr_weight: *const u8,
    p_act: *const f32,
    p_dst: *mut f32,
    n_rows: usize,
    k_cols: usize,
) -> i32 {
    if ptr_weight.is_null() || p_act.is_null() || p_dst.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::gemv_q4_k(ptr_weight, p_act, p_dst, n_rows, k_cols);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates Rotary Position Embedding (RoPE) in-place on tensor data.
///
/// # Safety
/// Pointer `x` must point to valid allocation of length `num_heads * seq_len * head_dim`.
#[no_mangle]
pub unsafe extern "C" fn zc_rope_f32(
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    pos_offset: usize,
    theta_base: f32,
    x: *mut f32,
) -> i32 {
    if x.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::rope_f32(num_heads, seq_len, head_dim, pos_offset, theta_base, x);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

// ============================================================================
// PagedAttention & End-to-End Transformer Decoder Layer
// ============================================================================

/// Evaluates vLLM-style PagedAttention over non-contiguous KV-Cache blocks.
///
/// # Safety
/// Pointers must point to valid allocations matching the specified dimensions.
#[no_mangle]
pub unsafe extern "C" fn zc_paged_attention_f32(
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    context_len: usize,
    scale: f32,
    q: *const f32,
    k_cache: *const f32,
    v_cache: *const f32,
    block_table: *const i32,
    out: *mut f32,
) -> i32 {
    if q.is_null() || k_cache.is_null() || v_cache.is_null() || block_table.is_null() || out.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::paged_attention_f32(
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            context_len,
            scale,
            q,
            k_cache,
            v_cache,
            block_table,
            out,
        );
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates an End-to-End FP32 Transformer Decoder Layer for single-token decoding at `pos`.
///
/// # Safety
/// All pointers must point to valid allocations matching the layer dimensions.
#[no_mangle]
pub unsafe extern "C" fn zc_transformer_layer_f32(
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    rms_eps: f32,
    rope_theta: f32,
    x: *mut f32,
    rms_attn_w: *const f32,
    wq: *const f32,
    wk: *const f32,
    wv: *const f32,
    wo: *const f32,
    rms_ffn_w: *const f32,
    w_gate: *const f32,
    w_up: *const f32,
    w_down: *const f32,
    k_cache: *mut f32,
    v_cache: *mut f32,
) -> i32 {
    if x.is_null()
        || rms_attn_w.is_null()
        || wq.is_null()
        || wk.is_null()
        || wv.is_null()
        || wo.is_null()
        || rms_ffn_w.is_null()
        || w_gate.is_null()
        || w_up.is_null()
        || w_down.is_null()
        || k_cache.is_null()
        || v_cache.is_null()
    {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::transformer_layer_f32(
            hidden_dim,
            intermediate_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            pos,
            rms_eps,
            rope_theta,
            x,
            rms_attn_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_gate,
            w_up,
            w_down,
            k_cache,
            v_cache,
        );
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates an End-to-End Quantized Q4_0 Transformer Decoder Layer for single-token decoding at `pos`.
///
/// # Safety
/// All pointers must point to valid allocations matching the layer dimensions.
#[no_mangle]
pub unsafe extern "C" fn zc_transformer_layer_q4_0(
    hidden_dim: usize,
    intermediate_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    pos: usize,
    rms_eps: f32,
    rope_theta: f32,
    x: *mut f32,
    rms_attn_w: *const f32,
    wq_q4: *const u8,
    wk_q4: *const u8,
    wv_q4: *const u8,
    wo_q4: *const u8,
    rms_ffn_w: *const f32,
    w_gate_q4: *const u8,
    w_up_q4: *const u8,
    w_down_q4: *const u8,
    k_cache: *mut f32,
    v_cache: *mut f32,
) -> i32 {
    if x.is_null()
        || rms_attn_w.is_null()
        || wq_q4.is_null()
        || wk_q4.is_null()
        || wv_q4.is_null()
        || wo_q4.is_null()
        || rms_ffn_w.is_null()
        || w_gate_q4.is_null()
        || w_up_q4.is_null()
        || w_down_q4.is_null()
        || k_cache.is_null()
        || v_cache.is_null()
    {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        zero_tensor::transformer_layer_q4_0(
            hidden_dim,
            intermediate_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            pos,
            rms_eps,
            rope_theta,
            x,
            rms_attn_w,
            wq_q4,
            wk_q4,
            wv_q4,
            wo_q4,
            rms_ffn_w,
            w_gate_q4,
            w_up_q4,
            w_down_q4,
            k_cache,
            v_cache,
        );
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates high-performance Int8 dot product with Int32 accumulation.
///
/// # Safety
/// Pointers `a`, `b`, and `out` must point to valid allocations.
#[no_mangle]
pub unsafe extern "C" fn zc_dot_i8(n: usize, a: *const i8, b: *const i8, out: *mut i32) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        *out = zero_simd::dot_i8(n, a, b);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

