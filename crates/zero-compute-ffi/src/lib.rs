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

// ============================================================================
// Vector Similarity, Batch Retrieval & Token Overlap (AI / RAG / ZFlow Kernels)
// ============================================================================

/// Evaluates high-performance FP32 dot product using AVX-512 / AVX2+FMA / NEON.
///
/// # Safety
/// Pointers `a`, `b`, and `out` must point to valid allocations of length `n`.
#[no_mangle]
pub unsafe extern "C" fn zc_dot_f32(n: usize, a: *const f32, b: *const f32, out: *mut f32) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        *out = zero_simd::dot_f32(n, a, b);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates FP32 Cosine Similarity between two vectors `a` and `b` of length `n`.
/// Automatically normalizes by `sqrt(||a||^2 * ||b||^2)` and clamps to `[-1.0, 1.0]`.
///
/// # Safety
/// Pointers `a`, `b`, and `out` must point to valid allocations of length `n`.
#[no_mangle]
pub unsafe extern "C" fn zc_cosine_similarity_f32(
    n: usize,
    a: *const f32,
    b: *const f32,
    out: *mut f32,
) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() {
        return ZC_ERR_NULL_POINTER;
    }
    if n == 0 {
        *out = 0.0;
        return ZC_SUCCESS;
    }

    let res = catch_unwind(|| {
        let dot = zero_simd::dot_f32(n, a, b);
        let norm_a = zero_simd::dot_f32(n, a, a);
        let norm_b = zero_simd::dot_f32(n, b, b);
        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-12 {
            *out = (dot / denom).clamp(-1.0, 1.0);
        } else {
            *out = 0.0;
        }
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Normalizes an FP32 vector `x` of length `n` in-place to unit L2 norm (`||x||_2 = 1.0`).
/// Writes the original L2 norm to `out_norm` if non-null.
///
/// # Safety
/// Pointer `x` must point to a valid mutable allocation of length `n`.
#[no_mangle]
pub unsafe extern "C" fn zc_l2_normalize_f32(
    n: usize,
    x: *mut f32,
    out_norm: *mut f32,
) -> i32 {
    if x.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        let norm_sq = zero_simd::dot_f32(n, x, x);
        let norm = norm_sq.sqrt();
        if !out_norm.is_null() {
            *out_norm = norm;
        }
        if norm > 1e-12 {
            let inv = 1.0f32 / norm;
            for i in 0..n {
                *x.add(i) *= inv;
            }
        }
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

/// Evaluates batch dot-product / cosine similarity between a contiguous `[n_rows, dim]`
/// matrix of L2-normalized embeddings and a `dim`-dimensional query vector.
///
/// # Safety
/// `matrix` must have `n_rows * dim` elements, `query` `dim` elements, and `out_scores` `n_rows` elements.
#[no_mangle]
pub unsafe extern "C" fn zc_batch_dot_f32(
    n_rows: usize,
    dim: usize,
    matrix: *const f32,
    query: *const f32,
    out_scores: *mut f32,
) -> i32 {
    if matrix.is_null() || query.is_null() || out_scores.is_null() {
        return ZC_ERR_NULL_POINTER;
    }
    if dim == 0 {
        return ZC_ERR_INVALID_DIMENSION;
    }

    let res = catch_unwind(|| {
        for r in 0..n_rows {
            let row_ptr = matrix.add(r * dim);
            let score = zero_simd::dot_f32(dim, row_ptr, query).clamp(-1.0, 1.0);
            *out_scores.add(r) = score;
        }
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}

#[inline(always)]
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'+' || b == b'#' || b >= 0x80
}

#[inline(always)]
fn ascii_lower(b: u8) -> u8 {
    if b >= b'A' && b <= b'Z' {
        b + 32
    } else {
        b
    }
}

const MAX_TOKEN_HASHES: usize = 2048;

fn extract_fnv64_token_set(bytes: &[u8], out_hashes: &mut [u64; MAX_TOKEN_HASHES]) -> usize {
    let mut raw_count = 0usize;
    let mut i = 0usize;
    let len = bytes.len();

    while i < len && raw_count < MAX_TOKEN_HASHES {
        while i < len && !is_token_byte(bytes[i]) {
            i += 1;
        }
        if i >= len {
            break;
        }
        let start = i;
        let mut hash: u64 = 0xcbf29ce484222325;
        while i < len && is_token_byte(bytes[i]) {
            hash ^= ascii_lower(bytes[i]) as u64;
            hash = hash.wrapping_mul(0x100000001b3);
            i += 1;
        }
        if i - start >= 1 {
            out_hashes[raw_count] = hash;
            raw_count += 1;
        }
    }

    if raw_count <= 1 {
        return raw_count;
    }

    // Sort in-place and deduplicate in O(N log N) instead of O(N^2)
    let slice = &mut out_hashes[..raw_count];
    slice.sort_unstable();
    let mut write_idx = 1usize;
    for read_idx in 1..raw_count {
        if slice[read_idx] != slice[write_idx - 1] {
            slice[write_idx] = slice[read_idx];
            write_idx += 1;
        }
    }
    write_idx
}

/// Evaluates zero-allocation FNV-1a 64-bit token set Jaccard similarity (`|A ∩ B| / |A ∪ B|`)
/// and Recall overlap (`|A ∩ B| / |A|`) over two UTF-8 byte slices using O(N log N) sort
/// and O(N_A + N_B) two-pointer merge intersection (up to 2048 unique tokens per slice).
///
/// # Safety
/// Pointers `a_ptr`, `b_ptr`, `out_jaccard`, and `out_recall` must be valid.
#[no_mangle]
pub unsafe extern "C" fn zc_jaccard_token_overlap_utf8(
    a_ptr: *const u8,
    a_len: usize,
    b_ptr: *const u8,
    b_len: usize,
    out_jaccard: *mut f32,
    out_recall: *mut f32,
) -> i32 {
    if a_ptr.is_null() || b_ptr.is_null() || out_jaccard.is_null() || out_recall.is_null() {
        return ZC_ERR_NULL_POINTER;
    }

    let res = catch_unwind(|| {
        let a_slice = core::slice::from_raw_parts(a_ptr, a_len);
        let b_slice = core::slice::from_raw_parts(b_ptr, b_len);

        let mut set_a = [0u64; MAX_TOKEN_HASHES];
        let mut set_b = [0u64; MAX_TOKEN_HASHES];
        let na = extract_fnv64_token_set(a_slice, &mut set_a);
        let nb = extract_fnv64_token_set(b_slice, &mut set_b);

        if na == 0 || nb == 0 {
            *out_jaccard = 0.0;
            *out_recall = 0.0;
            return;
        }

        // O(na + nb) two-pointer intersection on sorted unique hashes
        let mut ia = 0usize;
        let mut ib = 0usize;
        let mut inter = 0usize;
        while ia < na && ib < nb {
            let ha = set_a[ia];
            let hb = set_b[ib];
            if ha == hb {
                inter += 1;
                ia += 1;
                ib += 1;
            } else if ha < hb {
                ia += 1;
            } else {
                ib += 1;
            }
        }

        let union = na + nb - inter;
        *out_jaccard = if union > 0 {
            (inter as f32) / (union as f32)
        } else {
            0.0
        };
        *out_recall = (inter as f32) / (na as f32);
    });

    match res {
        Ok(_) => ZC_SUCCESS,
        Err(_) => ZC_ERR_PANIC,
    }
}


