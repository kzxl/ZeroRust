//! End-to-End Fused LLM Transformer Decoder Layer (LLaMA / Qwen / Gemma Architecture).
//!
//! Combines the entire Transformer decoder block into a single native execution pass:
//! 1. Pre-Attention RMSNorm
//! 2. Q, K, V Linear Projections (FP32 GEMV or Streaming Q4_0 Quantized GEMV)
//! 3. In-place Rotary Position Embedding (RoPE) on Q and K
//! 4. KV-Cache Append & Grouped-Query Online-Softmax Attention
//! 5. Output Projection (Wo) + Residual Connection
//! 6. Post-Attention FFN RMSNorm
//! 7. Gate & Up Projections (W_gate, W_up) + Fused SwiGLU Activation
//! 8. Down Projection (W_down) + Residual Connection

use crate::ops::{rmsnorm_f32, rope_f32, swiglu_f32};
use crate::quant::gemv_q4_0;

/// Evaluates a single FP32 Matrix-Vector multiplication: y[rows] = W[rows, cols] * x[cols].
#[inline(always)]
unsafe fn matvec_f32(w: *const f32, x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    #[cfg(feature = "std")]
    if rows * cols >= 16384 {
        use rayon::prelude::*;
        let w_addr = w as usize;
        let x_addr = x as usize;
        let y_addr = y as usize;
        (0..rows).into_par_iter().for_each(|r| {
            let p_w = (w_addr as *const f32).add(r * cols);
            let p_x = x_addr as *const f32;
            let p_y = (y_addr as *mut f32).add(r);
            *p_y = zero_simd::dot_f32(cols, p_w, p_x);
        });
        return;
    }

    for r in 0..rows {
        *y.add(r) = zero_simd::dot_f32(cols, w.add(r * cols), x);
    }
}

/// Evaluates an End-to-End FP32 Transformer Decoder Layer for single-token decoding at `pos`.
///
/// Updates hidden state `x` (`[hidden_dim]`) in-place and appends current K/V into `k_cache` and `v_cache`
/// (`[max_seq_len, num_kv_heads * head_dim]`).
///
/// # Safety
/// All pointers must reference valid memory buffers matching the layer dimensions.
#[cfg(feature = "std")]
pub unsafe fn transformer_layer_f32(
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
) {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let gqa_ratio = num_heads / num_kv_heads;
    let seq_len = pos + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut x_norm = vec![0.0f32; hidden_dim];
    let mut q = vec![0.0f32; q_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut proj_out = vec![0.0f32; hidden_dim];
    let mut gate_buf = vec![0.0f32; intermediate_dim];
    let mut up_buf = vec![0.0f32; intermediate_dim];
    let mut swiglu_buf = vec![0.0f32; intermediate_dim];

    // 1. Pre-Attention RMSNorm
    rmsnorm_f32(hidden_dim, x, rms_attn_w, x_norm.as_mut_ptr(), rms_eps);

    // 2. Q, K, V Projections directly into Q buffer and KV-Cache slot at `pos`
    let k_slot = k_cache.add(pos * kv_dim);
    let v_slot = v_cache.add(pos * kv_dim);
    matvec_f32(wq, x_norm.as_ptr(), q.as_mut_ptr(), q_dim, hidden_dim);
    matvec_f32(wk, x_norm.as_ptr(), k_slot, kv_dim, hidden_dim);
    matvec_f32(wv, x_norm.as_ptr(), v_slot, kv_dim, hidden_dim);

    // 3. Apply RoPE to Q and current K slot
    rope_f32(num_heads, 1, head_dim, pos, rope_theta, q.as_mut_ptr());
    rope_f32(num_kv_heads, 1, head_dim, pos, rope_theta, k_slot);

    // 4. Multi-Head / Grouped-Query Online-Softmax Attention over [0..=pos]
    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        let p_q = q.as_ptr().add(h * head_dim);
        let p_out = attn_out.as_mut_ptr().add(h * head_dim);

        let mut m_prev = f32::NEG_INFINITY;
        let mut l_prev = 0.0f32;

        for t in 0..seq_len {
            let p_k = k_cache.add(t * kv_dim + kv_h * head_dim);
            let p_v = v_cache.add(t * kv_dim + kv_h * head_dim);

            let score = zero_simd::dot_f32(head_dim, p_q, p_k) * scale;
            let m_curr = if score > m_prev { score } else { m_prev };
            let alpha = zero_simd::fast_exp(m_prev - m_curr);
            let beta = zero_simd::fast_exp(score - m_curr);
            l_prev = l_prev * alpha + beta;

            for d in 0..head_dim {
                *p_out.add(d) = *p_out.add(d) * alpha + beta * *p_v.add(d);
            }
            m_prev = m_curr;
        }

        if l_prev > 0.0 {
            let inv_l = 1.0f32 / l_prev;
            for d in 0..head_dim {
                *p_out.add(d) *= inv_l;
            }
        }
    }

    // 5. Output Projection (Wo) + Residual Connection
    matvec_f32(wo, attn_out.as_ptr(), proj_out.as_mut_ptr(), hidden_dim, q_dim);
    for i in 0..hidden_dim {
        *x.add(i) += proj_out[i];
    }

    // 6. Post-Attention FFN RMSNorm
    rmsnorm_f32(hidden_dim, x, rms_ffn_w, x_norm.as_mut_ptr(), rms_eps);

    // 7. FFN Gate & Up Projections + SwiGLU
    matvec_f32(w_gate, x_norm.as_ptr(), gate_buf.as_mut_ptr(), intermediate_dim, hidden_dim);
    matvec_f32(w_up, x_norm.as_ptr(), up_buf.as_mut_ptr(), intermediate_dim, hidden_dim);
    swiglu_f32(
        intermediate_dim,
        gate_buf.as_ptr(),
        up_buf.as_ptr(),
        swiglu_buf.as_mut_ptr(),
    );

    // 8. FFN Down Projection + Residual Connection
    matvec_f32(w_down, swiglu_buf.as_ptr(), proj_out.as_mut_ptr(), hidden_dim, intermediate_dim);
    for i in 0..hidden_dim {
        *x.add(i) += proj_out[i];
    }
}

/// Evaluates an End-to-End Quantized Q4_0 Transformer Decoder Layer for single-token decoding at `pos`.
///
/// Consumes raw Q4_0 block weights directly from GGUF memory without dequantizing to FP32 DRAM.
///
/// # Safety
/// All pointers must reference valid memory buffers; `hidden_dim`, `q_dim`, and `intermediate_dim` must be multiples of 32.
#[cfg(feature = "std")]
pub unsafe fn transformer_layer_q4_0(
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
) {
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let gqa_ratio = num_heads / num_kv_heads;
    let seq_len = pos + 1;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut x_norm = vec![0.0f32; hidden_dim];
    let mut q = vec![0.0f32; q_dim];
    let mut attn_out = vec![0.0f32; q_dim];
    let mut proj_out = vec![0.0f32; hidden_dim];
    let mut gate_buf = vec![0.0f32; intermediate_dim];
    let mut up_buf = vec![0.0f32; intermediate_dim];
    let mut swiglu_buf = vec![0.0f32; intermediate_dim];

    // 1. Pre-Attention RMSNorm
    rmsnorm_f32(hidden_dim, x, rms_attn_w, x_norm.as_mut_ptr(), rms_eps);

    // 2. Streaming Q4_0 Q, K, V Projections
    let k_slot = k_cache.add(pos * kv_dim);
    let v_slot = v_cache.add(pos * kv_dim);
    gemv_q4_0(wq_q4, x_norm.as_ptr(), q.as_mut_ptr(), q_dim, hidden_dim);
    gemv_q4_0(wk_q4, x_norm.as_ptr(), k_slot, kv_dim, hidden_dim);
    gemv_q4_0(wv_q4, x_norm.as_ptr(), v_slot, kv_dim, hidden_dim);

    // 3. Apply RoPE to Q and K
    rope_f32(num_heads, 1, head_dim, pos, rope_theta, q.as_mut_ptr());
    rope_f32(num_kv_heads, 1, head_dim, pos, rope_theta, k_slot);

    // 4. Grouped-Query Online-Softmax Attention
    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        let p_q = q.as_ptr().add(h * head_dim);
        let p_out = attn_out.as_mut_ptr().add(h * head_dim);

        let mut m_prev = f32::NEG_INFINITY;
        let mut l_prev = 0.0f32;

        for t in 0..seq_len {
            let p_k = k_cache.add(t * kv_dim + kv_h * head_dim);
            let p_v = v_cache.add(t * kv_dim + kv_h * head_dim);

            let score = zero_simd::dot_f32(head_dim, p_q, p_k) * scale;
            let m_curr = if score > m_prev { score } else { m_prev };
            let alpha = zero_simd::fast_exp(m_prev - m_curr);
            let beta = zero_simd::fast_exp(score - m_curr);
            l_prev = l_prev * alpha + beta;

            for d in 0..head_dim {
                *p_out.add(d) = *p_out.add(d) * alpha + beta * *p_v.add(d);
            }
            m_prev = m_curr;
        }

        if l_prev > 0.0 {
            let inv_l = 1.0f32 / l_prev;
            for d in 0..head_dim {
                *p_out.add(d) *= inv_l;
            }
        }
    }

    // 5. Streaming Q4_0 Output Projection + Residual
    gemv_q4_0(wo_q4, attn_out.as_ptr(), proj_out.as_mut_ptr(), hidden_dim, q_dim);
    for i in 0..hidden_dim {
        *x.add(i) += proj_out[i];
    }

    // 6. Post-Attention FFN RMSNorm
    rmsnorm_f32(hidden_dim, x, rms_ffn_w, x_norm.as_mut_ptr(), rms_eps);

    // 7. Streaming Q4_0 Gate & Up + SwiGLU
    gemv_q4_0(w_gate_q4, x_norm.as_ptr(), gate_buf.as_mut_ptr(), intermediate_dim, hidden_dim);
    gemv_q4_0(w_up_q4, x_norm.as_ptr(), up_buf.as_mut_ptr(), intermediate_dim, hidden_dim);
    swiglu_f32(
        intermediate_dim,
        gate_buf.as_ptr(),
        up_buf.as_ptr(),
        swiglu_buf.as_mut_ptr(),
    );

    // 8. Streaming Q4_0 Down Projection + Residual
    gemv_q4_0(w_down_q4, swiglu_buf.as_ptr(), proj_out.as_mut_ptr(), hidden_dim, intermediate_dim);
    for i in 0..hidden_dim {
        *x.add(i) += proj_out[i];
    }
}
