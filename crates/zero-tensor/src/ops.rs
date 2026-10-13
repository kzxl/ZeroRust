//! High-level tensor neural operators (RMSNorm, SiLU, SwiGLU).

use zero_simd::{norm, fast_math};

/// Evaluates Root Mean Square Normalization across a vector:
/// dst[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
pub unsafe fn rmsnorm_f32(
    n: usize,
    x: *const f32,
    weight: *const f32,
    dst: *mut f32,
    eps: f32,
) {
    norm::rms_norm(n, x, weight, dst, eps);
}

/// Evaluates Vectorized SiLU activation: dst[i] = x[i] / (1 + exp(-x[i])).
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
pub unsafe fn silu_f32(n: usize, src: *const f32, dst: *mut f32) {
    fast_math::vector_silu(n, src, dst);
}

/// Evaluates Vectorized SwiGLU: dst[i] = silu(gate[i]) * up[i].
///
/// # Safety
/// Pointers must point to valid allocations of length `n`.
pub unsafe fn swiglu_f32(n: usize, gate: *const f32, up: *const f32, dst: *mut f32) {
    if n == 0 {
        return;
    }
    #[cfg(feature = "std")]
    {
        use rayon::prelude::*;
        let gate_addr = gate as usize;
        let up_addr = up as usize;
        let dst_addr = dst as usize;
        let chunk_size = 8192;
        let num_chunks = (n + chunk_size - 1) / chunk_size;

        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n);
            let g_ptr = (gate_addr as *const f32).add(start);
            let u_ptr = (up_addr as *const f32).add(start);
            let d_ptr = (dst_addr as *mut f32).add(start);
            let len = end - start;
            for i in 0..len {
                let g = *g_ptr.add(i);
                let u = *u_ptr.add(i);
                let silu = g / (1.0 + fast_math::fast_exp(-g));
                *d_ptr.add(i) = silu * u;
            }
        });
    }
    #[cfg(not(feature = "std"))]
    {
        fast_math::vector_swiglu(n, gate, up, dst);
    }
}

/// Evaluates Rotary Position Embedding (RoPE) in-place:
/// Precomputes sinusoidal rotary tables once across all heads,
/// then evaluates rotary embeddings across heads in parallel.
///
/// # Safety
/// Pointers must point to valid allocations of length `num_heads * seq_len * head_dim`.
pub unsafe fn rope_f32(
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    pos_offset: usize,
    theta_base: f32,
    x: *mut f32,
) {
    if num_heads == 0 || seq_len == 0 || head_dim == 0 || head_dim % 2 != 0 {
        return;
    }

    let half_dim = head_dim / 2;

    #[cfg(feature = "std")]
    {
        use rayon::prelude::*;

        // 1. Precalculate inv_freq: size = half_dim
        let mut inv_freq = vec![0.0f32; half_dim];
        for i in 0..half_dim {
            let exp = (2 * i) as f32 / head_dim as f32;
            inv_freq[i] = 1.0f32 / theta_base.powf(exp);
        }

        // 2. Precalculate cos & sin table ONCE for all sequence positions: size = seq_len * half_dim
        let total_table = seq_len * half_dim;
        let mut cos_table = vec![0.0f32; total_table];
        let mut sin_table = vec![0.0f32; total_table];

        for s in 0..seq_len {
            let pos = (pos_offset + s) as f32;
            let s_offset = s * half_dim;
            for i in 0..half_dim {
                let angle = pos * inv_freq[i];
                cos_table[s_offset + i] = angle.cos();
                sin_table[s_offset + i] = angle.sin();
            }
        }

        let cos_ptr = cos_table.as_ptr() as usize;
        let sin_ptr = sin_table.as_ptr() as usize;
        let x_addr = x as usize;

        // 3. Multi-threaded parallel processing across heads using Rayon
        (0..num_heads).into_par_iter().for_each(|h| {
            let p_cos = cos_ptr as *const f32;
            let p_sin = sin_ptr as *const f32;
            let p_base = x_addr as *mut f32;
            let head_offset = h * seq_len * head_dim;

            for s in 0..seq_len {
                let p_vec = p_base.add(head_offset + s * head_dim);
                let s_table_offset = s * half_dim;
                let cur_cos = p_cos.add(s_table_offset);
                let cur_sin = p_sin.add(s_table_offset);

                for i in 0..half_dim {
                    let idx0 = 2 * i;
                    let idx1 = 2 * i + 1;

                    let v0 = *p_vec.add(idx0);
                    let v1 = *p_vec.add(idx1);
                    let c = *cur_cos.add(i);
                    let s_val = *cur_sin.add(i);

                    *p_vec.add(idx0) = v0 * c - v1 * s_val;
                    *p_vec.add(idx1) = v0 * s_val + v1 * c;
                }
            }
        });
    }

    #[cfg(not(feature = "std"))]
    {
        zero_simd::rope_f32(num_heads, seq_len, head_dim, pos_offset, theta_base, x);
    }
}

