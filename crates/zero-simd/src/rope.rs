//! High-performance Rotary Position Embedding (RoPE).

/// Evaluates Rotary Position Embedding (RoPE) in-place on tensor data.
///
/// Standard LLM layout: `[num_heads, seq_len, head_dim]`.
///
/// # Safety
/// Pointer `x` must point to valid allocation of length `num_heads * seq_len * head_dim`.
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

    // Stack-allocated precomputation buffer for inv_freq (supports head_dim up to 512)
    let mut inv_freq_stack = [0.0f32; 256];
    let max_half = half_dim.min(256);
    for i in 0..max_half {
        let exp = (2 * i) as f32 / head_dim as f32;
        #[cfg(feature = "std")]
        {
            inv_freq_stack[i] = 1.0f32 / theta_base.powf(exp);
        }
        #[cfg(not(feature = "std"))]
        {
            inv_freq_stack[i] = 1.0f32 / crate::fast_math::fast_exp(exp * 9.21034f32);
        }
    }

    for h in 0..num_heads {
        let head_offset = h * seq_len * head_dim;

        for s in 0..seq_len {
            let pos = (pos_offset + s) as f32;
            let p_vec = x.add(head_offset + s * head_dim);

            for i in 0..half_dim {
                let freq = if i < 256 {
                    inv_freq_stack[i]
                } else {
                    let exp = (2 * i) as f32 / head_dim as f32;
                    #[cfg(feature = "std")]
                    { 1.0f32 / theta_base.powf(exp) }
                    #[cfg(not(feature = "std"))]
                    { 1.0f32 / crate::fast_math::fast_exp(exp * 9.21034f32) }
                };

                let angle = pos * freq;
                #[cfg(feature = "std")]
                let (sin_val, cos_val) = (angle.sin(), angle.cos());
                #[cfg(not(feature = "std"))]
                let (sin_val, cos_val) = (angle.sin(), angle.cos());

                let idx0 = 2 * i;
                let idx1 = 2 * i + 1;

                let v0 = *p_vec.add(idx0);
                let v1 = *p_vec.add(idx1);

                *p_vec.add(idx0) = v0 * cos_val - v1 * sin_val;
                *p_vec.add(idx1) = v0 * sin_val + v1 * cos_val;
            }
        }
    }
}
