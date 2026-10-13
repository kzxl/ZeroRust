//! vLLM-Style PagedAttention with Non-Contiguous Physical KV-Cache Blocks.
//!
//! Eliminates KV-cache memory fragmentation for long-context LLM serving by mapping
//! logical sequence tokens to non-contiguous physical KV blocks via a block table,
//! combined with single-pass online softmax (O(1) intermediate memory).

#[cfg(feature = "std")]
use rayon::prelude::*;

/// Evaluates PagedAttention for single-token decoding over non-contiguous KV-cache pages.
///
/// Memory Layouts:
/// - `q`: `[num_heads, head_dim]`
/// - `k_cache`: `[num_physical_blocks, block_size, num_kv_heads, head_dim]`
/// - `v_cache`: `[num_physical_blocks, block_size, num_kv_heads, head_dim]`
/// - `block_table`: `[num_logical_blocks]` mapping logical block index -> physical block index
/// - `out`: `[num_heads, head_dim]`
///
/// # Safety
/// Pointers must point to valid allocations matching the specified dimensions and block indices.
pub unsafe fn paged_attention_f32(
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
) {
    if num_heads == 0
        || num_kv_heads == 0
        || head_dim == 0
        || block_size == 0
        || context_len == 0
        || num_heads % num_kv_heads != 0
    {
        return;
    }

    let gqa_ratio = num_heads / num_kv_heads;

    #[cfg(feature = "std")]
    if num_heads >= 4 {
        let q_addr = q as usize;
        let k_addr = k_cache as usize;
        let v_addr = v_cache as usize;
        let bt_addr = block_table as usize;
        let out_addr = out as usize;

        (0..num_heads).into_par_iter().for_each(|h| {
            let kv_h = h / gqa_ratio;
            paged_attention_single_head(
                h,
                kv_h,
                num_kv_heads,
                head_dim,
                block_size,
                context_len,
                scale,
                q_addr as *const f32,
                k_addr as *const f32,
                v_addr as *const f32,
                bt_addr as *const i32,
                out_addr as *mut f32,
            );
        });
        return;
    }

    for h in 0..num_heads {
        let kv_h = h / gqa_ratio;
        paged_attention_single_head(
            h,
            kv_h,
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
    }
}

#[inline(always)]
unsafe fn paged_attention_single_head(
    h: usize,
    kv_h: usize,
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
) {
    let p_q = q.add(h * head_dim);
    let p_out = out.add(h * head_dim);

    for d in 0..head_dim {
        *p_out.add(d) = 0.0;
    }

    let mut m_prev = f32::NEG_INFINITY;
    let mut l_prev = 0.0f32;

    let num_logical_blocks = (context_len + block_size - 1) / block_size;

    for lb in 0..num_logical_blocks {
        let phys_block = (*block_table.add(lb)) as usize;
        let tokens_in_block = (context_len - lb * block_size).min(block_size);

        for bo in 0..tokens_in_block {
            let kv_offset = ((phys_block * block_size + bo) * num_kv_heads + kv_h) * head_dim;
            let p_k = k_cache.add(kv_offset);
            let p_v = v_cache.add(kv_offset);

            let dot = zero_simd::dot_f32(head_dim, p_q, p_k);
            let score = dot * scale;

            let m_curr = if score > m_prev { score } else { m_prev };
            let alpha = zero_simd::fast_exp(m_prev - m_curr);
            let beta = zero_simd::fast_exp(score - m_curr);
            l_prev = l_prev * alpha + beta;

            for d in 0..head_dim {
                let prev_val = *p_out.add(d);
                *p_out.add(d) = prev_val * alpha + beta * *p_v.add(d);
            }

            m_prev = m_curr;
        }
    }

    if l_prev > 0.0 {
        let inv_l = 1.0f32 / l_prev;
        for d in 0..head_dim {
            *p_out.add(d) *= inv_l;
        }
    }
}
