/**
 * @file zero_compute.h
 * @brief High-Performance Native Compute Substrate C-ABI Header for ZeroUniverse.
 * @author ZeroUniverse Architecture Team
 * @copyright MIT License
 */

#ifndef ZERO_COMPUTE_H
#define ZERO_COMPUTE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return / Status Codes */
#define ZC_SUCCESS                 0
#define ZC_ERR_NULL_POINTER       -1
#define ZC_ERR_INVALID_DIMENSION  -2
#define ZC_ERR_UNSUPPORTED        -3
#define ZC_ERR_PANIC             -99

/**
 * @brief Returns the C-ABI version of ZeroCompute (e.g. 100 for 1.0.0).
 */
uint32_t zc_version(void);

/**
 * @brief Returns 1 if AVX2 and FMA hardware SIMD acceleration is supported, 0 otherwise.
 */
int32_t zc_is_simd_supported(void);

/**
 * @brief Evaluates General Matrix Multiplication (GEMM): C = alpha * (A @ B) + beta * C.
 * 
 * Supports arbitrary row and column strides for A, B, and C.
 *
 * @param m Number of rows in A and C.
 * @param n Number of columns in B and C.
 * @param k Number of columns in A / rows in B.
 * @param alpha Scalar multiplier for A @ B.
 * @param a Pointer to matrix A buffer (FP32).
 * @param rsa Row stride of A.
 * @param csa Column stride of A.
 * @param b Pointer to matrix B buffer (FP32).
 * @param rsb Row stride of B.
 * @param csb Column stride of B.
 * @param beta Scalar multiplier for matrix C.
 * @param c Pointer to matrix C destination buffer (FP32).
 * @param rsc Row stride of C.
 * @param csc Column stride of C.
 * @param parallel 1 to enable multi-threaded execution, 0 for sequential execution.
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_gemm_f32(
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    const float* a,
    intptr_t rsa,
    intptr_t csa,
    const float* b,
    intptr_t rsb,
    intptr_t csb,
    float beta,
    float* c,
    intptr_t rsc,
    intptr_t csc,
    int32_t parallel
);

/**
 * @brief Evaluates FlashAttention-2 with online softmax and 2D block-tiling.
 *
 * @param batch_count Number of attention batches (e.g. batch_size * num_heads).
 * @param sq Query sequence length.
 * @param sk Key/Value sequence length.
 * @param d Head dimension for Q and K.
 * @param dv Head dimension for V and Out.
 * @param scale_factor Softmax scaling factor (typically 1.0 / sqrt(d)).
 * @param is_gqa 1 if Grouped Query Attention is active, 0 otherwise.
 * @param hq Number of query heads.
 * @param hkv Number of key/value heads.
 * @param gqa_group GQA group size (hq / hkv).
 * @param q Pointer to Query matrix buffer.
 * @param k Pointer to Key matrix buffer.
 * @param v Pointer to Value matrix buffer.
 * @param mask Pointer to attention mask buffer (or NULL if unmasked).
 * @param out Pointer to destination output buffer.
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_flash_attention_f32(
    size_t batch_count,
    size_t sq,
    size_t sk,
    size_t d,
    size_t dv,
    float scale_factor,
    int32_t is_gqa,
    size_t hq,
    size_t hkv,
    size_t gqa_group,
    const float* q,
    const float* k,
    const float* v,
    const float* mask,
    float* out
);

/**
 * @brief Evaluates streaming on-the-fly Q4_0 Matrix-Vector multiplication (GEMV).
 *
 * Directly consumes raw 4-bit quantized byte streams without dequantizing into DRAM.
 *
 * @param ptr_weight Pointer to raw Q4_0 block bytes (18 bytes per 32 weights).
 * @param p_act Pointer to activation vector buffer (FP32).
 * @param p_dst Pointer to destination vector buffer (FP32).
 * @param n_rows Number of rows in weight matrix.
 * @param k_cols Number of columns in weight matrix (must be multiple of 32).
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_gemv_q4_0(
    const uint8_t* ptr_weight,
    const float* p_act,
    float* p_dst,
    size_t n_rows,
    size_t k_cols
);

/**
 * @brief Evaluates streaming on-the-fly Q8_0 Matrix-Vector multiplication (GEMV).
 *
 * Directly consumes raw 8-bit quantized byte streams (34 bytes per 32 weights).
 *
 * @param ptr_weight Pointer to raw Q8_0 block bytes.
 * @param p_act Pointer to activation vector buffer (FP32).
 * @param p_dst Pointer to destination vector buffer (FP32).
 * @param n_rows Number of rows in weight matrix.
 * @param k_cols Number of columns in weight matrix (must be multiple of 32).
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_gemv_q8_0(
    const uint8_t* ptr_weight,
    const float* p_act,
    float* p_dst,
    size_t n_rows,
    size_t k_cols
);

/**
 * @brief Evaluates Root Mean Square Normalization (RMSNorm).
 *
 * dst[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
 *
 * @param n Number of elements.
 * @param x Input vector buffer (FP32).
 * @param weight Scaling weight vector buffer (FP32).
 * @param dst Destination vector buffer (FP32).
 * @param eps Epsilon value for numerical stability (e.g. 1e-5f).
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_rmsnorm_f32(
    size_t n,
    const float* x,
    const float* weight,
    float* dst,
    float eps
);

/**
 * @brief Evaluates Vectorized SiLU activation: dst[i] = x[i] / (1 + exp(-x[i])).
 *
 * @param n Number of elements.
 * @param src Input vector buffer (FP32).
 * @param dst Destination vector buffer (FP32).
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_silu_f32(
    size_t n,
    const float* src,
    float* dst
);

/**
 * @brief Evaluates Vectorized SwiGLU activation: dst[i] = silu(gate[i]) * up[i].
 *
 * @param n Number of elements.
 * @param gate Input gate vector buffer (FP32).
 * @param up Input up vector buffer (FP32).
 * @param dst Destination vector buffer (FP32).
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_swiglu_f32(
    size_t n,
    const float* gate,
    const float* up,
    float* dst
);

/**
 * @brief Evaluates streaming on-the-fly FP8 (E4M3) Matrix-Vector multiplication.
 *
 * @param ptr_weight Pointer to raw FP8 weights.
 * @param scale FP8 dequantization scale factor.
 * @param p_act Pointer to activation vector buffer (FP32).
 * @param p_dst Pointer to destination vector buffer (FP32).
 * @param n_rows Number of rows in weight matrix.
 * @param k_cols Number of columns in weight matrix.
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_gemv_fp8(
    const uint8_t* ptr_weight,
    float scale,
    const float* p_act,
    float* p_dst,
    size_t n_rows,
    size_t k_cols
);

/**
 * @brief Evaluates streaming on-the-fly Q4_K Matrix-Vector multiplication.
 *
 * @param ptr_weight Pointer to raw Q4_K super-block bytes (144 bytes per 256 weights).
 * @param p_act Pointer to activation vector buffer (FP32).
 * @param p_dst Pointer to destination vector buffer (FP32).
 * @param n_rows Number of rows in weight matrix.
 * @param k_cols Number of columns in weight matrix (must be multiple of 256).
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_gemv_q4_k(
    const uint8_t* ptr_weight,
    const float* p_act,
    float* p_dst,
    size_t n_rows,
    size_t k_cols
);

/**
 * @brief Evaluates Rotary Position Embedding (RoPE) in-place on tensor data.
 *
 * Shape: [num_heads, seq_len, head_dim].
 *
 * @param num_heads Number of attention heads.
 * @param seq_len Number of tokens in sequence.
 * @param head_dim Dimension per head (must be even).
 * @param pos_offset Token starting position offset.
 * @param theta_base Rotary base frequency (e.g. 10000.0f or 500000.0f).
 * @param x Pointer to tensor buffer (FP32).
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_rope_f32(
    size_t num_heads,
    size_t seq_len,
    size_t head_dim,
    size_t pos_offset,
    float theta_base,
    float* x
);

/**
 * @brief Evaluates vLLM-style PagedAttention over non-contiguous KV-Cache blocks.
 *
 * @param num_heads Number of query heads.
 * @param num_kv_heads Number of key/value heads (supports GQA).
 * @param head_dim Dimension per attention head.
 * @param block_size Number of tokens per physical KV block.
 * @param context_len Total valid tokens in context.
 * @param scale Attention scaling factor (1.0f / sqrt(head_dim)).
 * @param q Query vector [num_heads, head_dim].
 * @param k_cache Physical Key block pool [num_blocks, block_size, num_kv_heads, head_dim].
 * @param v_cache Physical Value block pool [num_blocks, block_size, num_kv_heads, head_dim].
 * @param block_table Logical-to-physical block index mapping table.
 * @param out Output attention vector [num_heads, head_dim].
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_paged_attention_f32(
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t block_size,
    size_t context_len,
    float scale,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    const int32_t* block_table,
    float* out
);

/**
 * @brief Evaluates an End-to-End FP32 Transformer Decoder Layer for single-token decoding at `pos`.
 *
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_transformer_layer_f32(
    size_t hidden_dim,
    size_t intermediate_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t pos,
    float rms_eps,
    float rope_theta,
    float* x,
    const float* rms_attn_w,
    const float* wq,
    const float* wk,
    const float* wv,
    const float* wo,
    const float* rms_ffn_w,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* k_cache,
    float* v_cache
);

/**
 * @brief Evaluates an End-to-End Quantized Q4_0 Transformer Decoder Layer for single-token decoding at `pos`.
 *
 * @return int32_t ZC_SUCCESS (0) on success, or negative error code.
 */
int32_t zc_transformer_layer_q4_0(
    size_t hidden_dim,
    size_t intermediate_dim,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t pos,
    float rms_eps,
    float rope_theta,
    float* x,
    const float* rms_attn_w,
    const uint8_t* wq_q4,
    const uint8_t* wk_q4,
    const uint8_t* wv_q4,
    const uint8_t* wo_q4,
    const float* rms_ffn_w,
    const uint8_t* w_gate_q4,
    const uint8_t* w_up_q4,
    const uint8_t* w_down_q4,
    float* k_cache,
    float* v_cache
);

/**
 * @brief Evaluates high-performance Int8 dot product with Int32 accumulation.
 */
int32_t zc_dot_i8(
    size_t n,
    const int8_t* a,
    const int8_t* b,
    int32_t* out
);

/**
 * @brief Evaluates high-performance FP32 dot product using AVX-512 / AVX2+FMA / NEON.
 */
int32_t zc_dot_f32(
    size_t n,
    const float* a,
    const float* b,
    float* out
);

/**
 * @brief Evaluates FP32 Cosine Similarity between two vectors `a` and `b` of length `n`.
 */
int32_t zc_cosine_similarity_f32(
    size_t n,
    const float* a,
    const float* b,
    float* out
);

/**
 * @brief Normalizes an FP32 vector `x` in-place to unit L2 norm.
 */
int32_t zc_l2_normalize_f32(
    size_t n,
    float* x,
    float* out_norm
);

/**
 * @brief Evaluates batch dot-product / cosine similarity between `[n_rows, dim]` matrix and `query`.
 */
int32_t zc_batch_dot_f32(
    size_t n_rows,
    size_t dim,
    const float* matrix,
    const float* query,
    float* out_scores
);

/**
 * @brief Evaluates zero-allocation 64-bit FNV-1a token set Jaccard similarity and Recall over UTF-8 slices.
 */
int32_t zc_jaccard_token_overlap_utf8(
    const uint8_t* a_ptr,
    size_t a_len,
    const uint8_t* b_ptr,
    size_t b_len,
    float* out_jaccard,
    float* out_recall
);


/* ========================================================================= */
/* Backward Compatibility Aliases for zerorust_*                             */
/* ========================================================================= */
uint32_t zerorust_version(void);
int32_t zerorust_is_simd_supported(void);
int32_t zerorust_gemm_f32(
    size_t m, size_t n, size_t k, float alpha,
    const float* a, intptr_t rsa, intptr_t csa,
    const float* b, intptr_t rsb, intptr_t csb,
    float beta, float* c, intptr_t rsc, intptr_t csc,
    int32_t parallel
);
int32_t zerorust_flash_attention_f32(
    size_t batch_count, size_t sq, size_t sk, size_t d, size_t dv,
    float scale_factor, int32_t is_gqa,
    size_t hq, size_t hkv, size_t gqa_group,
    const float* q, const float* k, const float* v, const float* mask,
    float* out
);
int32_t zerorust_gemv_q4_0(
    const uint8_t* ptr_weight, const float* p_act, float* p_dst,
    size_t n_rows, size_t k_cols
);
int32_t zerorust_gemv_q8_0(
    const uint8_t* ptr_weight, const float* p_act, float* p_dst,
    size_t n_rows, size_t k_cols
);

#ifdef __cplusplus
}
#endif

#endif /* ZERO_COMPUTE_H */
