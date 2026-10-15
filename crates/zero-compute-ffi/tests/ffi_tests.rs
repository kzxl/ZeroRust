use zero_compute::*;

#[test]
fn test_ffi_version_and_capabilities() {
    assert_eq!(zc_version(), 100);
    assert_eq!(zerorust_version(), 100);

    let simd = zc_is_simd_supported();
    println!("ZC SIMD Supported: {}", simd);
    assert_eq!(simd, zerorust_is_simd_supported());
}

#[test]
fn test_ffi_gemm() {
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0f32, 8.0, 9.0, 1.0, 2.0, 3.0];
    let mut c = [0.0f32; 4];

    unsafe {
        let status = zc_gemm_f32(
            2,
            2,
            3,
            1.0,
            a.as_ptr(),
            3,
            1,
            b.as_ptr(),
            2,
            1,
            0.0,
            c.as_mut_ptr(),
            2,
            1,
            0,
        );
        assert_eq!(status, ZC_SUCCESS);
    }

    assert_eq!(c[0], 31.0);
    assert_eq!(c[1], 19.0);
    assert_eq!(c[2], 85.0);
    assert_eq!(c[3], 55.0);
}

#[test]
fn test_ffi_rmsnorm() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let w = [1.0f32, 1.0, 1.0, 1.0];
    let mut dst = [0.0f32; 4];

    unsafe {
        let status = zc_rmsnorm_f32(4, x.as_ptr(), w.as_ptr(), dst.as_mut_ptr(), 1e-5);
        assert_eq!(status, ZC_SUCCESS);
    }

    // Mean square = (1 + 4 + 9 + 16) / 4 = 7.5. rms = sqrt(7.5) = 2.7386.
    // dst[0] = 1.0 / 2.7386 = 0.3651
    assert!((dst[0] - 0.3651).abs() < 1e-3);
}

#[test]
fn test_ffi_swiglu() {
    let gate = [0.0f32, 2.0];
    let up = [5.0f32, 10.0];
    let mut dst = [0.0f32; 2];

    unsafe {
        let status = zc_swiglu_f32(2, gate.as_ptr(), up.as_ptr(), dst.as_mut_ptr());
        assert_eq!(status, ZC_SUCCESS);
    }

    // silu(0) = 0 / (1 + 1) = 0. dst[0] = 0 * 5 = 0.
    assert_eq!(dst[0], 0.0);
    // silu(2) = 2 / (1 + exp(-2)) = 2 / (1 + 0.1353) = 1.7616. dst[1] = 1.7616 * 10 = 17.616
    assert!((dst[1] - 17.616).abs() < 0.1);
}

#[test]
fn test_ffi_rope() {
    let mut x = [1.0f32, 0.0, 0.0, 1.0]; // 1 token, 1 head, head_dim = 4
    unsafe {
        let status = zc_rope_f32(1, 1, 4, 0, 10000.0, x.as_mut_ptr());
        assert_eq!(status, ZC_SUCCESS);
    }
    // At pos = 0, angle = 0, cos(0) = 1, sin(0) = 0 => x remains identical
    assert_eq!(x[0], 1.0);
    assert_eq!(x[1], 0.0);
    assert_eq!(x[2], 0.0);
    assert_eq!(x[3], 1.0);
}

#[test]
fn test_ffi_fp8() {
    // 2 rows, 4 cols
    let weights = [0x38u8, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38, 0x38]; // 0x38 in E4M3 is 1.0f32
    let act = [2.0f32, 2.0, 2.0, 2.0];
    let mut dst = [0.0f32; 2];

    unsafe {
        let status = zc_gemv_fp8(weights.as_ptr(), 1.0, act.as_ptr(), dst.as_mut_ptr(), 2, 4);
        assert_eq!(status, ZC_SUCCESS);
    }

    // 4 * (1.0 * 2.0) = 8.0
    assert!((dst[0] - 8.0).abs() < 1e-3);
    assert!((dst[1] - 8.0).abs() < 1e-3);
}

#[test]
fn test_ffi_paged_attention() {
    // 2 query heads, 1 kv head (GQA 2:1), head_dim = 4, block_size = 2, context_len = 3
    // Physical blocks: block 0 and block 1 (mapped in reverse via block_table = [1, 0])
    let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    // 2 physical blocks * 2 tokens/block * 1 kv_head * 4 dim = 16 floats
    let k_cache = [1.0f32; 16];
    let v_cache = [
        // Physical block 0 (logical block 1: token 2)
        2.0f32, 4.0, 6.0, 8.0,
        0.0, 0.0, 0.0, 0.0,
        // Physical block 1 (logical block 0: tokens 0 & 1)
        2.0, 4.0, 6.0, 8.0,
        2.0, 4.0, 6.0, 8.0,
    ];
    let block_table = [1i32, 0];
    let mut out = [0.0f32; 8];

    unsafe {
        let status = zc_paged_attention_f32(
            2,
            1,
            4,
            2,
            3,
            0.5,
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            block_table.as_ptr(),
            out.as_mut_ptr(),
        );
        assert_eq!(status, ZC_SUCCESS);
    }

    // All 3 active tokens have V = [2, 4, 6, 8], so attention weighted average must equal [2, 4, 6, 8]
    for h in 0..2 {
        assert!((out[h * 4] - 2.0).abs() < 1e-3);
        assert!((out[h * 4 + 1] - 4.0).abs() < 1e-3);
        assert!((out[h * 4 + 2] - 6.0).abs() < 1e-3);
        assert!((out[h * 4 + 3] - 8.0).abs() < 1e-3);
    }
}

#[test]
fn test_ffi_transformer_layer_f32() {
    let hidden_dim = 8;
    let intermediate_dim = 16;
    let num_heads = 2;
    let num_kv_heads = 2;
    let head_dim = 4;

    let mut x = vec![1.0f32; hidden_dim];
    let rms_attn_w = vec![1.0f32; hidden_dim];
    let wq = vec![0.1f32; hidden_dim * hidden_dim];
    let wk = vec![0.1f32; hidden_dim * hidden_dim];
    let wv = vec![0.1f32; hidden_dim * hidden_dim];
    let wo = vec![0.1f32; hidden_dim * hidden_dim];
    let rms_ffn_w = vec![1.0f32; hidden_dim];
    let w_gate = vec![0.1f32; intermediate_dim * hidden_dim];
    let w_up = vec![0.1f32; intermediate_dim * hidden_dim];
    let w_down = vec![0.1f32; hidden_dim * intermediate_dim];
    let mut k_cache = vec![0.0f32; 4 * hidden_dim];
    let mut v_cache = vec![0.0f32; 4 * hidden_dim];

    unsafe {
        let status = zc_transformer_layer_f32(
            hidden_dim,
            intermediate_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            0,
            1e-5,
            10000.0,
            x.as_mut_ptr(),
            rms_attn_w.as_ptr(),
            wq.as_ptr(),
            wk.as_ptr(),
            wv.as_ptr(),
            wo.as_ptr(),
            rms_ffn_w.as_ptr(),
            w_gate.as_ptr(),
            w_up.as_ptr(),
            w_down.as_ptr(),
            k_cache.as_mut_ptr(),
            v_cache.as_mut_ptr(),
        );
        assert_eq!(status, ZC_SUCCESS);
    }

    // Verify residual update modified x and produced finite values
    for &val in &x {
        assert!(val.is_finite());
        assert!(val > 1.0);
    }
}

#[test]
fn test_ffi_dot_i8() {
    let a = [2i8; 32];
    let b = [3i8; 32];
    let mut out = 0i32;

    unsafe {
        let status = zc_dot_i8(32, a.as_ptr(), b.as_ptr(), &mut out);
        assert_eq!(status, ZC_SUCCESS);
    }
    assert_eq!(out, 32 * 6);
}

#[test]
fn test_ffi_cosine_batch_and_jaccard() {
    let mut a = [3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [6.0f32, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut cos = 0.0f32;
    let mut norm = 0.0f32;

    unsafe {
        assert_eq!(
            zc_cosine_similarity_f32(8, a.as_ptr(), b.as_ptr(), &mut cos),
            ZC_SUCCESS
        );
        assert!((cos - 1.0).abs() < 1e-5);

        assert_eq!(
            zc_l2_normalize_f32(8, a.as_mut_ptr(), &mut norm),
            ZC_SUCCESS
        );
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((a[0] - 0.6).abs() < 1e-5);
        assert!((a[1] - 0.8).abs() < 1e-5);

        // Batch dot product over 2 rows of dim 8
        let mut matrix = [0.0f32; 16];
        matrix[0] = 0.6;
        matrix[1] = 0.8;
        matrix[8] = -0.6;
        matrix[9] = -0.8;
        let mut scores = [0.0f32; 2];
        assert_eq!(
            zc_batch_dot_f32(2, 8, matrix.as_ptr(), a.as_ptr(), scores.as_mut_ptr()),
            ZC_SUCCESS
        );
        assert!((scores[0] - 1.0).abs() < 1e-5);
        assert!((scores[1] + 1.0).abs() < 1e-5);

        // Jaccard & Recall UTF-8 token overlap
        let s1 = "chính sách bảo hành c# đổi trả 30 ngày".as_bytes();
        let s2 = "chính sách bảo hành c# trọn đời".as_bytes();
        let mut jaccard = 0.0f32;
        let mut recall = 0.0f32;
        assert_eq!(
            zc_jaccard_token_overlap_utf8(
                s1.as_ptr(),
                s1.len(),
                s2.as_ptr(),
                s2.len(),
                &mut jaccard,
                &mut recall
            ),
            ZC_SUCCESS
        );
        assert!(jaccard > 0.4 && jaccard < 0.7);
        assert!((recall - (5.0 / 9.0)).abs() < 1e-4);
    }
}

#[test]
fn test_ffi_graphics_reexports() {
    assert_eq!(zg_version(), 100);
    let simd = zg_is_simd_supported();
    assert!(simd == 0 || simd == 1);

    let bgra = [29u8, 150, 77, 255, 255, 255, 255, 255];
    let mut gray = [0u8; 2];
    unsafe {
        assert_eq!(
            zg_bgra32_to_gray8(bgra.as_ptr(), 8, gray.as_mut_ptr(), 2, 2, 1),
            ZG_SUCCESS
        );
    }
    assert_eq!(gray[1], 255);
}



