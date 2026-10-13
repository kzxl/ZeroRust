use zero_tensor::*;

#[test]
fn test_version_and_simd() {
    let simd = zero_simd::is_avx2_fma_supported();
    println!("AVX2/FMA supported: {}", simd);
}

#[test]
fn test_gemm_correctness() {
    // A: 2x3, B: 3x2
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[7, 8], [9, 1], [2, 3]]
    // C = A @ B = [[31, 19], [85, 55]]
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0f32, 8.0, 9.0, 1.0, 2.0, 3.0];
    let mut c = [0.0f32; 4];

    unsafe {
        gemm_f32(
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
            false,
        );
    }

    assert_eq!(c[0], 31.0);
    assert_eq!(c[1], 19.0);
    assert_eq!(c[2], 85.0);
    assert_eq!(c[3], 55.0);
}

#[test]
fn test_gemm_larger_matrix() {
    let m = 32;
    let n = 32;
    let k = 32;
    let a = vec![1.0f32; m * k];
    let b = vec![2.0f32; k * n];
    let mut c = vec![0.0f32; m * n];

    unsafe {
        gemm_f32(
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
            true,
        );
    }

    // Each element should be k * 1 * 2 = 64
    for val in c.iter() {
        assert_eq!(*val, 64.0);
    }
}

#[test]
fn test_flash_attention() {
    let batch_count = 1;
    let sq = 4;
    let sk = 4;
    let d = 8;
    let dv = 8;
    let scale = 1.0 / (d as f32).sqrt();

    let q = vec![1.0f32; sq * d];
    let k = vec![1.0f32; sk * d];
    let v = vec![2.0f32; sk * dv];
    let mut out = vec![0.0f32; sq * dv];

    unsafe {
        flash_attention_f32(
            batch_count,
            sq,
            sk,
            d,
            dv,
            scale,
            false,
            1,
            1,
            1,
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            std::ptr::null(),
            out.as_mut_ptr(),
        );
    }

    // Since all Q and K are identical, attention weights are 1/4 uniform
    // Out = sum(0.25 * 2.0) = 2.0
    for val in out.iter() {
        assert!((val - 2.0).abs() < 1e-4, "Expected 2.0, got {}", val);
    }
}
