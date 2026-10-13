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
