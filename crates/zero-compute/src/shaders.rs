//! Canonical WebGPU Shading Language (WGSL) Compute Shaders for ZeroCompute.
//!
//! These shaders target cross-platform GPU execution across Vulkan, DirectX 12,
//! Metal, and WebGPU runtimes without requiring proprietary CUDA toolchains.

/// Tiled 16x16 Matrix Multiplication (GEMM) WGSL Compute Shader using shared workgroup SRAM.
pub const WGSL_GEMM_F32: &str = r#"
struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
};

@group(0) @binding(0) var<uniform> params: GemmParams;
@group(0) @binding(1) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(2) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> matrix_c: array<f32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc: f32 = 0.0;
    let num_tiles = (params.k + 15u) / 16u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * 16u + local_col;
        let b_row = t * 16u + local_row;

        if (row < params.m && a_col < params.k) {
            tile_a[local_row * 16u + local_col] = matrix_a[row * params.k + a_col];
        } else {
            tile_a[local_row * 16u + local_col] = 0.0;
        }

        if (b_row < params.k && col < params.n) {
            tile_b[local_row * 16u + local_col] = matrix_b[b_row * params.n + col];
        } else {
            tile_b[local_row * 16u + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var k_idx: u32 = 0u; k_idx < 16u; k_idx = k_idx + 1u) {
            acc = fma(tile_a[local_row * 16u + k_idx], tile_b[k_idx * 16u + local_col], acc);
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        let c_idx = row * params.n + col;
        if (params.beta == 0.0) {
            matrix_c[c_idx] = params.alpha * acc;
        } else {
            matrix_c[c_idx] = params.alpha * acc + params.beta * matrix_c[c_idx];
        }
    }
}
"#;

/// Root Mean Square Normalization (RMSNorm) WGSL Compute Shader.
pub const WGSL_RMSNORM_F32: &str = r#"
struct RmsNormParams {
    n: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> params: RmsNormParams;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;

var<workgroup> shared_sq: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let row_offset = group_id.x * params.n;

    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < params.n; i = i + 256u) {
        let val = input_x[row_offset + i];
        local_sum = fma(val, val, local_sum);
    }
    shared_sq[tid] = local_sum;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            shared_sq[tid] = shared_sq[tid] + shared_sq[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let inv_rms = inverseSqrt((shared_sq[0] / f32(params.n)) + params.eps);

    for (var i: u32 = tid; i < params.n; i = i + 256u) {
        output_y[row_offset + i] = (input_x[row_offset + i] * inv_rms) * weight[i];
    }
}
"#;

/// Fused SwiGLU Activation WGSL Compute Shader: dst[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i].
pub const WGSL_SWIGLU_F32: &str = r#"
struct SwiGluParams {
    n: u32,
};

@group(0) @binding(0) var<uniform> params: SwiGluParams;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.n) {
        let g = gate[idx];
        let silu = g / (1.0 + exp(-g));
        dst[idx] = silu * up[idx];
    }
}
"#;

/// Rotary Position Embedding (RoPE) WGSL Compute Shader.
pub const WGSL_ROPE_F32: &str = r#"
struct RopeParams {
    num_heads: u32,
    seq_len: u32,
    head_dim: u32,
    pos_offset: u32,
    theta_base: f32,
};

@group(0) @binding(0) var<uniform> params: RopeParams;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.num_heads * params.seq_len * half_dim;
    let pair_idx = global_id.x;

    if (pair_idx >= total_pairs) {
        return;
    }

    let i = pair_idx % half_dim;
    let rem = pair_idx / half_dim;
    let s = rem % params.seq_len;
    let h = rem / params.seq_len;

    let pos = f32(params.pos_offset + s);
    let exponent = f32(2u * i) / f32(params.head_dim);
    let freq = 1.0 / pow(params.theta_base, exponent);
    let angle = pos * freq;
    let c = cos(angle);
    let sn = sin(angle);

    let base_idx = (h * params.seq_len + s) * params.head_dim + (2u * i);
    let v0 = x[base_idx];
    let v1 = x[base_idx + 1u];

    x[base_idx] = v0 * c - v1 * sn;
    x[base_idx + 1u] = v0 * sn + v1 * c;
}
"#;

/// FlashAttention Online-Softmax WGSL Compute Shader.
pub const WGSL_FLASH_ATTENTION_F32: &str = r#"
struct AttentionParams {
    sq: u32,
    sk: u32,
    d: u32,
    dv: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let qi = global_id.x;
    if (qi >= params.sq) {
        return;
    }

    let q_offset = qi * params.d;
    let out_offset = qi * params.dv;

    var m_prev: f32 = -3.402823e+38;
    var l_prev: f32 = 0.0;

    for (var vi: u32 = 0u; vi < params.dv; vi = vi + 1u) {
        out[out_offset + vi] = 0.0;
    }

    for (var kj: u32 = 0u; kj < params.sk; kj = kj + 1u) {
        let k_offset = kj * params.d;
        var dot: f32 = 0.0;
        for (var di: u32 = 0u; di < params.d; di = di + 1u) {
            dot = fma(q[q_offset + di], k[k_offset + di], dot);
        }
        let score = dot * params.scale;

        let m_curr = max(m_prev, score);
        let alpha = exp(m_prev - m_curr);
        let beta = exp(score - m_curr);
        l_prev = l_prev * alpha + beta;

        let v_offset = kj * params.dv;
        for (var vi: u32 = 0u; vi < params.dv; vi = vi + 1u) {
            out[out_offset + vi] = out[out_offset + vi] * alpha + beta * v[v_offset + vi];
        }
        m_prev = m_curr;
    }

    let inv_l = 1.0 / l_prev;
    for (var vi: u32 = 0u; vi < params.dv; vi = vi + 1u) {
        out[out_offset + vi] = out[out_offset + vi] * inv_l;
    }
}
"#;
