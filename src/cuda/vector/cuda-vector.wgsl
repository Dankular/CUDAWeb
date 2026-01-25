// CUDA Vector Types for WebGPU (WGSL)
// Provides CUDA-compatible vector operations

// ============================================================================
// Type Aliases (WGSL has built-in vec types)
// ============================================================================

// Note: WGSL already has vec2<f32>, vec3<f32>, vec4<f32>
// We provide helper functions for CUDA-style operations

// ============================================================================
// Vector Construction Functions
// ============================================================================

fn make_float2(x: f32, y: f32) -> vec2<f32> {
    return vec2<f32>(x, y);
}

fn make_float3(x: f32, y: f32, z: f32) -> vec3<f32> {
    return vec3<f32>(x, y, z);
}

fn make_float4(x: f32, y: f32, z: f32, w: f32) -> vec4<f32> {
    return vec4<f32>(x, y, z, w);
}

fn make_int2(x: i32, y: i32) -> vec2<i32> {
    return vec2<i32>(x, y);
}

fn make_int3(x: i32, y: i32, z: i32) -> vec3<i32> {
    return vec3<i32>(x, y, z);
}

fn make_int4(x: i32, y: i32, z: i32, w: i32) -> vec4<i32> {
    return vec4<i32>(x, y, z, w);
}

fn make_uint2(x: u32, y: u32) -> vec2<u32> {
    return vec2<u32>(x, y);
}

fn make_uint3(x: u32, y: u32, z: u32) -> vec3<u32> {
    return vec3<u32>(x, y, z);
}

fn make_uint4(x: u32, y: u32, z: u32, w: u32) -> vec4<u32> {
    return vec4<u32>(x, y, z, w);
}

// ============================================================================
// Float2 Operations
// ============================================================================

fn float2_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

fn float2_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

fn float2_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a * b;
}

fn float2_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a / b;
}

fn float2_scale(a: vec2<f32>, s: f32) -> vec2<f32> {
    return a * s;
}

fn float2_dot(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return dot(a, b);
}

fn float2_length(a: vec2<f32>) -> f32 {
    return length(a);
}

fn float2_normalize(a: vec2<f32>) -> vec2<f32> {
    return normalize(a);
}

fn float2_lerp(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> {
    return mix(a, b, t);
}

fn float2_min(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return min(a, b);
}

fn float2_max(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return max(a, b);
}

fn float2_clamp(a: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> vec2<f32> {
    return clamp(a, lo, hi);
}

fn float2_abs(a: vec2<f32>) -> vec2<f32> {
    return abs(a);
}

fn float2_floor(a: vec2<f32>) -> vec2<f32> {
    return floor(a);
}

fn float2_ceil(a: vec2<f32>) -> vec2<f32> {
    return ceil(a);
}

fn float2_frac(a: vec2<f32>) -> vec2<f32> {
    return fract(a);
}

fn float2_negate(a: vec2<f32>) -> vec2<f32> {
    return -a;
}

fn float2_reflect(i: vec2<f32>, n: vec2<f32>) -> vec2<f32> {
    return reflect(i, n);
}

// ============================================================================
// Float3 Operations
// ============================================================================

fn float3_add(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a + b;
}

fn float3_sub(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a - b;
}

fn float3_mul(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a * b;
}

fn float3_div(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return a / b;
}

fn float3_scale(a: vec3<f32>, s: f32) -> vec3<f32> {
    return a * s;
}

fn float3_dot(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return dot(a, b);
}

fn float3_cross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return cross(a, b);
}

fn float3_length(a: vec3<f32>) -> f32 {
    return length(a);
}

fn float3_normalize(a: vec3<f32>) -> vec3<f32> {
    return normalize(a);
}

fn float3_lerp(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return mix(a, b, t);
}

fn float3_min(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return min(a, b);
}

fn float3_max(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return max(a, b);
}

fn float3_clamp(a: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    return clamp(a, lo, hi);
}

fn float3_abs(a: vec3<f32>) -> vec3<f32> {
    return abs(a);
}

fn float3_floor(a: vec3<f32>) -> vec3<f32> {
    return floor(a);
}

fn float3_ceil(a: vec3<f32>) -> vec3<f32> {
    return ceil(a);
}

fn float3_frac(a: vec3<f32>) -> vec3<f32> {
    return fract(a);
}

fn float3_negate(a: vec3<f32>) -> vec3<f32> {
    return -a;
}

fn float3_reflect(i: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return reflect(i, n);
}

fn float3_refract(i: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    return refract(i, n, eta);
}

fn float3_faceforward(n: vec3<f32>, i: vec3<f32>, nref: vec3<f32>) -> vec3<f32> {
    return faceForward(n, i, nref);
}

// ============================================================================
// Float4 Operations
// ============================================================================

fn float4_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a + b;
}

fn float4_sub(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a - b;
}

fn float4_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a * b;
}

fn float4_div(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a / b;
}

fn float4_scale(a: vec4<f32>, s: f32) -> vec4<f32> {
    return a * s;
}

fn float4_dot(a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(a, b);
}

fn float4_length(a: vec4<f32>) -> f32 {
    return length(a);
}

fn float4_normalize(a: vec4<f32>) -> vec4<f32> {
    return normalize(a);
}

fn float4_lerp(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    return mix(a, b, t);
}

fn float4_min(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return min(a, b);
}

fn float4_max(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return max(a, b);
}

fn float4_clamp(a: vec4<f32>, lo: vec4<f32>, hi: vec4<f32>) -> vec4<f32> {
    return clamp(a, lo, hi);
}

fn float4_abs(a: vec4<f32>) -> vec4<f32> {
    return abs(a);
}

fn float4_floor(a: vec4<f32>) -> vec4<f32> {
    return floor(a);
}

fn float4_ceil(a: vec4<f32>) -> vec4<f32> {
    return ceil(a);
}

fn float4_frac(a: vec4<f32>) -> vec4<f32> {
    return fract(a);
}

fn float4_negate(a: vec4<f32>) -> vec4<f32> {
    return -a;
}

// ============================================================================
// Matrix Operations (4x4)
// ============================================================================

fn mat4_mul_vec4(m: mat4x4<f32>, v: vec4<f32>) -> vec4<f32> {
    return m * v;
}

fn mat4_mul_mat4(a: mat4x4<f32>, b: mat4x4<f32>) -> mat4x4<f32> {
    return a * b;
}

fn mat4_transpose(m: mat4x4<f32>) -> mat4x4<f32> {
    return transpose(m);
}

fn mat4_determinant(m: mat4x4<f32>) -> f32 {
    return determinant(m);
}

// Matrix inversion (using cofactor expansion)
fn mat4_inverse(m: mat4x4<f32>) -> mat4x4<f32> {
    let n11 = m[0][0]; let n12 = m[1][0]; let n13 = m[2][0]; let n14 = m[3][0];
    let n21 = m[0][1]; let n22 = m[1][1]; let n23 = m[2][1]; let n24 = m[3][1];
    let n31 = m[0][2]; let n32 = m[1][2]; let n33 = m[2][2]; let n34 = m[3][2];
    let n41 = m[0][3]; let n42 = m[1][3]; let n43 = m[2][3]; let n44 = m[3][3];

    let t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    let t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    let t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    let t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    let det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    let idet = 1.0 / det;

    var r: mat4x4<f32>;
    r[0][0] = t11 * idet;
    r[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    r[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    r[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    r[1][0] = t12 * idet;
    r[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    r[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    r[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    r[2][0] = t13 * idet;
    r[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    r[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    r[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    r[3][0] = t14 * idet;
    r[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    r[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    r[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return r;
}

// ============================================================================
// Matrix Operations (3x3)
// ============================================================================

fn mat3_mul_vec3(m: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
    return m * v;
}

fn mat3_mul_mat3(a: mat3x3<f32>, b: mat3x3<f32>) -> mat3x3<f32> {
    return a * b;
}

fn mat3_transpose(m: mat3x3<f32>) -> mat3x3<f32> {
    return transpose(m);
}

fn mat3_determinant(m: mat3x3<f32>) -> f32 {
    return determinant(m);
}

fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let det = determinant(m);
    let idet = 1.0 / det;

    var r: mat3x3<f32>;
    r[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * idet;
    r[0][1] = (m[2][1] * m[0][2] - m[0][1] * m[2][2]) * idet;
    r[0][2] = (m[0][1] * m[1][2] - m[1][1] * m[0][2]) * idet;
    r[1][0] = (m[2][0] * m[1][2] - m[1][0] * m[2][2]) * idet;
    r[1][1] = (m[0][0] * m[2][2] - m[2][0] * m[0][2]) * idet;
    r[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * idet;
    r[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * idet;
    r[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * idet;
    r[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * idet;

    return r;
}

// ============================================================================
// Quaternion Operations (stored as vec4)
// ============================================================================

fn quat_identity() -> vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let half_angle = angle * 0.5;
    let s = sin(half_angle);
    return vec4<f32>(axis * s, cos(half_angle));
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
    );
}

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn quat_inverse(q: vec4<f32>) -> vec4<f32> {
    let len_sq = dot(q, q);
    return quat_conjugate(q) / len_sq;
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    return normalize(q);
}

fn quat_rotate_vec3(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec4<f32>(v, 0.0);
    let result = quat_mul(quat_mul(q, qv), quat_conjugate(q));
    return result.xyz;
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let xx = q.x * q.x;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let xw = q.x * q.w;
    let yy = q.y * q.y;
    let yz = q.y * q.z;
    let yw = q.y * q.w;
    let zz = q.z * q.z;
    let zw = q.z * q.w;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (yy + zz), 2.0 * (xy + zw), 2.0 * (xz - yw)),
        vec3<f32>(2.0 * (xy - zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + xw)),
        vec3<f32>(2.0 * (xz + yw), 2.0 * (yz - xw), 1.0 - 2.0 * (xx + yy))
    );
}

fn quat_slerp(a: vec4<f32>, b: vec4<f32>, t: f32) -> vec4<f32> {
    var bq = b;
    var cos_half_theta = dot(a, b);

    if (cos_half_theta < 0.0) {
        bq = -b;
        cos_half_theta = -cos_half_theta;
    }

    if (cos_half_theta > 0.9999) {
        return normalize(mix(a, bq, t));
    }

    let half_theta = acos(cos_half_theta);
    let sin_half_theta = sqrt(1.0 - cos_half_theta * cos_half_theta);

    let ratio_a = sin((1.0 - t) * half_theta) / sin_half_theta;
    let ratio_b = sin(t * half_theta) / sin_half_theta;

    return a * ratio_a + bq * ratio_b;
}

// ============================================================================
// Color Space Conversions (using float3/float4)
// ============================================================================

fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.x;
    let g = rgb.y;
    let b = rgb.z;

    let max_val = max(max(r, g), b);
    let min_val = min(min(r, g), b);
    let delta = max_val - min_val;

    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let v = max_val;

    if (delta > 0.00001) {
        s = delta / max_val;

        if (r >= max_val) {
            h = (g - b) / delta;
        } else if (g >= max_val) {
            h = 2.0 + (b - r) / delta;
        } else {
            h = 4.0 + (r - g) / delta;
        }

        h = h / 6.0;
        if (h < 0.0) {
            h = h + 1.0;
        }
    }

    return vec3<f32>(h, s, v);
}

fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let h = hsv.x;
    let s = hsv.y;
    let v = hsv.z;

    if (s <= 0.0) {
        return vec3<f32>(v, v, v);
    }

    let hh = h * 6.0;
    let i = i32(floor(hh));
    let ff = hh - f32(i);
    let p = v * (1.0 - s);
    let q = v * (1.0 - (s * ff));
    let t = v * (1.0 - (s * (1.0 - ff)));

    switch (i) {
        case 0: { return vec3<f32>(v, t, p); }
        case 1: { return vec3<f32>(q, v, p); }
        case 2: { return vec3<f32>(p, v, t); }
        case 3: { return vec3<f32>(p, q, v); }
        case 4: { return vec3<f32>(t, p, v); }
        default: { return vec3<f32>(v, p, q); }
    }
}

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.0031308);
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, linear <= cutoff);
}

fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let low = srgb / 12.92;
    let high = pow((srgb + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, srgb <= cutoff);
}

// ============================================================================
// Integer Vector Operations
// ============================================================================

fn int2_add(a: vec2<i32>, b: vec2<i32>) -> vec2<i32> {
    return a + b;
}

fn int2_sub(a: vec2<i32>, b: vec2<i32>) -> vec2<i32> {
    return a - b;
}

fn int2_mul(a: vec2<i32>, b: vec2<i32>) -> vec2<i32> {
    return a * b;
}

fn int2_div(a: vec2<i32>, b: vec2<i32>) -> vec2<i32> {
    return a / b;
}

fn int3_add(a: vec3<i32>, b: vec3<i32>) -> vec3<i32> {
    return a + b;
}

fn int3_sub(a: vec3<i32>, b: vec3<i32>) -> vec3<i32> {
    return a - b;
}

fn int3_mul(a: vec3<i32>, b: vec3<i32>) -> vec3<i32> {
    return a * b;
}

fn int3_div(a: vec3<i32>, b: vec3<i32>) -> vec3<i32> {
    return a / b;
}

fn int4_add(a: vec4<i32>, b: vec4<i32>) -> vec4<i32> {
    return a + b;
}

fn int4_sub(a: vec4<i32>, b: vec4<i32>) -> vec4<i32> {
    return a - b;
}

fn int4_mul(a: vec4<i32>, b: vec4<i32>) -> vec4<i32> {
    return a * b;
}

fn int4_div(a: vec4<i32>, b: vec4<i32>) -> vec4<i32> {
    return a / b;
}

// ============================================================================
// Unsigned Integer Vector Operations
// ============================================================================

fn uint2_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return a + b;
}

fn uint2_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return a - b;
}

fn uint2_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return a * b;
}

fn uint3_add(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
    return a + b;
}

fn uint3_sub(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
    return a - b;
}

fn uint3_mul(a: vec3<u32>, b: vec3<u32>) -> vec3<u32> {
    return a * b;
}

fn uint4_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return a + b;
}

fn uint4_sub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return a - b;
}

fn uint4_mul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    return a * b;
}
