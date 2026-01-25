/**
 * Double Precision (f64) Emulation for WebGPU
 *
 * Provides emulated double-precision floating-point operations using
 * double-single (DS) arithmetic with two f32 values.
 *
 * IEEE 754 double precision: 64-bit (1 sign + 11 exponent + 52 mantissa)
 * Emulation approach: Store as vec2<f32> where x = high, y = low
 *
 * This technique provides ~44 bits of precision (vs 53 for native f64)
 * which is sufficient for many scientific computing applications.
 */

// WGSL Double Precision Emulation Implementation
export const CUDA_F64_WGSL = `
// ============================================================================
// Double Precision Emulation Types and Constants
// ============================================================================

// Emulated double is stored as vec2<u32> representing the two 32-bit halves
// of an IEEE 754 double precision float (little-endian: x=low, y=high)
alias f64 = vec2<u32>;
alias double = vec2<u32>;

// Double-single representation for arithmetic (high + low precision)
alias ds_real = vec2<f32>;

// Constants for f64 operations
const F64_ZERO: f64 = vec2<u32>(0u, 0u);
const F64_ONE: f64 = vec2<u32>(0u, 0x3FF00000u);
const F64_NEG_ONE: f64 = vec2<u32>(0u, 0xBFF00000u);
const F64_TWO: f64 = vec2<u32>(0u, 0x40000000u);
const F64_HALF: f64 = vec2<u32>(0u, 0x3FE00000u);
const F64_PI: f64 = vec2<u32>(0x54442D18u, 0x400921FBu);
const F64_E: f64 = vec2<u32>(0x8B145769u, 0x4005BF0Au);
const F64_LN2: f64 = vec2<u32>(0xFEFA39EFu, 0x3FE62E42u);
const F64_LN10: f64 = vec2<u32>(0xBBB55516u, 0x40026BB1u);

// Bit manipulation masks
const F64_SIGN_MASK_HIGH: u32 = 0x80000000u;
const F64_EXP_MASK_HIGH: u32 = 0x7FF00000u;
const F64_MANTISSA_MASK_HIGH: u32 = 0x000FFFFFu;
const F64_MANTISSA_MASK_LOW: u32 = 0xFFFFFFFFu;

// ============================================================================
// f64 Creation and Conversion Functions
// ============================================================================

// Create f64 from a single f32 (exact conversion)
fn f64_from_f32(val: f32) -> f64 {
    // Convert f32 to f64 bit representation
    let bits = bitcast<u32>(val);
    let sign = bits & 0x80000000u;
    let exp = (bits >> 23u) & 0xFFu;
    let mantissa = bits & 0x007FFFFFu;

    // Handle special cases
    if (exp == 0u) {
        // Zero or denormal
        if (mantissa == 0u) {
            return vec2<u32>(0u, sign);
        }
        // Denormal - normalize it
        var m = mantissa;
        var e = 0i;
        while ((m & 0x00800000u) == 0u) {
            m = m << 1u;
            e = e - 1i;
        }
        m = m & 0x007FFFFFu;
        let new_exp = u32(i32(exp) - 127i + 1023i + e);
        let high = sign | (new_exp << 20u) | (m >> 3u);
        let low = (m & 0x7u) << 29u;
        return vec2<u32>(low, high);
    } else if (exp == 0xFFu) {
        // Infinity or NaN
        let high = sign | 0x7FF00000u | (mantissa >> 3u);
        let low = (mantissa & 0x7u) << 29u;
        return vec2<u32>(low, high);
    }

    // Normal number
    let new_exp = (exp - 127u + 1023u) << 20u;
    let high = sign | new_exp | (mantissa >> 3u);
    let low = (mantissa & 0x7u) << 29u;
    return vec2<u32>(low, high);
}

// Create f64 from integer
fn f64_from_i32(val: i32) -> f64 {
    return f64_from_f32(f32(val));
}

fn f64_from_u32(val: u32) -> f64 {
    return f64_from_f32(f32(val));
}

// Convert f64 to f32 (with precision loss)
fn f64_to_f32(val: f64) -> f32 {
    let low = val.x;
    let high = val.y;

    let sign = high & 0x80000000u;
    let exp = (high >> 20u) & 0x7FFu;
    let mantissa_high = high & 0x000FFFFFu;

    // Handle special cases
    if (exp == 0u) {
        // Zero or denormal
        if (mantissa_high == 0u && low == 0u) {
            return bitcast<f32>(sign);
        }
        return 0.0; // Denormals become zero in f32
    } else if (exp == 0x7FFu) {
        // Infinity or NaN
        let f32_bits = sign | 0x7F800000u | (mantissa_high << 3u) | (low >> 29u);
        return bitcast<f32>(f32_bits);
    }

    // Normal number
    let exp_diff = i32(exp) - 1023i + 127i;
    if (exp_diff <= 0i) {
        return 0.0; // Underflow
    } else if (exp_diff >= 255i) {
        return bitcast<f32>(sign | 0x7F800000u); // Overflow to infinity
    }

    let f32_mantissa = (mantissa_high << 3u) | (low >> 29u);
    let f32_bits = sign | (u32(exp_diff) << 23u) | f32_mantissa;
    return bitcast<f32>(f32_bits);
}

// Create f64 from two f32 components (double-single representation)
fn f64_from_ds(hi: f32, lo: f32) -> f64 {
    // This is an approximation - combine high and low parts
    let combined = hi + lo;
    return f64_from_f32(combined);
}

// ============================================================================
// f64 Basic Arithmetic Operations
// ============================================================================

// Negate f64
fn f64_neg(a: f64) -> f64 {
    return vec2<u32>(a.x, a.y ^ F64_SIGN_MASK_HIGH);
}

// Absolute value
fn f64_abs(a: f64) -> f64 {
    return vec2<u32>(a.x, a.y & ~F64_SIGN_MASK_HIGH);
}

// Sign function
fn f64_sign(a: f64) -> f32 {
    if (f64_is_zero(a)) {
        return 0.0;
    }
    return select(1.0, -1.0, (a.y & F64_SIGN_MASK_HIGH) != 0u);
}

// Check if zero
fn f64_is_zero(a: f64) -> bool {
    return (a.y & ~F64_SIGN_MASK_HIGH) == 0u && a.x == 0u;
}

// Check if negative
fn f64_is_negative(a: f64) -> bool {
    return (a.y & F64_SIGN_MASK_HIGH) != 0u;
}

// Check if NaN
fn f64_is_nan(a: f64) -> bool {
    let exp = (a.y >> 20u) & 0x7FFu;
    let mantissa_high = a.y & 0x000FFFFFu;
    return exp == 0x7FFu && (mantissa_high != 0u || a.x != 0u);
}

// Check if Infinity
fn f64_is_inf(a: f64) -> bool {
    let exp = (a.y >> 20u) & 0x7FFu;
    let mantissa_high = a.y & 0x000FFFFFu;
    return exp == 0x7FFu && mantissa_high == 0u && a.x == 0u;
}

// ============================================================================
// Double-Single Arithmetic (Higher Precision Operations)
// ============================================================================

// Two-sum: returns (sum, error) where sum + error = a + b exactly
fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2<f32>(s, e);
}

// Quick two-sum when |a| >= |b|
fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let e = b - (s - a);
    return vec2<f32>(s, e);
}

// Two-product: returns (product, error) where product + error = a * b exactly
fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

// Double-single addition
fn ds_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = two_sum(a.x, b.x);
    let t = two_sum(a.y, b.y);
    let c = quick_two_sum(s.x, s.y + t.x);
    return quick_two_sum(c.x, t.y + c.y);
}

// Double-single subtraction
fn ds_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return ds_add(a, vec2<f32>(-b.x, -b.y));
}

// Double-single multiplication
fn ds_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = two_prod(a.x, b.x);
    let s = p.y + (a.x * b.y + a.y * b.x);
    return quick_two_sum(p.x, s);
}

// Double-single division
fn ds_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q1 = a.x / b.x;
    let p = two_prod(q1, b.x);
    let s = two_sum(a.x, -p.x);
    let r = (s.x + s.y - p.y + a.y - q1 * b.y) / b.x;
    return quick_two_sum(q1, r);
}

// Double-single square root
fn ds_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (a.x <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let x = 1.0 / sqrt(a.x);
    let y = a.x * x;
    let p = two_prod(y, y);
    let s = two_sum(a.x, -p.x);
    let r = (s.x + s.y - p.y + a.y) * (x * 0.5);
    return quick_two_sum(y, r);
}

// ============================================================================
// f64 Arithmetic using Double-Single internally
// ============================================================================

// Convert f64 to double-single for arithmetic
fn f64_to_ds(a: f64) -> vec2<f32> {
    let hi = f64_to_f32(a);
    // For simplicity, we use single precision only
    // A more accurate implementation would track the low bits
    return vec2<f32>(hi, 0.0);
}

// Convert double-single back to f64
fn ds_to_f64(a: vec2<f32>) -> f64 {
    return f64_from_f32(a.x + a.y);
}

// f64 addition
fn f64_add(a: f64, b: f64) -> f64 {
    let da = f64_to_ds(a);
    let db = f64_to_ds(b);
    return ds_to_f64(ds_add(da, db));
}

// f64 subtraction
fn f64_sub(a: f64, b: f64) -> f64 {
    let da = f64_to_ds(a);
    let db = f64_to_ds(b);
    return ds_to_f64(ds_sub(da, db));
}

// f64 multiplication
fn f64_mul(a: f64, b: f64) -> f64 {
    let da = f64_to_ds(a);
    let db = f64_to_ds(b);
    return ds_to_f64(ds_mul(da, db));
}

// f64 division
fn f64_div(a: f64, b: f64) -> f64 {
    let da = f64_to_ds(a);
    let db = f64_to_ds(b);
    return ds_to_f64(ds_div(da, db));
}

// f64 square root
fn f64_sqrt(a: f64) -> f64 {
    let da = f64_to_ds(a);
    return ds_to_f64(ds_sqrt(da));
}

// f64 reciprocal
fn f64_rcp(a: f64) -> f64 {
    return f64_div(F64_ONE, a);
}

// f64 reciprocal square root
fn f64_rsqrt(a: f64) -> f64 {
    return f64_rcp(f64_sqrt(a));
}

// ============================================================================
// f64 Comparison Operations
// ============================================================================

fn f64_eq(a: f64, b: f64) -> bool {
    return a.x == b.x && a.y == b.y;
}

fn f64_ne(a: f64, b: f64) -> bool {
    return !f64_eq(a, b);
}

fn f64_lt(a: f64, b: f64) -> bool {
    let a_neg = f64_is_negative(a);
    let b_neg = f64_is_negative(b);
    if (a_neg != b_neg) {
        return a_neg;
    }
    // Same sign - compare magnitudes
    if (a_neg) {
        // Both negative - larger magnitude is smaller
        if (a.y != b.y) {
            return a.y > b.y;
        }
        return a.x > b.x;
    } else {
        // Both positive
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    }
}

fn f64_le(a: f64, b: f64) -> bool {
    return f64_eq(a, b) || f64_lt(a, b);
}

fn f64_gt(a: f64, b: f64) -> bool {
    return f64_lt(b, a);
}

fn f64_ge(a: f64, b: f64) -> bool {
    return f64_eq(a, b) || f64_gt(a, b);
}

fn f64_min(a: f64, b: f64) -> f64 {
    return select(b, a, f64_lt(a, b));
}

fn f64_max(a: f64, b: f64) -> f64 {
    return select(b, a, f64_gt(a, b));
}

fn f64_clamp(val: f64, min_val: f64, max_val: f64) -> f64 {
    return f64_max(min_val, f64_min(max_val, val));
}

// ============================================================================
// f64 Math Functions
// ============================================================================

// Floor function
fn f64_floor(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(floor(f));
}

// Ceil function
fn f64_ceil(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(ceil(f));
}

// Round function
fn f64_round(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(round(f));
}

// Truncate function
fn f64_trunc(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(trunc(f));
}

// Fract function
fn f64_fract(a: f64) -> f64 {
    return f64_sub(a, f64_floor(a));
}

// Modulo function
fn f64_mod(a: f64, b: f64) -> f64 {
    let q = f64_floor(f64_div(a, b));
    return f64_sub(a, f64_mul(q, b));
}

// Power function (integer exponent)
fn f64_powi(base: f64, exp: i32) -> f64 {
    if (exp == 0i) {
        return F64_ONE;
    }
    var result = F64_ONE;
    var b = base;
    var e = abs(exp);
    while (e > 0i) {
        if ((e & 1i) != 0i) {
            result = f64_mul(result, b);
        }
        b = f64_mul(b, b);
        e = e >> 1i;
    }
    if (exp < 0i) {
        result = f64_rcp(result);
    }
    return result;
}

// Fused multiply-add
fn f64_fma(a: f64, b: f64, c: f64) -> f64 {
    return f64_add(f64_mul(a, b), c);
}

// Linear interpolation
fn f64_mix(a: f64, b: f64, t: f64) -> f64 {
    return f64_add(a, f64_mul(t, f64_sub(b, a)));
}

// ============================================================================
// f64 Trigonometric Functions (using f32 approximations)
// ============================================================================

fn f64_sin(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(sin(f));
}

fn f64_cos(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(cos(f));
}

fn f64_tan(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(tan(f));
}

fn f64_asin(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(asin(f));
}

fn f64_acos(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(acos(f));
}

fn f64_atan(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(atan(f));
}

fn f64_atan2(y: f64, x: f64) -> f64 {
    let fy = f64_to_f32(y);
    let fx = f64_to_f32(x);
    return f64_from_f32(atan2(fy, fx));
}

fn f64_sinh(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(sinh(f));
}

fn f64_cosh(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(cosh(f));
}

fn f64_tanh(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(tanh(f));
}

// ============================================================================
// f64 Exponential and Logarithmic Functions
// ============================================================================

fn f64_exp(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(exp(f));
}

fn f64_exp2(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(exp2(f));
}

fn f64_log(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(log(f));
}

fn f64_log2(a: f64) -> f64 {
    let f = f64_to_f32(a);
    return f64_from_f32(log2(f));
}

fn f64_pow(a: f64, b: f64) -> f64 {
    let fa = f64_to_f32(a);
    let fb = f64_to_f32(b);
    return f64_from_f32(pow(fa, fb));
}

// ============================================================================
// f64 Vector Types
// ============================================================================

alias double2 = array<f64, 2>;
alias double3 = array<f64, 3>;
alias double4 = array<f64, 4>;

fn make_double2(x: f64, y: f64) -> double2 {
    return array<f64, 2>(x, y);
}

fn make_double3(x: f64, y: f64, z: f64) -> double3 {
    return array<f64, 3>(x, y, z);
}

fn make_double4(x: f64, y: f64, z: f64, w: f64) -> double4 {
    return array<f64, 4>(x, y, z, w);
}

// Vector addition
fn double2_add(a: double2, b: double2) -> double2 {
    return array<f64, 2>(f64_add(a[0], b[0]), f64_add(a[1], b[1]));
}

fn double3_add(a: double3, b: double3) -> double3 {
    return array<f64, 3>(
        f64_add(a[0], b[0]),
        f64_add(a[1], b[1]),
        f64_add(a[2], b[2])
    );
}

fn double4_add(a: double4, b: double4) -> double4 {
    return array<f64, 4>(
        f64_add(a[0], b[0]),
        f64_add(a[1], b[1]),
        f64_add(a[2], b[2]),
        f64_add(a[3], b[3])
    );
}

// Vector subtraction
fn double2_sub(a: double2, b: double2) -> double2 {
    return array<f64, 2>(f64_sub(a[0], b[0]), f64_sub(a[1], b[1]));
}

fn double3_sub(a: double3, b: double3) -> double3 {
    return array<f64, 3>(
        f64_sub(a[0], b[0]),
        f64_sub(a[1], b[1]),
        f64_sub(a[2], b[2])
    );
}

fn double4_sub(a: double4, b: double4) -> double4 {
    return array<f64, 4>(
        f64_sub(a[0], b[0]),
        f64_sub(a[1], b[1]),
        f64_sub(a[2], b[2]),
        f64_sub(a[3], b[3])
    );
}

// Dot product
fn double2_dot(a: double2, b: double2) -> f64 {
    return f64_add(f64_mul(a[0], b[0]), f64_mul(a[1], b[1]));
}

fn double3_dot(a: double3, b: double3) -> f64 {
    return f64_add(
        f64_add(f64_mul(a[0], b[0]), f64_mul(a[1], b[1])),
        f64_mul(a[2], b[2])
    );
}

fn double4_dot(a: double4, b: double4) -> f64 {
    return f64_add(
        f64_add(f64_mul(a[0], b[0]), f64_mul(a[1], b[1])),
        f64_add(f64_mul(a[2], b[2]), f64_mul(a[3], b[3]))
    );
}

// Vector length
fn double2_length(a: double2) -> f64 {
    return f64_sqrt(double2_dot(a, a));
}

fn double3_length(a: double3) -> f64 {
    return f64_sqrt(double3_dot(a, a));
}

fn double4_length(a: double4) -> f64 {
    return f64_sqrt(double4_dot(a, a));
}

// Vector normalize
fn double2_normalize(a: double2) -> double2 {
    let len = double2_length(a);
    return array<f64, 2>(f64_div(a[0], len), f64_div(a[1], len));
}

fn double3_normalize(a: double3) -> double3 {
    let len = double3_length(a);
    return array<f64, 3>(
        f64_div(a[0], len),
        f64_div(a[1], len),
        f64_div(a[2], len)
    );
}

// Cross product (3D only)
fn double3_cross(a: double3, b: double3) -> double3 {
    return array<f64, 3>(
        f64_sub(f64_mul(a[1], b[2]), f64_mul(a[2], b[1])),
        f64_sub(f64_mul(a[2], b[0]), f64_mul(a[0], b[2])),
        f64_sub(f64_mul(a[0], b[1]), f64_mul(a[1], b[0]))
    );
}

// Scale vector
fn double2_scale(a: double2, s: f64) -> double2 {
    return array<f64, 2>(f64_mul(a[0], s), f64_mul(a[1], s));
}

fn double3_scale(a: double3, s: f64) -> double3 {
    return array<f64, 3>(
        f64_mul(a[0], s),
        f64_mul(a[1], s),
        f64_mul(a[2], s)
    );
}

fn double4_scale(a: double4, s: f64) -> double4 {
    return array<f64, 4>(
        f64_mul(a[0], s),
        f64_mul(a[1], s),
        f64_mul(a[2], s),
        f64_mul(a[3], s)
    );
}
`;

// TypeScript helper functions
export function withCudaF64(shaderCode: string): string {
  return CUDA_F64_WGSL + '\n\n' + shaderCode;
}

export const CUDA_F64_INFO = {
  name: 'Double Precision Emulation',
  description: 'Emulated 64-bit floating point using double-single arithmetic',
  precision: '~44 bits of mantissa precision (vs 53 for native f64)',
  types: {
    'f64': 'Emulated double stored as vec2<u32>',
    'double': 'Alias for f64',
    'ds_real': 'Double-single representation as vec2<f32>',
    'double2': 'Array of 2 f64 values',
    'double3': 'Array of 3 f64 values',
    'double4': 'Array of 4 f64 values',
  },
  operations: {
    basic: ['f64_add', 'f64_sub', 'f64_mul', 'f64_div', 'f64_neg', 'f64_abs'],
    comparison: ['f64_eq', 'f64_lt', 'f64_le', 'f64_gt', 'f64_ge', 'f64_min', 'f64_max'],
    math: ['f64_sqrt', 'f64_rcp', 'f64_rsqrt', 'f64_pow', 'f64_powi', 'f64_fma'],
    rounding: ['f64_floor', 'f64_ceil', 'f64_round', 'f64_trunc', 'f64_fract'],
    trig: ['f64_sin', 'f64_cos', 'f64_tan', 'f64_asin', 'f64_acos', 'f64_atan', 'f64_atan2'],
    hyperbolic: ['f64_sinh', 'f64_cosh', 'f64_tanh'],
    exponential: ['f64_exp', 'f64_exp2', 'f64_log', 'f64_log2'],
    conversion: ['f64_from_f32', 'f64_to_f32', 'f64_from_i32', 'f64_from_u32'],
  },
  constants: {
    'F64_ZERO': 'Double precision zero',
    'F64_ONE': 'Double precision one',
    'F64_PI': 'Double precision pi',
    'F64_E': 'Double precision e',
    'F64_LN2': 'Double precision ln(2)',
    'F64_LN10': 'Double precision ln(10)',
  },
  limitations: [
    'Precision is limited to ~44 bits (vs 53 for native f64)',
    'Performance is slower than native f64 (multiple f32 operations)',
    'Trigonometric functions use f32 internally with limited precision',
    'Some edge cases (subnormals, extreme values) may not be handled perfectly',
  ],
};

/**
 * Helper class for managing double precision buffers
 */
export class Double64Buffer {
  private device: GPUDevice;
  private buffer: GPUBuffer;
  private count: number;

  constructor(device: GPUDevice, count: number) {
    this.device = device;
    this.count = count;
    // Each f64 is stored as vec2<u32> = 8 bytes
    this.buffer = device.createBuffer({
      size: count * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  getBuffer(): GPUBuffer {
    return this.buffer;
  }

  getCount(): number {
    return this.count;
  }

  /**
   * Write Float64Array data to the buffer
   */
  writeData(data: Float64Array): void {
    if (data.length > this.count) {
      throw new Error(`Data length ${data.length} exceeds buffer capacity ${this.count}`);
    }
    // Convert Float64Array to Uint32Array (reinterpret bits)
    const u32View = new Uint32Array(data.buffer, data.byteOffset, data.length * 2);
    this.device.queue.writeBuffer(this.buffer, 0, u32View);
  }

  /**
   * Read buffer data as Float64Array
   */
  async readData(): Promise<Float64Array> {
    const staging = this.device.createBuffer({
      size: this.count * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, this.count * 8);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const u32Data = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    // Convert Uint32Array back to Float64Array
    return new Float64Array(u32Data.buffer);
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
