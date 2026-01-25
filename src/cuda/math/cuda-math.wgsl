// CUDA Math Library for WebGPU (WGSL)
// Provides CUDA-compatible math functions

// ============================================================================
// Constants
// ============================================================================

const CUDART_PI_F: f32 = 3.14159265358979323846f;
const CUDART_2PI_F: f32 = 6.28318530717958647692f;
const CUDART_PI_2_F: f32 = 1.57079632679489661923f;
const CUDART_PI_4_F: f32 = 0.78539816339744830962f;
const CUDART_1_PI_F: f32 = 0.31830988618379067154f;
const CUDART_2_PI_F: f32 = 0.63661977236758134308f;
const CUDART_SQRT_2_F: f32 = 1.41421356237309504880f;
const CUDART_SQRT_1_2_F: f32 = 0.70710678118654752440f;
const CUDART_E_F: f32 = 2.71828182845904523536f;
const CUDART_LOG2E_F: f32 = 1.44269504088896340736f;
const CUDART_LOG10E_F: f32 = 0.43429448190325182765f;
const CUDART_LN2_F: f32 = 0.69314718055994530942f;
const CUDART_LN10_F: f32 = 2.30258509299404568402f;

const CUDART_INF_F: f32 = 0x1.p127f * 2.0f;
const CUDART_NAN_F: f32 = 0.0f / 0.0f;

// ============================================================================
// Basic Math Functions
// ============================================================================

// Fast approximate sine (CUDA __sinf equivalent)
fn __sinf(x: f32) -> f32 {
    return sin(x);
}

// Fast approximate cosine (CUDA __cosf equivalent)
fn __cosf(x: f32) -> f32 {
    return cos(x);
}

// Fast approximate tangent
fn __tanf(x: f32) -> f32 {
    return tan(x);
}

// Sine and cosine combined (CUDA sincosf equivalent)
fn sincosf(x: f32, s: ptr<function, f32>, c: ptr<function, f32>) {
    *s = sin(x);
    *c = cos(x);
}

// Fast sine and cosine combined
fn __sincosf(x: f32, s: ptr<function, f32>, c: ptr<function, f32>) {
    *s = sin(x);
    *c = cos(x);
}

// Sine with pi multiplier (sinpi)
fn sinpif(x: f32) -> f32 {
    return sin(x * CUDART_PI_F);
}

// Cosine with pi multiplier (cospi)
fn cospif(x: f32) -> f32 {
    return cos(x * CUDART_PI_F);
}

// Sincospi
fn sincospif(x: f32, s: ptr<function, f32>, c: ptr<function, f32>) {
    let angle = x * CUDART_PI_F;
    *s = sin(angle);
    *c = cos(angle);
}

// ============================================================================
// Inverse Trigonometric Functions
// ============================================================================

fn asinf_custom(x: f32) -> f32 {
    return asin(x);
}

fn acosf_custom(x: f32) -> f32 {
    return acos(x);
}

fn atanf_custom(x: f32) -> f32 {
    return atan(x);
}

fn atan2f_custom(y: f32, x: f32) -> f32 {
    return atan2(y, x);
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

fn sinhf_custom(x: f32) -> f32 {
    return sinh(x);
}

fn coshf_custom(x: f32) -> f32 {
    return cosh(x);
}

fn tanhf_custom(x: f32) -> f32 {
    return tanh(x);
}

fn asinhf_custom(x: f32) -> f32 {
    return asinh(x);
}

fn acoshf_custom(x: f32) -> f32 {
    return acosh(x);
}

fn atanhf_custom(x: f32) -> f32 {
    return atanh(x);
}

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

// Fast exponential (CUDA __expf equivalent)
fn __expf(x: f32) -> f32 {
    return exp(x);
}

// Exponential base 2
fn exp2f_custom(x: f32) -> f32 {
    return exp2(x);
}

// Fast exp2
fn __exp2f(x: f32) -> f32 {
    return exp2(x);
}

// Exponential base 10
fn exp10f(x: f32) -> f32 {
    return pow(10.0f, x);
}

// Fast exp10
fn __exp10f(x: f32) -> f32 {
    return pow(10.0f, x);
}

// exp(x) - 1 for small x (more accurate)
fn expm1f(x: f32) -> f32 {
    if (abs(x) < 0.001f) {
        // Taylor series for small x: x + x^2/2 + x^3/6
        return x * (1.0f + x * (0.5f + x / 6.0f));
    }
    return exp(x) - 1.0f;
}

// Natural logarithm
fn __logf(x: f32) -> f32 {
    return log(x);
}

// Logarithm base 2
fn log2f_custom(x: f32) -> f32 {
    return log2(x);
}

fn __log2f(x: f32) -> f32 {
    return log2(x);
}

// Logarithm base 10
fn log10f_custom(x: f32) -> f32 {
    return log(x) * CUDART_LOG10E_F;
}

fn __log10f(x: f32) -> f32 {
    return log(x) * CUDART_LOG10E_F;
}

// log(1 + x) for small x (more accurate)
fn log1pf(x: f32) -> f32 {
    if (abs(x) < 0.001f) {
        // Taylor series: x - x^2/2 + x^3/3
        return x * (1.0f - x * (0.5f - x / 3.0f));
    }
    return log(1.0f + x);
}

// Log of gamma function
fn lgammaf(x: f32) -> f32 {
    // Stirling's approximation for large x
    if (x > 10.0f) {
        return (x - 0.5f) * log(x) - x + 0.5f * log(CUDART_2PI_F);
    }
    // Use reflection for negative x
    if (x < 0.5f) {
        return log(CUDART_PI_F / sin(CUDART_PI_F * x)) - lgammaf(1.0f - x);
    }
    // Lanczos approximation
    let g: f32 = 7.0f;
    let coeffs = array<f32, 9>(
        0.99999999999980993f,
        676.5203681218851f,
        -1259.1392167224028f,
        771.32342877765313f,
        -176.61502916214059f,
        12.507343278686905f,
        -0.13857109526572012f,
        9.9843695780195716e-6f,
        1.5056327351493116e-7f
    );

    var sum: f32 = coeffs[0];
    for (var i: i32 = 1; i < 9; i = i + 1) {
        sum = sum + coeffs[i] / (x + f32(i) - 1.0f);
    }

    let t: f32 = x + g - 0.5f;
    return 0.5f * log(CUDART_2PI_F) + (x - 0.5f) * log(t) - t + log(sum);
}

// Gamma function
fn tgammaf(x: f32) -> f32 {
    return exp(lgammaf(x));
}

// ============================================================================
// Power Functions
// ============================================================================

fn powf_custom(x: f32, y: f32) -> f32 {
    return pow(x, y);
}

// Fast power (CUDA __powf equivalent)
fn __powf(x: f32, y: f32) -> f32 {
    return pow(x, y);
}

fn sqrtf_custom(x: f32) -> f32 {
    return sqrt(x);
}

fn rsqrtf(x: f32) -> f32 {
    return inverseSqrt(x);
}

// Fast rsqrt (CUDA __rsqrtf equivalent)
fn __rsqrtf(x: f32) -> f32 {
    return inverseSqrt(x);
}

fn cbrtf(x: f32) -> f32 {
    // Cube root
    let sign_x = sign(x);
    return sign_x * pow(abs(x), 1.0f / 3.0f);
}

// Reciprocal cube root
fn rcbrtf(x: f32) -> f32 {
    let sign_x = sign(x);
    return sign_x * pow(abs(x), -1.0f / 3.0f);
}

// Hypotenuse (sqrt(x^2 + y^2))
fn hypotf_custom(x: f32, y: f32) -> f32 {
    return sqrt(x * x + y * y);
}

// Reciprocal hypotenuse
fn rhypotf(x: f32, y: f32) -> f32 {
    return inverseSqrt(x * x + y * y);
}

// Norm3d (sqrt(x^2 + y^2 + z^2))
fn norm3df(x: f32, y: f32, z: f32) -> f32 {
    return sqrt(x * x + y * y + z * z);
}

// Reciprocal norm3d
fn rnorm3df(x: f32, y: f32, z: f32) -> f32 {
    return inverseSqrt(x * x + y * y + z * z);
}

// Norm4d
fn norm4df(x: f32, y: f32, z: f32, w: f32) -> f32 {
    return sqrt(x * x + y * y + z * z + w * w);
}

// Reciprocal norm4d
fn rnorm4df(x: f32, y: f32, z: f32, w: f32) -> f32 {
    return inverseSqrt(x * x + y * y + z * z + w * w);
}

// ============================================================================
// Rounding Functions
// ============================================================================

fn floorf_custom(x: f32) -> f32 {
    return floor(x);
}

fn ceilf_custom(x: f32) -> f32 {
    return ceil(x);
}

fn truncf_custom(x: f32) -> f32 {
    return trunc(x);
}

fn roundf_custom(x: f32) -> f32 {
    return round(x);
}

// Round to nearest even (banker's rounding)
fn rintf(x: f32) -> f32 {
    let r = round(x);
    let d = x - r;
    if (abs(d) == 0.5f) {
        // Round to even
        let ri = i32(r);
        if ((ri & 1) != 0) {
            return r - sign(d);
        }
    }
    return r;
}

// Nearest integer
fn nearbyintf(x: f32) -> f32 {
    return rintf(x);
}

// ============================================================================
// Remainder and Modulo Functions
// ============================================================================

fn fmodf_custom(x: f32, y: f32) -> f32 {
    return x - y * trunc(x / y);
}

fn remainderf(x: f32, y: f32) -> f32 {
    return x - y * round(x / y);
}

// Modulo with floored division
fn modff(x: f32, iptr: ptr<function, f32>) -> f32 {
    *iptr = trunc(x);
    return x - *iptr;
}

// ============================================================================
// Floating Point Manipulation
// ============================================================================

fn copysignf_custom(x: f32, y: f32) -> f32 {
    return abs(x) * sign(y);
}

fn fdimf(x: f32, y: f32) -> f32 {
    return max(x - y, 0.0f);
}

fn fmaf_custom(x: f32, y: f32, z: f32) -> f32 {
    return fma(x, y, z);
}

fn frexpf_custom(x: f32, exp_out: ptr<function, i32>) -> f32 {
    if (x == 0.0f) {
        *exp_out = 0;
        return 0.0f;
    }
    let e = i32(floor(log2(abs(x)))) + 1;
    *exp_out = e;
    return x / exp2(f32(e));
}

fn ldexpf_custom(x: f32, exp_val: i32) -> f32 {
    return x * exp2(f32(exp_val));
}

fn scalbnf(x: f32, n: i32) -> f32 {
    return x * exp2(f32(n));
}

fn scalblnf(x: f32, n: i32) -> f32 {
    return scalbnf(x, n);
}

// ============================================================================
// Comparison Functions
// ============================================================================

fn fminf_custom(x: f32, y: f32) -> f32 {
    return min(x, y);
}

fn fmaxf_custom(x: f32, y: f32) -> f32 {
    return max(x, y);
}

fn isnan_custom(x: f32) -> bool {
    return x != x;
}

fn isinf_custom(x: f32) -> bool {
    return abs(x) == CUDART_INF_F;
}

fn isfinite_custom(x: f32) -> bool {
    return !isnan_custom(x) && !isinf_custom(x);
}

fn signbit_custom(x: f32) -> bool {
    return x < 0.0f || (x == 0.0f && 1.0f / x < 0.0f);
}

// ============================================================================
// Special Functions
// ============================================================================

// Error function
fn erff(x: f32) -> f32 {
    // Approximation using Horner's method
    let a1: f32 = 0.254829592f;
    let a2: f32 = -0.284496736f;
    let a3: f32 = 1.421413741f;
    let a4: f32 = -1.453152027f;
    let a5: f32 = 1.061405429f;
    let p: f32 = 0.3275911f;

    let sign_x = sign(x);
    let ax = abs(x);
    let t = 1.0f / (1.0f + p * ax);
    let y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);

    return sign_x * y;
}

// Complementary error function
fn erfcf(x: f32) -> f32 {
    return 1.0f - erff(x);
}

// Inverse error function
fn erfinvf(x: f32) -> f32 {
    // Approximation
    let a: f32 = 0.147f;
    let sign_x = sign(x);
    let ax = abs(x);

    let ln_term = log(1.0f - ax * ax);
    let term1 = 2.0f / (CUDART_PI_F * a) + ln_term / 2.0f;
    let term2 = ln_term / a;

    return sign_x * sqrt(sqrt(term1 * term1 - term2) - term1);
}

// Inverse complementary error function
fn erfcinvf(x: f32) -> f32 {
    return erfinvf(1.0f - x);
}

// Scaled complementary error function
fn erfcxf(x: f32) -> f32 {
    return exp(x * x) * erfcf(x);
}

// Normal CDF (cumulative distribution function)
fn normcdff(x: f32) -> f32 {
    return 0.5f * (1.0f + erff(x * CUDART_SQRT_1_2_F));
}

// Inverse normal CDF
fn normcdfinvf(p: f32) -> f32 {
    return CUDART_SQRT_2_F * erfinvf(2.0f * p - 1.0f);
}

// Bessel function of first kind, order 0
fn j0f(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 8.0f) {
        let y = x * x;
        let ans1 = 57568490574.0f + y * (-13362590354.0f + y * (651619640.7f
            + y * (-11214424.18f + y * (77392.33017f + y * (-184.9052456f)))));
        let ans2 = 57568490411.0f + y * (1029532985.0f + y * (9494680.718f
            + y * (59272.64853f + y * (267.8532712f + y * 1.0f))));
        return ans1 / ans2;
    }
    let z = 8.0f / ax;
    let y = z * z;
    let xx = ax - 0.785398164f;
    let ans1 = 1.0f + y * (-0.1098628627e-2f + y * (0.2734510407e-4f
        + y * (-0.2073370639e-5f + y * 0.2093887211e-6f)));
    let ans2 = -0.1562499995e-1f + y * (0.1430488765e-3f
        + y * (-0.6911147651e-5f + y * (0.7621095161e-6f - y * 0.934935152e-7f)));
    return sqrt(0.636619772f / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
}

// Bessel function of first kind, order 1
fn j1f(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 8.0f) {
        let y = x * x;
        let ans1 = x * (72362614232.0f + y * (-7895059235.0f + y * (242396853.1f
            + y * (-2972611.439f + y * (15704.48260f + y * (-30.16036606f))))));
        let ans2 = 144725228442.0f + y * (2300535178.0f + y * (18583304.74f
            + y * (99447.43394f + y * (376.9991397f + y * 1.0f))));
        return ans1 / ans2;
    }
    let z = 8.0f / ax;
    let y = z * z;
    let xx = ax - 2.356194491f;
    let ans1 = 1.0f + y * (0.183105e-2f + y * (-0.3516396496e-4f
        + y * (0.2457520174e-5f + y * (-0.240337019e-6f))));
    let ans2 = 0.04687499995f + y * (-0.2002690873e-3f
        + y * (0.8449199096e-5f + y * (-0.88228987e-6f + y * 0.105787412e-6f)));
    let ans = sqrt(0.636619772f / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
    return select(ans, -ans, x < 0.0f);
}

// Bessel function of second kind, order 0
fn y0f(x: f32) -> f32 {
    if (x < 8.0f) {
        let y = x * x;
        let ans1 = -2957821389.0f + y * (7062834065.0f + y * (-512359803.6f
            + y * (10879881.29f + y * (-86327.92757f + y * 228.4622733f))));
        let ans2 = 40076544269.0f + y * (745249964.8f + y * (7189466.438f
            + y * (47447.26470f + y * (226.1030244f + y * 1.0f))));
        return (ans1 / ans2) + 0.636619772f * j0f(x) * log(x);
    }
    let z = 8.0f / x;
    let y = z * z;
    let xx = x - 0.785398164f;
    let ans1 = 1.0f + y * (-0.1098628627e-2f + y * (0.2734510407e-4f
        + y * (-0.2073370639e-5f + y * 0.2093887211e-6f)));
    let ans2 = -0.1562499995e-1f + y * (0.1430488765e-3f
        + y * (-0.6911147651e-5f + y * (0.7621095161e-6f + y * (-0.934945152e-7f))));
    return sqrt(0.636619772f / x) * (sin(xx) * ans1 + z * cos(xx) * ans2);
}

// Bessel function of second kind, order 1
fn y1f(x: f32) -> f32 {
    if (x < 8.0f) {
        let y = x * x;
        let ans1 = x * (-0.4900604943e13f + y * (0.1275274390e13f
            + y * (-0.5153438139e11f + y * (0.7349264551e9f
            + y * (-0.4237922726e7f + y * 0.8511937935e4f)))));
        let ans2 = 0.2499580570e14f + y * (0.4244419664e12f
            + y * (0.3733650367e10f + y * (0.2245904002e8f
            + y * (0.1020426050e6f + y * (0.3549632885e3f + y)))));
        return (ans1 / ans2) + 0.636619772f * (j1f(x) * log(x) - 1.0f / x);
    }
    let z = 8.0f / x;
    let y = z * z;
    let xx = x - 2.356194491f;
    let ans1 = 1.0f + y * (0.183105e-2f + y * (-0.3516396496e-4f
        + y * (0.2457520174e-5f + y * (-0.240337019e-6f))));
    let ans2 = 0.04687499995f + y * (-0.2002690873e-3f
        + y * (0.8449199096e-5f + y * (-0.88228987e-6f + y * 0.105787412e-6f)));
    return sqrt(0.636619772f / x) * (sin(xx) * ans1 + z * cos(xx) * ans2);
}

// Cylinder Bessel function (regular modified)
fn cyl_bessel_i0f(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 3.75f) {
        let y = (x / 3.75f) * (x / 3.75f);
        return 1.0f + y * (3.5156229f + y * (3.0899424f + y * (1.2067492f
            + y * (0.2659732f + y * (0.360768e-1f + y * 0.45813e-2f)))));
    }
    let y = 3.75f / ax;
    return (exp(ax) / sqrt(ax)) * (0.39894228f + y * (0.1328592e-1f
        + y * (0.225319e-2f + y * (-0.157565e-2f + y * (0.916281e-2f
        + y * (-0.2057706e-1f + y * (0.2635537e-1f + y * (-0.1647633e-1f
        + y * 0.392377e-2f))))))));
}

// Cylinder Bessel function (regular modified, order 1)
fn cyl_bessel_i1f(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 3.75f) {
        let y = (x / 3.75f) * (x / 3.75f);
        let ans = ax * (0.5f + y * (0.87890594f + y * (0.51498869f
            + y * (0.15084934f + y * (0.2658733e-1f + y * (0.301532e-2f
            + y * 0.32411e-3f))))));
        return select(ans, -ans, x < 0.0f);
    }
    let y = 3.75f / ax;
    let ans = 0.2282967e-1f + y * (-0.2895312e-1f + y * (0.1787654e-1f
        + y * (-0.420059e-2f)));
    let ans2 = 0.39894228f + y * (-0.3988024e-1f + y * (-0.362018e-2f
        + y * (0.163801e-2f + y * (-0.1031555e-1f + y * ans))));
    let result = ans2 * exp(ax) / sqrt(ax);
    return select(result, -result, x < 0.0f);
}

// ============================================================================
// Integer Math Functions
// ============================================================================

fn __clz(x: u32) -> u32 {
    return countLeadingZeros(x);
}

fn __clzll(x: u64) -> u32 {
    let hi = u32(x >> 32u);
    let lo = u32(x & 0xFFFFFFFFu);
    if (hi != 0u) {
        return countLeadingZeros(hi);
    }
    return 32u + countLeadingZeros(lo);
}

fn __ffs(x: u32) -> u32 {
    if (x == 0u) { return 0u; }
    return countTrailingZeros(x) + 1u;
}

fn __ffsll(x: u64) -> u32 {
    if (x == 0u) { return 0u; }
    let lo = u32(x & 0xFFFFFFFFu);
    if (lo != 0u) {
        return countTrailingZeros(lo) + 1u;
    }
    return countTrailingZeros(u32(x >> 32u)) + 33u;
}

fn __popc(x: u32) -> u32 {
    return countOneBits(x);
}

fn __popcll(x: u64) -> u32 {
    return countOneBits(u32(x & 0xFFFFFFFFu)) + countOneBits(u32(x >> 32u));
}

fn __brev(x: u32) -> u32 {
    return reverseBits(x);
}

fn __brevll(x: u64) -> u64 {
    let hi = reverseBits(u32(x & 0xFFFFFFFFu));
    let lo = reverseBits(u32(x >> 32u));
    return (u64(hi) << 32u) | u64(lo);
}

// Byte swap
fn __byte_perm(x: u32, y: u32, s: u32) -> u32 {
    let bytes = array<u32, 8>(
        (x >> 0u) & 0xFFu,
        (x >> 8u) & 0xFFu,
        (x >> 16u) & 0xFFu,
        (x >> 24u) & 0xFFu,
        (y >> 0u) & 0xFFu,
        (y >> 8u) & 0xFFu,
        (y >> 16u) & 0xFFu,
        (y >> 24u) & 0xFFu
    );

    let b0 = bytes[(s >> 0u) & 0x7u];
    let b1 = bytes[(s >> 4u) & 0x7u];
    let b2 = bytes[(s >> 8u) & 0x7u];
    let b3 = bytes[(s >> 12u) & 0x7u];

    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

// ============================================================================
// Saturating Arithmetic
// ============================================================================

fn __saturatef(x: f32) -> f32 {
    return saturate(x);
}

fn __sad(x: i32, y: i32, z: i32) -> i32 {
    return abs(x - y) + z;
}

fn __usad(x: u32, y: u32, z: u32) -> u32 {
    return u32(abs(i32(x) - i32(y))) + z;
}

// ============================================================================
// Fused Multiply-Add Variants
// ============================================================================

fn __fmaf_rn(x: f32, y: f32, z: f32) -> f32 {
    return fma(x, y, z);
}

fn __fmaf_rz(x: f32, y: f32, z: f32) -> f32 {
    return fma(x, y, z);
}

fn __fmaf_ru(x: f32, y: f32, z: f32) -> f32 {
    return fma(x, y, z);
}

fn __fmaf_rd(x: f32, y: f32, z: f32) -> f32 {
    return fma(x, y, z);
}

// ============================================================================
// Division and Reciprocal
// ============================================================================

fn __fdividef(x: f32, y: f32) -> f32 {
    return x / y;
}

fn __frcp_rn(x: f32) -> f32 {
    return 1.0f / x;
}

fn __fsqrt_rn(x: f32) -> f32 {
    return sqrt(x);
}

// ============================================================================
// Lerp and smoothstep
// ============================================================================

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    return fma(t, b - a, a);
}

fn smoothstepf(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

fn smootherstepf(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}
