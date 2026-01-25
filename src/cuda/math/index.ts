/**
 * CUDA Math Library for WebGPU
 *
 * Provides CUDA-compatible math functions as both:
 * 1. WGSL shader code that can be included in compute shaders
 * 2. TypeScript implementations for CPU-side computations
 */

import cudaMathWGSL from './cuda-math.wgsl?raw';

// Export the WGSL source code for inclusion in shaders
export const CUDA_MATH_WGSL = cudaMathWGSL;

// Math constants matching CUDA
export const CUDART_PI_F = 3.14159265358979323846;
export const CUDART_2PI_F = 6.28318530717958647692;
export const CUDART_PI_2_F = 1.57079632679489661923;
export const CUDART_PI_4_F = 0.78539816339744830962;
export const CUDART_1_PI_F = 0.31830988618379067154;
export const CUDART_2_PI_F = 0.63661977236758134308;
export const CUDART_SQRT_2_F = 1.41421356237309504880;
export const CUDART_SQRT_1_2_F = 0.70710678118654752440;
export const CUDART_E_F = 2.71828182845904523536;
export const CUDART_LOG2E_F = 1.44269504088896340736;
export const CUDART_LOG10E_F = 0.43429448190325182765;
export const CUDART_LN2_F = 0.69314718055994530942;
export const CUDART_LN10_F = 2.30258509299404568402;

// ============================================================================
// CPU-side Math Functions (TypeScript implementations)
// ============================================================================

// Trigonometric functions
export function sinf(x: number): number {
  return Math.sin(x);
}

export function cosf(x: number): number {
  return Math.cos(x);
}

export function tanf(x: number): number {
  return Math.tan(x);
}

export function sincosf(x: number): { sin: number; cos: number } {
  return { sin: Math.sin(x), cos: Math.cos(x) };
}

export function sinpif(x: number): number {
  return Math.sin(x * Math.PI);
}

export function cospif(x: number): number {
  return Math.cos(x * Math.PI);
}

// Inverse trigonometric functions
export function asinf(x: number): number {
  return Math.asin(x);
}

export function acosf(x: number): number {
  return Math.acos(x);
}

export function atanf(x: number): number {
  return Math.atan(x);
}

export function atan2f(y: number, x: number): number {
  return Math.atan2(y, x);
}

// Hyperbolic functions
export function sinhf(x: number): number {
  return Math.sinh(x);
}

export function coshf(x: number): number {
  return Math.cosh(x);
}

export function tanhf(x: number): number {
  return Math.tanh(x);
}

export function asinhf(x: number): number {
  return Math.asinh(x);
}

export function acoshf(x: number): number {
  return Math.acosh(x);
}

export function atanhf(x: number): number {
  return Math.atanh(x);
}

// Exponential and logarithmic functions
export function expf(x: number): number {
  return Math.exp(x);
}

export function exp2f(x: number): number {
  return Math.pow(2, x);
}

export function exp10f(x: number): number {
  return Math.pow(10, x);
}

export function expm1f(x: number): number {
  return Math.expm1(x);
}

export function logf(x: number): number {
  return Math.log(x);
}

export function log2f(x: number): number {
  return Math.log2(x);
}

export function log10f(x: number): number {
  return Math.log10(x);
}

export function log1pf(x: number): number {
  return Math.log1p(x);
}

// Power functions
export function powf(x: number, y: number): number {
  return Math.pow(x, y);
}

export function sqrtf(x: number): number {
  return Math.sqrt(x);
}

export function rsqrtf(x: number): number {
  return 1 / Math.sqrt(x);
}

export function cbrtf(x: number): number {
  return Math.cbrt(x);
}

export function rcbrtf(x: number): number {
  return 1 / Math.cbrt(x);
}

export function hypotf(x: number, y: number): number {
  return Math.hypot(x, y);
}

export function rhypotf(x: number, y: number): number {
  return 1 / Math.hypot(x, y);
}

export function norm3df(x: number, y: number, z: number): number {
  return Math.sqrt(x * x + y * y + z * z);
}

export function rnorm3df(x: number, y: number, z: number): number {
  return 1 / Math.sqrt(x * x + y * y + z * z);
}

export function norm4df(x: number, y: number, z: number, w: number): number {
  return Math.sqrt(x * x + y * y + z * z + w * w);
}

export function rnorm4df(x: number, y: number, z: number, w: number): number {
  return 1 / Math.sqrt(x * x + y * y + z * z + w * w);
}

// Rounding functions
export function floorf(x: number): number {
  return Math.floor(x);
}

export function ceilf(x: number): number {
  return Math.ceil(x);
}

export function truncf(x: number): number {
  return Math.trunc(x);
}

export function roundf(x: number): number {
  return Math.round(x);
}

export function rintf(x: number): number {
  // Round to nearest even
  const r = Math.round(x);
  const d = x - r;
  if (Math.abs(d) === 0.5) {
    if (r % 2 !== 0) {
      return r - Math.sign(d);
    }
  }
  return r;
}

// Remainder and modulo
export function fmodf(x: number, y: number): number {
  return x % y;
}

export function remainderf(x: number, y: number): number {
  return x - y * Math.round(x / y);
}

// Comparison functions
export function fminf(x: number, y: number): number {
  return Math.min(x, y);
}

export function fmaxf(x: number, y: number): number {
  return Math.max(x, y);
}

export function fdimf(x: number, y: number): number {
  return Math.max(x - y, 0);
}

// Floating point manipulation
export function copysignf(x: number, y: number): number {
  return Math.sign(y) * Math.abs(x);
}

export function fmaf(x: number, y: number, z: number): number {
  return x * y + z;
}

export function frexpf(x: number): { mantissa: number; exponent: number } {
  if (x === 0) {
    return { mantissa: 0, exponent: 0 };
  }
  const exponent = Math.floor(Math.log2(Math.abs(x))) + 1;
  const mantissa = x / Math.pow(2, exponent);
  return { mantissa, exponent };
}

export function ldexpf(x: number, exp: number): number {
  return x * Math.pow(2, exp);
}

export function scalbnf(x: number, n: number): number {
  return x * Math.pow(2, n);
}

// Special functions
export function erff(x: number): number {
  // Approximation using Horner's method
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const signX = Math.sign(x);
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y =
    1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax));

  return signX * y;
}

export function erfcf(x: number): number {
  return 1 - erff(x);
}

export function erfinvf(x: number): number {
  const a = 0.147;
  const signX = Math.sign(x);
  const ax = Math.abs(x);

  const lnTerm = Math.log(1 - ax * ax);
  const term1 = 2 / (Math.PI * a) + lnTerm / 2;
  const term2 = lnTerm / a;

  return signX * Math.sqrt(Math.sqrt(term1 * term1 - term2) - term1);
}

export function erfcinvf(x: number): number {
  return erfinvf(1 - x);
}

export function normcdff(x: number): number {
  return 0.5 * (1 + erff(x * CUDART_SQRT_1_2_F));
}

export function normcdfinvf(p: number): number {
  return CUDART_SQRT_2_F * erfinvf(2 * p - 1);
}

// Gamma function
export function lgammaf(x: number): number {
  // Stirling's approximation for large x
  if (x > 10) {
    return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI);
  }
  // Use reflection for negative x
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgammaf(1 - x);
  }
  // Lanczos approximation
  const g = 7;
  const coeffs = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  let sum = coeffs[0];
  for (let i = 1; i < 9; i++) {
    sum += coeffs[i] / (x + i - 1);
  }

  const t = x + g - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x - 0.5) * Math.log(t) - t + Math.log(sum);
}

export function tgammaf(x: number): number {
  return Math.exp(lgammaf(x));
}

// Bessel functions (simplified implementations)
export function j0f(x: number): number {
  const ax = Math.abs(x);
  if (ax < 8) {
    const y = x * x;
    const ans1 =
      57568490574 +
      y *
        (-13362590354 +
          y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))));
    const ans2 =
      57568490411 +
      y * (1029532985 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
    return ans1 / ans2;
  }
  const z = 8 / ax;
  const y = z * z;
  const xx = ax - 0.785398164;
  const ans1 =
    1 +
    y * (-0.001098628627 + y * (0.00002734510407 + y * (-0.000002073370639 + y * 0.0000002093887211)));
  const ans2 =
    -0.01562499995 +
    y * (0.0001430488765 + y * (-0.000006911147651 + y * (0.0000007621095161 - y * 0.00000009349351520)));
  return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2);
}

export function j1f(x: number): number {
  const ax = Math.abs(x);
  if (ax < 8) {
    const y = x * x;
    const ans1 =
      x *
      (72362614232 +
        y *
          (-7895059235 +
            y * (242396853.1 + y * (-2972611.439 + y * (15704.4826 + y * -30.16036606)))));
    const ans2 =
      144725228442 +
      y * (2300535178 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
    return ans1 / ans2;
  }
  const z = 8 / ax;
  const y = z * z;
  const xx = ax - 2.356194491;
  const ans1 =
    1 + y * (0.00183105 + y * (-0.00003516396496 + y * (0.000002457520174 + y * -0.00000024033701)));
  const ans2 =
    0.04687499995 +
    y * (-0.0002002690873 + y * (0.000008449199096 + y * (-0.00000088228987 + y * 0.000000105787412)));
  const ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * ans1 - z * Math.sin(xx) * ans2);
  return x < 0 ? -ans : ans;
}

// Integer math functions
export function clz(x: number): number {
  return Math.clz32(x);
}

export function popc(x: number): number {
  let count = 0;
  let n = x >>> 0;
  while (n) {
    count += n & 1;
    n >>>= 1;
  }
  return count;
}

export function brev(x: number): number {
  let result = 0;
  let n = x >>> 0;
  for (let i = 0; i < 32; i++) {
    result = (result << 1) | (n & 1);
    n >>>= 1;
  }
  return result >>> 0;
}

export function ffs(x: number): number {
  if (x === 0) return 0;
  let n = x >>> 0;
  let pos = 1;
  while ((n & 1) === 0) {
    n >>>= 1;
    pos++;
  }
  return pos;
}

// Saturating arithmetic
export function saturatef(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export function sad(x: number, y: number, z: number): number {
  return Math.abs(x - y) + z;
}

// Lerp and smoothstep
export function lerpf(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

export function smoothstepf(edge0: number, edge1: number, x: number): number {
  const t = saturatef((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

export function smootherstepf(edge0: number, edge1: number, x: number): number {
  const t = saturatef((x - edge0) / (edge1 - edge0));
  return t * t * t * (t * (t * 6 - 15) + 10);
}

// Helper to prepend math library to WGSL shader code
export function withCudaMath(shaderCode: string): string {
  return CUDA_MATH_WGSL + '\n\n' + shaderCode;
}

// Math library info
export const CUDA_MATH_INFO = {
  version: '1.0.0',
  functions: {
    trigonometric: ['sin', 'cos', 'tan', 'sincos', 'sinpi', 'cospi', 'sincospi'],
    inverseTrig: ['asin', 'acos', 'atan', 'atan2'],
    hyperbolic: ['sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'],
    exponential: ['exp', 'exp2', 'exp10', 'expm1'],
    logarithmic: ['log', 'log2', 'log10', 'log1p'],
    power: ['pow', 'sqrt', 'rsqrt', 'cbrt', 'rcbrt', 'hypot', 'rhypot', 'norm3d', 'rnorm3d', 'norm4d', 'rnorm4d'],
    rounding: ['floor', 'ceil', 'trunc', 'round', 'rint', 'nearbyint'],
    remainder: ['fmod', 'remainder', 'modf'],
    floatManip: ['copysign', 'fdim', 'fma', 'frexp', 'ldexp', 'scalbn'],
    comparison: ['fmin', 'fmax', 'isnan', 'isinf', 'isfinite', 'signbit'],
    special: ['erf', 'erfc', 'erfinv', 'erfcinv', 'erfcx', 'normcdf', 'normcdfinv', 'lgamma', 'tgamma', 'j0', 'j1', 'y0', 'y1', 'cyl_bessel_i0', 'cyl_bessel_i1'],
    integer: ['clz', 'clzll', 'ffs', 'ffsll', 'popc', 'popcll', 'brev', 'brevll', 'byte_perm'],
    saturating: ['saturate', 'sad', 'usad'],
    fma: ['fmaf_rn', 'fmaf_rz', 'fmaf_ru', 'fmaf_rd'],
    division: ['fdividef', 'frcp_rn', 'fsqrt_rn'],
    interpolation: ['lerp', 'smoothstep', 'smootherstep'],
  },
};
