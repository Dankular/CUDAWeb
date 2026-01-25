/**
 * CUDA SDK for WebGPU
 *
 * Complete CUDA-compatible SDK for running GPU compute operations via WebGPU
 *
 * @module cuda
 */

// =============================================================================
// Runtime API
// =============================================================================

export * from './runtime';
export {
  // Device Management
  cudaInit,
  cudaGetDeviceCount,
  cudaGetDevice,
  cudaSetDevice,
  cudaGetDeviceProperties,
  cudaDeviceSynchronize,
  cudaDeviceReset,
  cudaGetLastError,
  cudaPeekAtLastError,
  cudaGetErrorString,
  cudaGetErrorName,
  CudaError,
  CudaMemcpyKind,
  CudaComputeMode,
  CudaDeviceAttr,

  // Memory Management
  cudaMalloc,
  cudaMallocPitch,
  cudaMalloc3D,
  cudaMallocHost,
  cudaMallocManaged,
  cudaFree,
  cudaFreeHost,
  cudaMemcpy,
  cudaMemcpyAsync,
  cudaMemset,
  cudaMemset2D,
  cudaMemset3D,
  cudaMemcpy2D,
  cudaMemcpy3D,
  cudaMemGetInfo,
  cudaPointerGetAttributes,

  // Stream and Event Management
  cudaStreamCreate,
  cudaStreamCreateWithFlags,
  cudaStreamCreateWithPriority,
  cudaStreamDestroy,
  cudaStreamSynchronize,
  cudaStreamQuery,
  cudaStreamWaitEvent,
  cudaEventCreate,
  cudaEventCreateWithFlags,
  cudaEventDestroy,
  cudaEventRecord,
  cudaEventSynchronize,
  cudaEventQuery,
  cudaEventElapsedTime,
  CudaStreamFlags,
  CudaEventFlags,
  CudaStreamCaptureStatus,

  // Kernel Launch
  dim3,
  cudaCreateKernel,
  cudaLaunchKernel,
  cudaLaunch,
  cudaGetKernel,
  cudaOccupancyMaxActiveBlocksPerMultiprocessor,
  cudaOccupancyMaxPotentialBlockSize,
  cudaLaunchCooperativeKernel,
  cudaFuncGetAttributes,
  cudaDestroyKernel,
} from './runtime';

// =============================================================================
// Math Library
// =============================================================================

export {
  CUDA_MATH_WGSL,
  CUDA_MATH_INFO,
  withCudaMath,
  // Constants
  CUDART_PI_F,
  CUDART_2PI_F,
  CUDART_PI_2_F,
  CUDART_PI_4_F,
  CUDART_1_PI_F,
  CUDART_2_PI_F,
  CUDART_SQRT_2_F,
  CUDART_SQRT_1_2_F,
  CUDART_E_F,
  CUDART_LOG2E_F,
  CUDART_LOG10E_F,
  CUDART_LN2_F,
  CUDART_LN10_F,
  // CPU-side math functions
  sinf, cosf, tanf, sincosf, sinpif, cospif,
  asinf, acosf, atanf, atan2f,
  sinhf, coshf, tanhf, asinhf, acoshf, atanhf,
  expf, exp2f, exp10f, expm1f,
  logf, log2f, log10f, log1pf,
  powf, sqrtf, rsqrtf, cbrtf, rcbrtf, hypotf, rhypotf,
  norm3df, rnorm3df, norm4df, rnorm4df,
  floorf, ceilf, truncf, roundf, rintf,
  fmodf, remainderf, fminf, fmaxf, fdimf,
  copysignf, fmaf, frexpf, ldexpf, scalbnf,
  erff, erfcf, erfinvf, erfcinvf, normcdff, normcdfinvf,
  lgammaf, tgammaf, j0f, j1f,
  clz, popc, brev, ffs,
  saturatef, sad, lerpf, smoothstepf, smootherstepf,
} from './math';

// =============================================================================
// Vector Types
// =============================================================================

export {
  CUDA_VECTOR_WGSL,
  withCudaVector,
  // Float vectors
  Float2, Float3, Float4,
  // Integer vectors
  Int2, Int3, Int4,
  // Unsigned integer vectors
  Uint2, Uint3, Uint4,
  // Matrices
  Mat3, Mat4,
  // Quaternion
  Quat,
  // Factory functions
  make_float2, make_float3, make_float4,
  make_int2, make_int3, make_int4,
  make_uint2, make_uint3, make_uint4,
} from './vector';

// =============================================================================
// Texture and Surface Operations
// =============================================================================

export {
  CUDA_TEXTURE_WGSL,
  withCudaTexture,
  cudaMallocArray,
  cudaMalloc3DArray,
  cudaFreeArray,
  cudaCreateTextureObject,
  cudaDestroyTextureObject,
  cudaCreateSurfaceObject,
  cudaDestroySurfaceObject,
  cudaMemcpyToArray,
  cudaMemcpy2DToArray,
  cudaCreateChannelDesc,
  getDefaultTextureDesc,
  CudaTextureAddressMode,
  CudaTextureFilterMode,
  CudaTextureReadMode,
  CudaChannelFormatKind,
} from './texture';

// =============================================================================
// Warp-Level Primitives
// =============================================================================

export {
  CUDA_WARP_WGSL,
  CUDA_WARP_INFO,
  withCudaWarp,
  WARP_SIZE,
} from './warp';

// =============================================================================
// Cooperative Groups
// =============================================================================

export {
  CUDA_COOPERATIVE_GROUPS_WGSL,
  COOPERATIVE_GROUPS_INFO,
  withCooperativeGroups,
} from './cooperative_groups';

// =============================================================================
// Atomic Operations
// =============================================================================

export {
  CUDA_ATOMIC_WGSL,
  ATOMIC_OPERATIONS_INFO,
  withCudaAtomic,
  AtomicBuffer,
} from './atomic';

// =============================================================================
// cuBLAS (Linear Algebra)
// =============================================================================

export {
  cublasCreate,
  cublasDestroy,
  cublasSetStream,
  cublasSaxpy,
  cublasSscal,
  cublasScopy,
  cublasSgemv,
  cublasSgemm,
  CublasStatus,
  CublasOperation,
  CublasFillMode,
  CublasDiagType,
  CublasSideMode,
} from './blas';

// =============================================================================
// cuFFT (Fast Fourier Transform)
// =============================================================================

export {
  cufftPlan1d,
  cufftPlan2d,
  cufftPlan3d,
  cufftDestroy,
  cufftExecC2C,
  cufftExecR2C,
  cufftExecC2R,
  CufftResult,
  CufftType,
  CufftDirection,
} from './fft';

// =============================================================================
// cuRAND (Random Number Generation)
// =============================================================================

export {
  curandCreateGenerator,
  curandDestroyGenerator,
  curandSetPseudoRandomGeneratorSeed,
  curandSetGeneratorOffset,
  curandSetStream,
  curandGenerateUniform,
  curandGenerateUniformDouble,
  curandGenerateNormal,
  curandGenerateLogNormal,
  curandGeneratePoisson,
  curandGenerate,
  CURAND_WGSL,
  withCurand,
  CurandStatus,
  CurandRngType,
  CurandOrdering,
} from './rand';

// =============================================================================
// Transpiler Utilities
// =============================================================================

export {
  TYPE_MAPPINGS,
  BUILTIN_MAPPINGS,
  FUNCTION_MAPPINGS,
  MEMORY_SPACE_MAPPINGS,
  PATTERN_EXAMPLES,
  CONCEPT_MAPPINGS,
  LIMITATIONS,
  transpileCudaToWgsl,
  generateKernelTemplate,
} from './transpiler';

// =============================================================================
// Convenience: Combined WGSL Libraries
// =============================================================================

import { CUDA_MATH_WGSL } from './math';
import { CUDA_VECTOR_WGSL } from './vector';
import { CUDA_WARP_WGSL } from './warp';
import { CUDA_COOPERATIVE_GROUPS_WGSL } from './cooperative_groups';
import { CUDA_ATOMIC_WGSL } from './atomic';
import { CUDA_TEXTURE_WGSL } from './texture';
import { CURAND_WGSL } from './rand';

/**
 * Complete CUDA-compatible WGSL library
 * Include this at the top of your compute shaders for full functionality
 */
export const CUDA_COMPLETE_WGSL = `
// =============================================================================
// CUDA SDK for WebGPU - Complete WGSL Library
// =============================================================================

${CUDA_MATH_WGSL}

${CUDA_VECTOR_WGSL}

${CUDA_ATOMIC_WGSL}

${CUDA_WARP_WGSL}

${CUDA_COOPERATIVE_GROUPS_WGSL}

${CURAND_WGSL}

${CUDA_TEXTURE_WGSL}
`;

/**
 * Helper to prepend the complete CUDA library to shader code
 */
export function withCudaComplete(shaderCode: string): string {
  return CUDA_COMPLETE_WGSL + '\n\n' + shaderCode;
}

// =============================================================================
// SDK Information
// =============================================================================

export const CUDA_SDK_INFO = {
  name: 'CUDA SDK for WebGPU',
  version: '1.0.0',
  description: 'CUDA-compatible GPU compute SDK using WebGPU',
  modules: {
    runtime: 'Device, memory, stream, event, and kernel management',
    math: 'Mathematical functions (trigonometric, exponential, special)',
    vector: 'Vector and matrix types with operations',
    texture: 'Texture and surface binding and sampling',
    warp: 'Warp-level primitives (shuffle, ballot, vote)',
    cooperativeGroups: 'Cooperative thread group operations',
    atomic: 'Atomic operations (native and emulated)',
    blas: 'cuBLAS-compatible linear algebra (SAXPY, SGEMM, etc.)',
    fft: 'cuFFT-compatible Fast Fourier Transform',
    rand: 'cuRAND-compatible random number generation (Philox RNG)',
    transpiler: 'CUDA to WGSL conversion utilities',
  },
  features: [
    'Full CUDA Runtime API compatibility layer',
    'Comprehensive math library with 100+ functions',
    'Vector types (float2/3/4, int2/3/4, matrices, quaternions)',
    'Texture and surface operations with samplers',
    'Warp-level primitives emulation',
    'Cooperative groups for collective operations',
    'Native and emulated atomic operations',
    'cuBLAS-compatible BLAS operations',
    'cuFFT-compatible FFT operations',
    'cuRAND-compatible RNG (Philox 4x32-10)',
    'CUDA to WGSL transpiler utilities',
  ],
  limitations: [
    'No native f64 (double) support',
    'Warp primitives emulated via shared memory',
    'No dynamic parallelism',
    'Maximum workgroup size ~256',
    'Float atomics use CAS loops',
    'No grid-wide synchronization',
  ],
};
