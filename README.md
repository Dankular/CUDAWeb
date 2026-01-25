# CUDAWeb - CUDA SDK for WebGPU

A comprehensive CUDA-compatible SDK for WebGPU, enabling GPU compute applications to run in the browser with near-native CUDA API compatibility.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Browser Environment                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         CUDA SDK for WebGPU                         │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │ Runtime  │ │   Math   │ │  Vector  │ │    f64   │ │ Texture  │   │    │
│  │  │   API    │ │ Library  │ │  Types   │ │ Emulation│ │ Surface  │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │  Atomic  │ │  Warp    │ │  Coop    │ │   Grid   │ │  cuBLAS  │   │    │
│  │  │   Ops    │ │ Prims    │ │  Groups  │ │   Sync   │ │  (BLAS)  │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                             │    │
│  │  │  cuFFT   │ │  cuRAND  │ │Transpiler│                             │    │
│  │  │  (FFT)   │ │  (RNG)   │ │ CUDA→WGSL│                             │    │
│  │  └──────────┘ └──────────┘ └──────────┘                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐  │
│  │                    WebGPU Compute Shaders (WGSL)                      │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  • CUDA Math Functions (100+)   • f64 Double Precision Emulation     │  │
│  │  • Warp Shuffle/Ballot/Vote     • Atomic Operations (i32/u32/f32/f64)│  │
│  │  • Matrix Operations (GEMM)     • FFT (Cooley-Tukey)                 │  │
│  │  • Grid-Wide Synchronization    • Cooperative Group Collectives      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Full CUDA Runtime API** - Device, memory, stream, event, and kernel management
- **Double Precision (f64)** - Emulated 64-bit floating point with ~44-bit precision
- **Grid-Wide Synchronization** - Atomic counter-based barriers across workgroups
- **100+ Math Functions** - Trigonometric, exponential, special functions
- **Vector Types** - float2/3/4, double2/3/4, int2/3/4, matrices, quaternions
- **Warp Primitives** - Shuffle, ballot, vote operations
- **Cooperative Groups** - Block and grid-level collective operations
- **cuBLAS/cuFFT/cuRAND** - Compatible linear algebra, FFT, and RNG

## CUDA SDK Features

### Runtime API

Complete CUDA Runtime API compatibility layer:

```typescript
import {
  cudaInit, cudaMalloc, cudaMemcpy, cudaFree,
  cudaStreamCreate, cudaEventRecord,
  cudaLaunchKernel, dim3, CudaMemcpyKind
} from './cuda';

// Initialize and allocate
await cudaInit();
const { ptr: d_data } = cudaMalloc(1024 * Float32Array.BYTES_PER_ELEMENT);

// Copy data to device
await cudaMemcpy(d_data, hostData, size, CudaMemcpyKind.cudaMemcpyHostToDevice);

// Launch kernel
await cudaLaunch(kernel, dim3(numBlocks), dim3(256), [d_data, n]);
```

### Double Precision (f64) Support

Emulated 64-bit floating point operations using double-single arithmetic:

```typescript
import { Double64Buffer, withCudaF64, Double3 } from './cuda';

// CPU-side double vectors
const v1 = Double3.make(1.0, 2.0, 3.0);
const v2 = Double3.make(4.0, 5.0, 6.0);
const dot = v1.dot(v2);  // Full f64 precision

// GPU buffer for double precision
const buffer = new Double64Buffer(device, 1000);
buffer.writeData(new Float64Array([1.0, 2.0, 3.0]));

// Use in shaders
const shader = withCudaF64(`
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      // f64 is stored as vec2<u32>
      let a = f64_from_f32(data[gid.x]);
      let b = f64_from_f32(data[gid.x + 1u]);

      // Full double precision arithmetic
      let sum = f64_add(a, b);
      let product = f64_mul(a, b);
      let root = f64_sqrt(sum);

      // Math functions
      let s = f64_sin(a);
      let e = f64_exp(b);

      result[gid.x] = f64_to_f32(sum);
  }
`);
```

**f64 Operations Available:**
- Arithmetic: `f64_add`, `f64_sub`, `f64_mul`, `f64_div`, `f64_neg`, `f64_abs`
- Comparison: `f64_eq`, `f64_lt`, `f64_le`, `f64_gt`, `f64_ge`, `f64_min`, `f64_max`
- Math: `f64_sqrt`, `f64_rcp`, `f64_rsqrt`, `f64_pow`, `f64_fma`
- Trig: `f64_sin`, `f64_cos`, `f64_tan`, `f64_asin`, `f64_acos`, `f64_atan`
- Exp/Log: `f64_exp`, `f64_exp2`, `f64_log`, `f64_log2`
- Rounding: `f64_floor`, `f64_ceil`, `f64_round`, `f64_trunc`
- Vectors: `double2`, `double3`, `double4` with dot, cross, normalize

### Grid-Wide Synchronization

Atomic counter-based grid barriers for synchronizing across all workgroups:

```typescript
import { GridSyncBuffer, MultiPassExecutor, withGridSync } from './cuda';

// Create sync buffer for up to 64 workgroups
const syncBuffer = new GridSyncBuffer(device, 64);
await syncBuffer.reset();

// Use in shader
const shader = withGridSync(`
  @group(0) @binding(0) var<storage, read_write> data: array<f32>;
  @group(0) @binding(1) var<storage, read_write> sync_buf: array<atomic<u32>>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let grid = this_grid();

      // Phase 1: All workgroups compute partial results
      compute_partial(gid.x);

      // Grid-wide barrier - all workgroups sync here
      grid_sync(&sync_buf, num_workgroups);

      // Phase 2: All workgroups can now see Phase 1 results
      aggregate_results(gid.x);
  }
`);

// Multi-pass execution helper
const executor = new MultiPassExecutor(device, 64);
await executor.executeWithSync([
  { pipeline: pass1Pipeline, bindGroups: [bg1], workgroupCount: [64, 1, 1] },
  { pipeline: pass2Pipeline, bindGroups: [bg2], workgroupCount: [64, 1, 1] },
]);
```

**Grid Sync Features:**
- `grid_sync()` - Full grid barrier
- Named barriers for partial synchronization
- Producer-consumer patterns
- Grid-wide reductions: `grid_reduce_sum_f32`, `grid_reduce_max_f32`
- Progress tracking and multi-phase coordination

### Math Library (100+ Functions)

| Category | Functions |
|----------|-----------|
| **Trigonometric** | sin, cos, tan, asin, acos, atan, atan2, sincos, sinpi, cospi |
| **Hyperbolic** | sinh, cosh, tanh, asinh, acosh, atanh |
| **Exponential** | exp, exp2, exp10, expm1, log, log2, log10, log1p |
| **Power** | pow, sqrt, rsqrt, cbrt, hypot, norm3d, norm4d |
| **Special** | erf, erfc, erfinv, lgamma, tgamma, j0, j1, y0, y1, bessel |
| **Rounding** | floor, ceil, trunc, round, rint, nearbyint |
| **Comparison** | fmin, fmax, fdim, fmod, remainder |
| **Integer** | clz, ctz, popc, brev, ffs |

```typescript
import { withCudaMath } from './cuda';

const shaderCode = withCudaMath(`
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let x = data[gid.x];
      result[gid.x] = erff(x) + lgammaf(abs(x));
  }
`);
```

### Vector & Matrix Types

```typescript
import { Float3, Float4, Double3, Mat4, Quat } from './cuda';

// Float vectors
const v1 = Float3.make(1, 2, 3);
const v2 = Float3.make(4, 5, 6);
const cross = v1.cross(v2);
const dot = v1.dot(v2);

// Double precision vectors
const d1 = Double3.make(1.0, 2.0, 3.0);
const d2 = Double3.make(4.0, 5.0, 6.0);
const preciseDot = d1.dot(d2);

// Matrix operations
const view = Mat4.lookAt(eye, center, up);
const proj = Mat4.perspective(fov, aspect, near, far);
const mvp = proj.mul(view);

// Quaternion rotation
const q = Quat.fromAxisAngle(Float3.make(0, 1, 0), Math.PI / 4);
const rotated = q.rotateVec3(position);
```

### Atomic Operations

Native and emulated atomics for i32, u32, f32, and f64:

```wgsl
${CUDA_ATOMIC_WGSL}

@compute @workgroup_size(256)
fn main() {
    // Native integer atomics
    atomicAdd_i32(&counter, 1);
    atomicMax_u32(&maxVal, myValue);

    // Emulated float atomics (CAS loop)
    atomicAdd_f32_emulated(&floatSum, myFloat);

    // Emulated double atomics (two-stage CAS)
    let myDouble = f64_from_f32(myFloat);
    atomicAdd_f64_emulated(&doubleLow, &doubleHigh, myDouble);
}
```

### Warp-Level Primitives

```wgsl
${CUDA_WARP_WGSL}

@compute @workgroup_size(256)
fn main() {
    // Warp reduction
    var sum = __warp_reduce_sum_f32(myValue);

    // Shuffle operations
    var neighbor = __shfl_xor_sync_f32(0xFFFFFFFF, myValue, 1u, 32u);

    // Voting
    var allTrue = __all_sync(0xFFFFFFFF, predicate);
    var ballot = __ballot_sync(0xFFFFFFFF, predicate);
}
```

### Cooperative Groups

```wgsl
${CUDA_COOPERATIVE_GROUPS_WGSL}

@compute @workgroup_size(256)
fn main() {
    let block = this_thread_block();

    // Block-wide reduction (f32 and f64)
    var sum = reduce_sum_f32(block, myValue);
    var doubleSum = reduce_sum_f64(block, myDoubleValue);

    // Prefix scan
    var prefix = inclusive_scan_sum_f32(block, myValue);

    // Grid-wide sync (requires sync buffer)
    let grid = this_grid();
    grid_group_sync_with_buffer(grid, &sync_buffer);
}
```

### cuBLAS (Linear Algebra)

```typescript
import {
  cublasCreate, cublasSgemm, cublasSaxpy,
  CublasOperation
} from './cuda';

const { handle } = cublasCreate();

// SAXPY: y = alpha * x + y
await cublasSaxpy(handle, n, alpha, x, 1, y, 1);

// SGEMM: C = alpha * A * B + beta * C
await cublasSgemm(
  handle,
  CublasOperation.CUBLAS_OP_N,
  CublasOperation.CUBLAS_OP_N,
  m, n, k,
  alpha, A, lda, B, ldb,
  beta, C, ldc
);
```

### cuFFT (Fast Fourier Transform)

```typescript
import {
  cufftPlan1d, cufftExecC2C, cufftDestroy,
  CufftType, CufftDirection
} from './cuda';

const { plan } = cufftPlan1d(1024, CufftType.CUFFT_C2C);
await cufftExecC2C(plan, input, output, CufftDirection.CUFFT_FORWARD);
await cufftExecC2C(plan, output, result, CufftDirection.CUFFT_INVERSE);
cufftDestroy(plan);
```

### cuRAND (Random Number Generation)

```typescript
import {
  curandCreateGenerator, curandSetPseudoRandomGeneratorSeed,
  curandGenerateUniform, curandGenerateNormal,
  CurandRngType
} from './cuda';

const { generator } = curandCreateGenerator(CurandRngType.CURAND_RNG_PSEUDO_PHILOX4_32_10);
curandSetPseudoRandomGeneratorSeed(generator, BigInt(12345));

await curandGenerateUniform(generator, d_output, count);
await curandGenerateNormal(generator, d_output, count, 0.0, 1.0);
```

## Project Structure

```
CUDAWeb/
├── src/
│   ├── cuda/                    # CUDA SDK for WebGPU
│   │   ├── runtime/             # Runtime API (device, memory, stream, kernel)
│   │   ├── math/                # Math library (WGSL + TypeScript)
│   │   ├── vector/              # Vector/matrix types (float, double, int)
│   │   ├── f64/                 # Double precision emulation
│   │   ├── grid_sync/           # Grid-wide synchronization
│   │   ├── texture/             # Texture/surface operations
│   │   ├── warp/                # Warp-level primitives
│   │   ├── cooperative_groups/  # Cooperative group operations
│   │   ├── atomic/              # Atomic operations (native + emulated)
│   │   ├── blas/                # cuBLAS implementation
│   │   ├── fft/                 # cuFFT implementation
│   │   ├── rand/                # cuRAND implementation
│   │   ├── transpiler/          # CUDA to WGSL utilities
│   │   └── index.ts             # Main exports
│   └── core/                    # Core WebGPU abstractions
│       ├── gpu-context.ts       # WebGPU device/context management
│       ├── buffer-manager.ts    # GPU buffer allocation
│       └── pipeline.ts          # Compute/render pipeline builders
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Browser Requirements

- Chrome 113+ or Firefox 121+ (WebGPU support required)
- WebAssembly support

## CUDA to WebGPU Migration Guide

The SDK includes transpiler utilities for converting CUDA code:

| CUDA | WebGPU/WGSL |
|------|-------------|
| `threadIdx.x` | `local_invocation_id.x` |
| `blockIdx.x` | `workgroup_id.x` |
| `blockDim.x` | `workgroup_size.x` |
| `__shared__` | `var<workgroup>` |
| `__syncthreads()` | `workgroupBarrier()` |
| `atomicAdd(&x, v)` | `atomicAdd(&x, v)` |
| `double` | `f64` (emulated via `vec2<u32>`) |
| `cooperative_groups::grid_group::sync()` | `grid_sync(&sync_buf, n)` |

```typescript
import { transpileCudaToWgsl, generateKernelTemplate } from './cuda';

// Auto-convert CUDA patterns
const wgslCode = transpileCudaToWgsl(cudaCode);

// Generate kernel template
const template = generateKernelTemplate({
  name: 'myKernel',
  inputs: [
    { name: 'input', type: 'array<f32>', access: 'read' },
    { name: 'output', type: 'array<f32>', access: 'read_write' }
  ],
  workgroupSize: [256, 1, 1]
});
```

## Feature Comparison

| Feature | CUDA | CUDAWeb | Notes |
|---------|------|---------|-------|
| Single precision (f32) | Native | Native | Full support |
| Double precision (f64) | Native | Emulated | ~44-bit precision via double-single arithmetic |
| Grid-wide sync | Native | Emulated | Atomic counter-based barriers |
| Warp primitives | Native | Emulated | Via shared memory |
| Integer atomics | Native | Native | i32, u32 |
| Float atomics | Native | Emulated | CAS loops |
| Double atomics | Native | Emulated | Two-stage CAS |
| Dynamic parallelism | Yes | No | Not supported in WebGPU |
| Max workgroup size | 1024 | ~256 | WebGPU limitation |

## Known Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| Double precision (f64) | Emulated | ~44-bit precision (vs 53-bit native) |
| Grid-wide sync | Emulated | Spin-waiting, requires sync buffer |
| Warp primitives | Emulated | Via shared memory |
| Float/Double atomics | Emulated | CAS loops (slower than native) |
| Dynamic parallelism | Not supported | WebGPU limitation |
| Maximum workgroup size | ~256 | WebGPU limitation |

## License

MIT License
