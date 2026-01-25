# CUDAWeb - CUDA SDK for WebGPU

A comprehensive CUDA-compatible SDK for WebGPU, enabling GPU compute applications to run in the browser. Includes a complete CUDA Runtime API, math libraries, and high-performance 3D Gaussian Splatting implementation.

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
│  │  │ Runtime  │ │   Math   │ │  Vector  │ │ Texture  │ │  Atomic  │   │    │
│  │  │   API    │ │ Library  │ │  Types   │ │ Surface  │ │   Ops    │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │  Warp    │ │  Coop    │ │  cuBLAS  │ │  cuFFT   │ │ cuRAND   │   │    │
│  │  │ Prims    │ │  Groups  │ │  (BLAS)  │ │  (FFT)   │ │  (RNG)   │   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│  ┌─────────────────┐  ┌────────────┴───────────┐  ┌─────────────────────┐   │
│  │   Web Frontend  │  │   WebGPU Abstraction   │  │    WASM Runtime     │   │
│  │   (TypeScript)  │  │        Layer           │  │    (Rust/WASM)      │   │
│  └────────┬────────┘  └────────────┬───────────┘  └──────────┬──────────┘   │
│           │                        │                         │              │
│           └────────────────────────┼─────────────────────────┘              │
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐  │
│  │                    WebGPU Compute Shaders (WGSL)                      │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  • CUDA Math Functions          • Cooperative Group Operations       │  │
│  │  • Warp Shuffle/Ballot/Vote     • Atomic Operations (int/float)      │  │
│  │  • Matrix Operations (GEMM)     • FFT (Cooley-Tukey)                 │  │
│  │  • Random Number Generation     • Gaussian Splatting Kernels         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
import { Float3, Float4, Mat4, Quat } from './cuda';

// Vector operations
const v1 = Float3.make(1, 2, 3);
const v2 = Float3.make(4, 5, 6);
const cross = v1.cross(v2);
const dot = v1.dot(v2);

// Matrix operations
const view = Mat4.lookAt(eye, center, up);
const proj = Mat4.perspective(fov, aspect, near, far);
const mvp = proj.mul(view);

// Quaternion rotation
const q = Quat.fromAxisAngle(Float3.make(0, 1, 0), Math.PI / 4);
const rotated = q.rotateVec3(position);
```

### Warp-Level Primitives

```wgsl
// Include warp primitives in your shader
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

    // Block-wide reduction
    var sum = reduce_sum_f32(block, myValue);

    // Prefix scan
    var prefix = inclusive_scan_sum_f32(block, myValue);

    // Collective voting
    var allAgree = all_predicate(block, condition);
}
```

### cuBLAS (Linear Algebra)

```typescript
import {
  cublasCreate, cublasSgemm, cublasSaxpy,
  CublasOperation
} from './cuda';

// Create handle
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

// Create plan
const { plan } = cufftPlan1d(1024, CufftType.CUFFT_C2C);

// Execute forward FFT
await cufftExecC2C(plan, input, output, CufftDirection.CUFFT_FORWARD);

// Execute inverse FFT
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

// Create Philox RNG
const { generator } = curandCreateGenerator(CurandRngType.CURAND_RNG_PSEUDO_PHILOX4_32_10);
curandSetPseudoRandomGeneratorSeed(generator, BigInt(12345));

// Generate uniform distribution [0, 1)
await curandGenerateUniform(generator, d_output, count);

// Generate normal distribution (mean=0, stddev=1)
await curandGenerateNormal(generator, d_output, count, 0.0, 1.0);
```

### Texture & Surface Operations

```typescript
import {
  cudaMallocArray, cudaCreateTextureObject,
  getDefaultTextureDesc, CudaChannelFormatKind
} from './cuda';

// Create texture array
const channelDesc = { x: 32, y: 32, z: 32, w: 32, f: CudaChannelFormatKind.cudaChannelFormatKindFloat };
const { array } = cudaMallocArray(width, height, channelDesc);

// Create texture object with sampler
const texDesc = getDefaultTextureDesc();
const { texObj } = cudaCreateTextureObject({ array, format: 'rgba32float' }, texDesc);
```

## Gaussian Splatting

Complete 3D Gaussian Splatting implementation for browser-based rendering:

### Key Features
- **Full 3DGS Rendering**: Complete implementation of 3D Gaussian Splatting
- **Large Model Support**: Streaming weight loader for 2GB+ models
- **360° View Synthesis**: Full novel view rendering capabilities
- **Tile-Based Rasterization**: Efficient GPU-based splatting

### Pipeline
1. **Preprocessing**: Transform 3D Gaussians to view space
2. **Covariance**: Compute 2D covariance from 3D + camera
3. **Tiling**: Assign Gaussians to screen tiles (16x16)
4. **Sorting**: Radix sort by depth within tiles
5. **Rasterization**: Alpha-blend Gaussians front-to-back

## Project Structure

```
CUDAWeb/
├── src/
│   ├── cuda/                    # CUDA SDK for WebGPU
│   │   ├── runtime/             # Runtime API (device, memory, stream, kernel)
│   │   ├── math/                # Math library (WGSL + TypeScript)
│   │   ├── vector/              # Vector/matrix types
│   │   ├── texture/             # Texture/surface operations
│   │   ├── warp/                # Warp-level primitives
│   │   ├── cooperative_groups/  # Cooperative group operations
│   │   ├── atomic/              # Atomic operations
│   │   ├── blas/                # cuBLAS implementation
│   │   ├── fft/                 # cuFFT implementation
│   │   ├── rand/                # cuRAND implementation
│   │   ├── transpiler/          # CUDA to WGSL utilities
│   │   └── index.ts             # Main exports
│   ├── core/                    # Core WebGPU abstractions
│   │   ├── gpu-context.ts       # WebGPU device/context management
│   │   ├── buffer-manager.ts    # GPU buffer allocation
│   │   └── pipeline.ts          # Compute/render pipeline builders
│   ├── shaders/                 # WGSL compute shaders
│   │   ├── gaussian/            # Gaussian splatting kernels
│   │   ├── sorting/             # Radix sort for depth ordering
│   │   └── rasterize/           # Rasterization shaders
│   ├── loaders/                 # Model/weight loading
│   │   ├── weight-loader.ts     # Streaming weight loader
│   │   ├── ply-loader.ts        # PLY point cloud loader
│   │   └── splat-loader.ts      # .splat format loader
│   ├── renderer/                # Rendering system
│   │   ├── gaussian-renderer.ts
│   │   └── camera.ts
│   └── wasm/                    # Rust/WASM modules
│       ├── src/
│       └── Cargo.toml
├── public/                      # Static assets
├── tests/                       # Test files
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Getting Started

```bash
# Install dependencies
npm install

# Build WASM module
cd src/wasm && wasm-pack build --target web

# Start development server
npm run dev

# Build for production
npm run build
```

## Browser Requirements

- Chrome 113+ or Firefox 121+ (WebGPU support required)
- 4GB+ GPU memory recommended for large scenes
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
| `__shfl_down_sync()` | Emulated via shared memory |

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

## Known Limitations

| Feature | Status |
|---------|--------|
| Double precision (f64) | Not supported in most WebGPU implementations |
| Warp primitives | Emulated via shared memory |
| Dynamic parallelism | Not supported |
| Maximum workgroup size | ~256 (vs CUDA 1024) |
| Float atomics | Emulated via CAS loops |
| Grid-wide sync | Not supported |

## License

MIT License
