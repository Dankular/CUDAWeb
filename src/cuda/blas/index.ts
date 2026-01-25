/**
 * cuBLAS-like Linear Algebra Operations for WebGPU
 *
 * Provides CUDA cuBLAS-compatible BLAS operations
 */

import { CudaError, cudaDevice } from '../runtime/cuda-device';
import { CudaPointer, getBuffer, cudaMalloc, cudaFree, cudaMemcpy, CudaMemcpyKind } from '../runtime/cuda-memory';
import { CudaStream } from '../runtime/cuda-stream';

export enum CublasStatus {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_MAPPING_ERROR = 11,
  CUBLAS_STATUS_EXECUTION_FAILED = 13,
  CUBLAS_STATUS_INTERNAL_ERROR = 14,
  CUBLAS_STATUS_NOT_SUPPORTED = 15,
}

export enum CublasOperation {
  CUBLAS_OP_N = 0, // No transpose
  CUBLAS_OP_T = 1, // Transpose
  CUBLAS_OP_C = 2, // Conjugate transpose
}

export enum CublasFillMode {
  CUBLAS_FILL_MODE_LOWER = 0,
  CUBLAS_FILL_MODE_UPPER = 1,
  CUBLAS_FILL_MODE_FULL = 2,
}

export enum CublasDiagType {
  CUBLAS_DIAG_NON_UNIT = 0,
  CUBLAS_DIAG_UNIT = 1,
}

export enum CublasSideMode {
  CUBLAS_SIDE_LEFT = 0,
  CUBLAS_SIDE_RIGHT = 1,
}

export interface CublasHandle {
  id: number;
  stream: CudaStream | null;
  pipelines: Map<string, GPUComputePipeline>;
  shaderModules: Map<string, GPUShaderModule>;
}

// WGSL shaders for BLAS operations
const BLAS_SHADERS = {
  // SAXPY: y = alpha * x + y
  saxpy: `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<f32>; // alpha, n, incx, incy

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = u32(params.y);
    let alpha = params.x;
    let incx = u32(params.z);
    let incy = u32(params.w);

    if (idx < n) {
        y[idx * incy] = alpha * x[idx * incx] + y[idx * incy];
    }
}`,

  // SSCAL: x = alpha * x
  sscal: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<f32>; // alpha, n, incx, 0

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = u32(params.y);
    let alpha = params.x;
    let incx = u32(params.z);

    if (idx < n) {
        x[idx * incx] = alpha * x[idx * incx];
    }
}`,

  // SDOT: dot product
  sdot: `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>; // n, incx, incy, 0

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    let n = u32(params.x);
    let incx = u32(params.y);
    let incy = u32(params.z);

    var sum: f32 = 0.0;
    if (idx < n) {
        sum = x[idx * incx] * y[idx * incy];
    }
    shared_sum[local_idx] = sum;
    workgroupBarrier();

    // Reduction
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        // Atomic add to result (as float bits)
        var old_bits: u32;
        loop {
            old_bits = atomicLoad(&result);
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val + shared_sum[0];
            let new_bits = bitcast<u32>(new_val);
            let res = atomicCompareExchangeWeak(&result, old_bits, new_bits);
            if (res.exchanged) { break; }
        }
    }
}`,

  // SNRM2: Euclidean norm
  snrm2: `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(2) var<uniform> params: vec2<f32>; // n, incx

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    let n = u32(params.x);
    let incx = u32(params.y);

    var sum: f32 = 0.0;
    if (idx < n) {
        let val = x[idx * incx];
        sum = val * val;
    }
    shared_sum[local_idx] = sum;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        var old_bits: u32;
        loop {
            old_bits = atomicLoad(&result);
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val + shared_sum[0];
            let new_bits = bitcast<u32>(new_val);
            let res = atomicCompareExchangeWeak(&result, old_bits, new_bits);
            if (res.exchanged) { break; }
        }
    }
}`,

  // SASUM: Sum of absolute values
  sasum: `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: atomic<u32>;
@group(0) @binding(2) var<uniform> params: vec2<f32>; // n, incx

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = gid.x;
    let local_idx = lid.x;
    let n = u32(params.x);
    let incx = u32(params.y);

    var sum: f32 = 0.0;
    if (idx < n) {
        sum = abs(x[idx * incx]);
    }
    shared_sum[local_idx] = sum;
    workgroupBarrier();

    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_sum[local_idx] = shared_sum[local_idx] + shared_sum[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        var old_bits: u32;
        loop {
            old_bits = atomicLoad(&result);
            let old_val = bitcast<f32>(old_bits);
            let new_val = old_val + shared_sum[0];
            let new_bits = bitcast<u32>(new_val);
            let res = atomicCompareExchangeWeak(&result, old_bits, new_bits);
            if (res.exchanged) { break; }
        }
    }
}`,

  // SGEMV: y = alpha * A * x + beta * y
  sgemv: `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: array<f32, 8>; // alpha, beta, m, n, lda, incx, incy, trans

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let alpha = params[0];
    let beta = params[1];
    let m = u32(params[2]);
    let n = u32(params[3]);
    let lda = u32(params[4]);
    let incx = u32(params[5]);
    let incy = u32(params[6]);
    let trans = u32(params[7]);

    if (row >= m) { return; }

    var sum: f32 = 0.0;
    if (trans == 0u) {
        // No transpose: y = alpha * A * x + beta * y
        for (var col: u32 = 0u; col < n; col = col + 1u) {
            sum = sum + A[row + col * lda] * x[col * incx];
        }
    } else {
        // Transpose: y = alpha * A^T * x + beta * y
        for (var col: u32 = 0u; col < n; col = col + 1u) {
            sum = sum + A[col + row * lda] * x[col * incx];
        }
    }

    y[row * incy] = alpha * sum + beta * y[row * incy];
}`,

  // SGEMM: C = alpha * A * B + beta * C
  sgemm: `
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: array<f32, 12>;
// params: alpha, beta, m, n, k, lda, ldb, ldc, transA, transB, 0, 0

const TILE_SIZE: u32 = 16u;
var<workgroup> As: array<array<f32, 16>, 16>;
var<workgroup> Bs: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let alpha = params[0];
    let beta = params[1];
    let M = u32(params[2]);
    let N = u32(params[3]);
    let K = u32(params[4]);
    let lda = u32(params[5]);
    let ldb = u32(params[6]);
    let ldc = u32(params[7]);
    let transA = u32(params[8]);
    let transB = u32(params[9]);

    let row = gid.y;
    let col = gid.x;
    let localRow = lid.y;
    let localCol = lid.x;

    var sum: f32 = 0.0;

    let numTiles = (K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        // Load A tile
        let aRow = wid.y * TILE_SIZE + localRow;
        let aCol = t * TILE_SIZE + localCol;
        if (aRow < M && aCol < K) {
            if (transA == 0u) {
                As[localRow][localCol] = A[aRow * lda + aCol];
            } else {
                As[localRow][localCol] = A[aCol * lda + aRow];
            }
        } else {
            As[localRow][localCol] = 0.0;
        }

        // Load B tile
        let bRow = t * TILE_SIZE + localRow;
        let bCol = wid.x * TILE_SIZE + localCol;
        if (bRow < K && bCol < N) {
            if (transB == 0u) {
                Bs[localRow][localCol] = B[bRow * ldb + bCol];
            } else {
                Bs[localRow][localCol] = B[bCol * ldb + bRow];
            }
        } else {
            Bs[localRow][localCol] = 0.0;
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + As[localRow][k] * Bs[k][localCol];
        }

        workgroupBarrier();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}`,

  // SCOPY: y = x
  scopy: `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<f32>; // n, incx, incy, 0

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = u32(params.x);
    let incx = u32(params.y);
    let incy = u32(params.z);

    if (idx < n) {
        y[idx * incy] = x[idx * incx];
    }
}`,
};

class CublasContext {
  private static instance: CublasContext | null = null;
  private handles: Map<number, CublasHandle> = new Map();
  private nextHandleId: number = 1;

  private constructor() {}

  static getInstance(): CublasContext {
    if (!CublasContext.instance) {
      CublasContext.instance = new CublasContext();
    }
    return CublasContext.instance;
  }

  private getDevice(): GPUDevice | null {
    return cudaDevice.getDevice();
  }

  // Handle management
  cublasCreate(): { handle: CublasHandle | null; status: CublasStatus } {
    const device = this.getDevice();
    if (!device) {
      return { handle: null, status: CublasStatus.CUBLAS_STATUS_NOT_INITIALIZED };
    }

    const handle: CublasHandle = {
      id: this.nextHandleId++,
      stream: null,
      pipelines: new Map(),
      shaderModules: new Map(),
    };

    // Pre-compile shaders
    for (const [name, code] of Object.entries(BLAS_SHADERS)) {
      const module = device.createShaderModule({ code, label: `cublas_${name}` });
      handle.shaderModules.set(name, module);
    }

    this.handles.set(handle.id, handle);
    return { handle, status: CublasStatus.CUBLAS_STATUS_SUCCESS };
  }

  cublasDestroy(handle: CublasHandle): CublasStatus {
    this.handles.delete(handle.id);
    return CublasStatus.CUBLAS_STATUS_SUCCESS;
  }

  cublasSetStream(handle: CublasHandle, stream: CudaStream): CublasStatus {
    const h = this.handles.get(handle.id);
    if (!h) return CublasStatus.CUBLAS_STATUS_INVALID_VALUE;
    h.stream = stream;
    return CublasStatus.CUBLAS_STATUS_SUCCESS;
  }

  // Get or create pipeline
  private getPipeline(
    handle: CublasHandle,
    name: string,
    bindGroupLayout: GPUBindGroupLayout
  ): GPUComputePipeline | null {
    const device = this.getDevice();
    if (!device) return null;

    const cached = handle.pipelines.get(name);
    if (cached) return cached;

    const module = handle.shaderModules.get(name);
    if (!module) return null;

    const layout = device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const pipeline = device.createComputePipeline({
      layout,
      compute: { module, entryPoint: 'main' },
    });

    handle.pipelines.set(name, pipeline);
    return pipeline;
  }

  // BLAS Level 1 Operations

  async cublasSaxpy(
    handle: CublasHandle,
    n: number,
    alpha: number,
    x: CudaPointer,
    incx: number,
    y: CudaPointer,
    incy: number
  ): Promise<CublasStatus> {
    const device = this.getDevice();
    if (!device) return CublasStatus.CUBLAS_STATUS_NOT_INITIALIZED;

    const xBuffer = getBuffer(x);
    const yBuffer = getBuffer(y);
    if (!xBuffer || !yBuffer) return CublasStatus.CUBLAS_STATUS_INVALID_VALUE;

    // Create uniform buffer
    const uniformData = new Float32Array([alpha, n, incx, incy]);
    const uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Create bind group layout
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.getPipeline(handle, 'saxpy', bindGroupLayout);
    if (!pipeline) return CublasStatus.CUBLAS_STATUS_INTERNAL_ERROR;

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: xBuffer } },
        { binding: 1, resource: { buffer: yBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    uniformBuffer.destroy();

    return CublasStatus.CUBLAS_STATUS_SUCCESS;
  }

  async cublasSscal(
    handle: CublasHandle,
    n: number,
    alpha: number,
    x: CudaPointer,
    incx: number
  ): Promise<CublasStatus> {
    const device = this.getDevice();
    if (!device) return CublasStatus.CUBLAS_STATUS_NOT_INITIALIZED;

    const xBuffer = getBuffer(x);
    if (!xBuffer) return CublasStatus.CUBLAS_STATUS_INVALID_VALUE;

    const uniformData = new Float32Array([alpha, n, incx, 0]);
    const uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.getPipeline(handle, 'sscal', bindGroupLayout);
    if (!pipeline) return CublasStatus.CUBLAS_STATUS_INTERNAL_ERROR;

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: xBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    uniformBuffer.destroy();

    return CublasStatus.CUBLAS_STATUS_SUCCESS;
  }

  async cublasScopy(
    handle: CublasHandle,
    n: number,
    x: CudaPointer,
    incx: number,
    y: CudaPointer,
    incy: number
  ): Promise<CublasStatus> {
    const device = this.getDevice();
    if (!device) return CublasStatus.CUBLAS_STATUS_NOT_INITIALIZED;

    const xBuffer = getBuffer(x);
    const yBuffer = getBuffer(y);
    if (!xBuffer || !yBuffer) return CublasStatus.CUBLAS_STATUS_INVALID_VALUE;

    const uniformData = new Float32Array([n, incx, incy, 0]);
    const uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.getPipeline(handle, 'scopy', bindGroupLayout);
    if (!pipeline) return CublasStatus.CUBLAS_STATUS_INTERNAL_ERROR;

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: xBuffer } },
        { binding: 1, resource: { buffer: yBuffer } },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 256));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    uniformBuffer.destroy();

    return CublasStatus.CUBLAS_STATUS_SUCCESS;
  }

  // BLAS Level 2: Matrix-vector operations

  async cublasSgemv(
    handle: CublasHandle,
    trans: CublasOperation,
    m: number,
    n: number,
    alpha: number,
    A: CudaPointer,
    lda: number,
    x: CudaPointer,
    incx: number,
    beta: number,
    y: CudaPointer,
    incy: number
  ): Promise<CublasStatus> {
    const device = this.getDevice();
    if (!device) return CublasStatus.CUBLAS_STATUS_NOT_INITIALIZED;

    const aBuffer = getBuffer(A);
    const xBuffer = getBuffer(x);
    const yBuffer = getBuffer(y);
    if (!aBuffer || !xBuffer || !yBuffer) return CublasStatus.CUBLAS_STATUS_INVALID_VALUE;

    const uniformData = new Float32Array([alpha, beta, m, n, lda, incx, incy, trans]);
    const uniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.getPipeline(handle, 'sgemv', bindGroupLayout);
    if (!pipeline) return CublasStatus.CUBLAS_STATUS_INTERNAL_ERROR;

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: xBuffer } },
        { binding: 2, resource: { buffer: yBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(m / 256));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    uniformBuffer.destroy();

    return CublasStatus.CUBLAS_STATUS_SUCCESS;
  }

  // BLAS Level 3: Matrix-matrix operations

  async cublasSgemm(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: number,
    n: number,
    k: number,
    alpha: number,
    A: CudaPointer,
    lda: number,
    B: CudaPointer,
    ldb: number,
    beta: number,
    C: CudaPointer,
    ldc: number
  ): Promise<CublasStatus> {
    const device = this.getDevice();
    if (!device) return CublasStatus.CUBLAS_STATUS_NOT_INITIALIZED;

    const aBuffer = getBuffer(A);
    const bBuffer = getBuffer(B);
    const cBuffer = getBuffer(C);
    if (!aBuffer || !bBuffer || !cBuffer) return CublasStatus.CUBLAS_STATUS_INVALID_VALUE;

    const uniformData = new Float32Array([alpha, beta, m, n, k, lda, ldb, ldc, transa, transb, 0, 0]);
    const uniformBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = this.getPipeline(handle, 'sgemm', bindGroupLayout);
    if (!pipeline) return CublasStatus.CUBLAS_STATUS_INTERNAL_ERROR;

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: aBuffer } },
        { binding: 1, resource: { buffer: bBuffer } },
        { binding: 2, resource: { buffer: cBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(n / 16), Math.ceil(m / 16));
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
    uniformBuffer.destroy();

    return CublasStatus.CUBLAS_STATUS_SUCCESS;
  }
}

// Export singleton
export const cublasContext = CublasContext.getInstance();

// CUDA-like API exports
export function cublasCreate(): { handle: CublasHandle | null; status: CublasStatus } {
  return cublasContext.cublasCreate();
}

export function cublasDestroy(handle: CublasHandle): CublasStatus {
  return cublasContext.cublasDestroy(handle);
}

export function cublasSetStream(handle: CublasHandle, stream: CudaStream): CublasStatus {
  return cublasContext.cublasSetStream(handle, stream);
}

export function cublasSaxpy(
  handle: CublasHandle, n: number, alpha: number,
  x: CudaPointer, incx: number, y: CudaPointer, incy: number
): Promise<CublasStatus> {
  return cublasContext.cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

export function cublasSscal(
  handle: CublasHandle, n: number, alpha: number, x: CudaPointer, incx: number
): Promise<CublasStatus> {
  return cublasContext.cublasSscal(handle, n, alpha, x, incx);
}

export function cublasScopy(
  handle: CublasHandle, n: number,
  x: CudaPointer, incx: number, y: CudaPointer, incy: number
): Promise<CublasStatus> {
  return cublasContext.cublasScopy(handle, n, x, incx, y, incy);
}

export function cublasSgemv(
  handle: CublasHandle, trans: CublasOperation,
  m: number, n: number, alpha: number,
  A: CudaPointer, lda: number,
  x: CudaPointer, incx: number,
  beta: number, y: CudaPointer, incy: number
): Promise<CublasStatus> {
  return cublasContext.cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

export function cublasSgemm(
  handle: CublasHandle, transa: CublasOperation, transb: CublasOperation,
  m: number, n: number, k: number,
  alpha: number, A: CudaPointer, lda: number,
  B: CudaPointer, ldb: number,
  beta: number, C: CudaPointer, ldc: number
): Promise<CublasStatus> {
  return cublasContext.cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
