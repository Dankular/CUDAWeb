/**
 * cuFFT-like FFT Operations for WebGPU
 *
 * Provides CUDA cuFFT-compatible Fast Fourier Transform operations
 */

import { CudaError, cudaDevice } from '../runtime/cuda-device';
import { CudaPointer, getBuffer } from '../runtime/cuda-memory';
import { CudaStream } from '../runtime/cuda-stream';

export enum CufftResult {
  CUFFT_SUCCESS = 0,
  CUFFT_INVALID_PLAN = 1,
  CUFFT_ALLOC_FAILED = 2,
  CUFFT_INVALID_TYPE = 3,
  CUFFT_INVALID_VALUE = 4,
  CUFFT_INTERNAL_ERROR = 5,
  CUFFT_EXEC_FAILED = 6,
  CUFFT_SETUP_FAILED = 7,
  CUFFT_INVALID_SIZE = 8,
  CUFFT_UNALIGNED_DATA = 9,
  CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
  CUFFT_INVALID_DEVICE = 11,
  CUFFT_PARSE_ERROR = 12,
  CUFFT_NO_WORKSPACE = 13,
  CUFFT_NOT_IMPLEMENTED = 14,
  CUFFT_NOT_SUPPORTED = 16,
}

export enum CufftType {
  CUFFT_R2C = 0x2a, // Real to Complex (interleaved)
  CUFFT_C2R = 0x2c, // Complex (interleaved) to Real
  CUFFT_C2C = 0x29, // Complex to Complex
  CUFFT_D2Z = 0x6a, // Double to Double-Complex
  CUFFT_Z2D = 0x6c, // Double-Complex to Double
  CUFFT_Z2Z = 0x69, // Double-Complex to Double-Complex
}

export enum CufftDirection {
  CUFFT_FORWARD = -1,
  CUFFT_INVERSE = 1,
}

export interface CufftHandle {
  id: number;
  type: CufftType;
  nx: number;
  ny: number;
  nz: number;
  batch: number;
  direction: CufftDirection;
  pipeline: GPUComputePipeline | null;
  workBuffer: GPUBuffer | null;
}

// Cooley-Tukey FFT implementation in WGSL
const FFT_SHADER_1D = `
struct Complex {
    re: f32,
    im: f32,
}

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.re + b.re, a.im + b.im);
}

fn complex_sub(a: Complex, b: Complex) -> Complex {
    return Complex(a.re - b.re, a.im - b.im);
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // n, log2n, stage, direction

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.x;
    let stage = params.z;
    let direction = f32(i32(params.w)); // -1 for forward, 1 for inverse

    if (idx >= n / 2u) { return; }

    let butterflySize = 1u << (stage + 1u);
    let halfSize = butterflySize / 2u;

    let group = idx / halfSize;
    let pair = idx % halfSize;

    let i = group * butterflySize + pair;
    let j = i + halfSize;

    if (j >= n) { return; }

    // Load complex values (interleaved re, im)
    let a = Complex(data[2u * i], data[2u * i + 1u]);
    let b = Complex(data[2u * j], data[2u * j + 1u]);

    // Twiddle factor
    let angle = direction * 2.0 * PI * f32(pair) / f32(butterflySize);
    let twiddle = Complex(cos(angle), sin(angle));

    // Butterfly operation
    let t = complex_mul(b, twiddle);
    let out_a = complex_add(a, t);
    let out_b = complex_sub(a, t);

    // Store results
    data[2u * i] = out_a.re;
    data[2u * i + 1u] = out_a.im;
    data[2u * j] = out_b.re;
    data[2u * j + 1u] = out_b.im;
}
`;

const FFT_BIT_REVERSE_SHADER = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: vec2<u32>; // n, log2n

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var result: u32 = 0u;
    var val = x;
    for (var i: u32 = 0u; i < bits; i = i + 1u) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = params.x;
    let log2n = params.y;

    if (idx >= n) { return; }

    let rev = bit_reverse(idx, log2n);

    output[2u * idx] = input[2u * rev];
    output[2u * idx + 1u] = input[2u * rev + 1u];
}
`;

const FFT_SCALE_SHADER = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: vec2<f32>; // n, scale

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = u32(params.x);
    let scale = params.y;

    if (idx >= n * 2u) { return; }

    data[idx] = data[idx] * scale;
}
`;

const FFT_R2C_SHADER = `
// Real to Complex conversion
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }

    output[2u * idx] = input[idx];     // Real part
    output[2u * idx + 1u] = 0.0;       // Imaginary part = 0
}
`;

const FFT_C2R_SHADER = `
// Complex to Real conversion (takes real part)
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }

    output[idx] = input[2u * idx];     // Take real part only
}
`;

class CufftContext {
  private static instance: CufftContext | null = null;
  private plans: Map<number, CufftHandle> = new Map();
  private nextPlanId: number = 1;
  private shaderModules: Map<string, GPUShaderModule> = new Map();
  private pipelines: Map<string, GPUComputePipeline> = new Map();

  private constructor() {}

  static getInstance(): CufftContext {
    if (!CufftContext.instance) {
      CufftContext.instance = new CufftContext();
    }
    return CufftContext.instance;
  }

  private getDevice(): GPUDevice | null {
    return cudaDevice.getDevice();
  }

  private initShaders(device: GPUDevice): void {
    if (this.shaderModules.size > 0) return;

    this.shaderModules.set('fft1d', device.createShaderModule({ code: FFT_SHADER_1D }));
    this.shaderModules.set('bitReverse', device.createShaderModule({ code: FFT_BIT_REVERSE_SHADER }));
    this.shaderModules.set('scale', device.createShaderModule({ code: FFT_SCALE_SHADER }));
    this.shaderModules.set('r2c', device.createShaderModule({ code: FFT_R2C_SHADER }));
    this.shaderModules.set('c2r', device.createShaderModule({ code: FFT_C2R_SHADER }));
  }

  // Plan creation
  cufftPlan1d(
    nx: number,
    type: CufftType,
    batch: number = 1
  ): { plan: CufftHandle | null; result: CufftResult } {
    const device = this.getDevice();
    if (!device) {
      return { plan: null, result: CufftResult.CUFFT_INVALID_DEVICE };
    }

    // Check power of 2
    if ((nx & (nx - 1)) !== 0) {
      return { plan: null, result: CufftResult.CUFFT_INVALID_SIZE };
    }

    this.initShaders(device);

    const plan: CufftHandle = {
      id: this.nextPlanId++,
      type,
      nx,
      ny: 1,
      nz: 1,
      batch,
      direction: CufftDirection.CUFFT_FORWARD,
      pipeline: null,
      workBuffer: device.createBuffer({
        size: nx * 2 * 4 * batch, // Complex float32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    };

    this.plans.set(plan.id, plan);
    return { plan, result: CufftResult.CUFFT_SUCCESS };
  }

  cufftPlan2d(
    nx: number,
    ny: number,
    type: CufftType
  ): { plan: CufftHandle | null; result: CufftResult } {
    const device = this.getDevice();
    if (!device) {
      return { plan: null, result: CufftResult.CUFFT_INVALID_DEVICE };
    }

    // Check power of 2
    if ((nx & (nx - 1)) !== 0 || (ny & (ny - 1)) !== 0) {
      return { plan: null, result: CufftResult.CUFFT_INVALID_SIZE };
    }

    this.initShaders(device);

    const plan: CufftHandle = {
      id: this.nextPlanId++,
      type,
      nx,
      ny,
      nz: 1,
      batch: 1,
      direction: CufftDirection.CUFFT_FORWARD,
      pipeline: null,
      workBuffer: device.createBuffer({
        size: nx * ny * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    };

    this.plans.set(plan.id, plan);
    return { plan, result: CufftResult.CUFFT_SUCCESS };
  }

  cufftPlan3d(
    nx: number,
    ny: number,
    nz: number,
    type: CufftType
  ): { plan: CufftHandle | null; result: CufftResult } {
    const device = this.getDevice();
    if (!device) {
      return { plan: null, result: CufftResult.CUFFT_INVALID_DEVICE };
    }

    if ((nx & (nx - 1)) !== 0 || (ny & (ny - 1)) !== 0 || (nz & (nz - 1)) !== 0) {
      return { plan: null, result: CufftResult.CUFFT_INVALID_SIZE };
    }

    this.initShaders(device);

    const plan: CufftHandle = {
      id: this.nextPlanId++,
      type,
      nx,
      ny,
      nz,
      batch: 1,
      direction: CufftDirection.CUFFT_FORWARD,
      pipeline: null,
      workBuffer: device.createBuffer({
        size: nx * ny * nz * 2 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
    };

    this.plans.set(plan.id, plan);
    return { plan, result: CufftResult.CUFFT_SUCCESS };
  }

  cufftDestroy(plan: CufftHandle): CufftResult {
    const p = this.plans.get(plan.id);
    if (!p) return CufftResult.CUFFT_INVALID_PLAN;

    if (p.workBuffer) p.workBuffer.destroy();
    this.plans.delete(plan.id);
    return CufftResult.CUFFT_SUCCESS;
  }

  // Execution
  async cufftExecC2C(
    plan: CufftHandle,
    idata: CudaPointer,
    odata: CudaPointer,
    direction: CufftDirection
  ): Promise<CufftResult> {
    const device = this.getDevice();
    if (!device) return CufftResult.CUFFT_INVALID_DEVICE;

    const p = this.plans.get(plan.id);
    if (!p) return CufftResult.CUFFT_INVALID_PLAN;

    const inputBuffer = getBuffer(idata);
    const outputBuffer = getBuffer(odata);
    if (!inputBuffer || !outputBuffer || !p.workBuffer) {
      return CufftResult.CUFFT_INVALID_VALUE;
    }

    const n = p.nx;
    const log2n = Math.log2(n);

    // Step 1: Bit-reverse permutation
    await this.bitReverse(device, inputBuffer, p.workBuffer, n, log2n);

    // Step 2: FFT butterfly stages
    for (let stage = 0; stage < log2n; stage++) {
      await this.fftStage(device, p.workBuffer, n, stage, direction);
    }

    // Step 3: Copy to output (and scale for inverse)
    if (direction === CufftDirection.CUFFT_INVERSE) {
      await this.scale(device, p.workBuffer, n, 1 / n);
    }

    // Copy work buffer to output
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(p.workBuffer, 0, outputBuffer, 0, n * 2 * 4);
    device.queue.submit([encoder.finish()]);

    return CufftResult.CUFFT_SUCCESS;
  }

  async cufftExecR2C(
    plan: CufftHandle,
    idata: CudaPointer,
    odata: CudaPointer
  ): Promise<CufftResult> {
    const device = this.getDevice();
    if (!device) return CufftResult.CUFFT_INVALID_DEVICE;

    const p = this.plans.get(plan.id);
    if (!p) return CufftResult.CUFFT_INVALID_PLAN;

    const inputBuffer = getBuffer(idata);
    const outputBuffer = getBuffer(odata);
    if (!inputBuffer || !outputBuffer || !p.workBuffer) {
      return CufftResult.CUFFT_INVALID_VALUE;
    }

    const n = p.nx;
    const log2n = Math.log2(n);

    // Step 1: Convert real to complex
    await this.realToComplex(device, inputBuffer, p.workBuffer, n);

    // Step 2: Bit-reverse
    const tempBuffer = device.createBuffer({
      size: n * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    await this.bitReverse(device, p.workBuffer, tempBuffer, n, log2n);

    // Copy back
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(tempBuffer, 0, p.workBuffer, 0, n * 2 * 4);
    device.queue.submit([copyEncoder.finish()]);

    // Step 3: FFT stages
    for (let stage = 0; stage < log2n; stage++) {
      await this.fftStage(device, p.workBuffer, n, stage, CufftDirection.CUFFT_FORWARD);
    }

    // Copy to output
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(p.workBuffer, 0, outputBuffer, 0, n * 2 * 4);
    device.queue.submit([encoder.finish()]);

    tempBuffer.destroy();
    return CufftResult.CUFFT_SUCCESS;
  }

  async cufftExecC2R(
    plan: CufftHandle,
    idata: CudaPointer,
    odata: CudaPointer
  ): Promise<CufftResult> {
    const device = this.getDevice();
    if (!device) return CufftResult.CUFFT_INVALID_DEVICE;

    const p = this.plans.get(plan.id);
    if (!p) return CufftResult.CUFFT_INVALID_PLAN;

    const inputBuffer = getBuffer(idata);
    const outputBuffer = getBuffer(odata);
    if (!inputBuffer || !outputBuffer || !p.workBuffer) {
      return CufftResult.CUFFT_INVALID_VALUE;
    }

    const n = p.nx;
    const log2n = Math.log2(n);

    // Step 1: Bit-reverse
    await this.bitReverse(device, inputBuffer, p.workBuffer, n, log2n);

    // Step 2: IFFT stages
    for (let stage = 0; stage < log2n; stage++) {
      await this.fftStage(device, p.workBuffer, n, stage, CufftDirection.CUFFT_INVERSE);
    }

    // Step 3: Scale
    await this.scale(device, p.workBuffer, n, 1 / n);

    // Step 4: Convert complex to real
    await this.complexToReal(device, p.workBuffer, outputBuffer, n);

    return CufftResult.CUFFT_SUCCESS;
  }

  // Helper methods
  private async bitReverse(
    device: GPUDevice,
    input: GPUBuffer,
    output: GPUBuffer,
    n: number,
    log2n: number
  ): Promise<void> {
    const module = this.shaderModules.get('bitReverse')!;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });

    const paramsBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([n, log2n]));

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();
  }

  private async fftStage(
    device: GPUDevice,
    data: GPUBuffer,
    n: number,
    stage: number,
    direction: CufftDirection
  ): Promise<void> {
    const module = this.shaderModules.get('fft1d')!;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });

    const directionValue = direction === CufftDirection.CUFFT_FORWARD ? -1 : 1;
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
      paramsBuffer,
      0,
      new Uint32Array([n, Math.log2(n), stage, directionValue >>> 0])
    );

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: data } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 2 / 256));
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();
  }

  private async scale(
    device: GPUDevice,
    data: GPUBuffer,
    n: number,
    scale: number
  ): Promise<void> {
    const module = this.shaderModules.get('scale')!;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });

    const paramsBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([n, scale]));

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: data } },
        { binding: 1, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n * 2 / 256));
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();
  }

  private async realToComplex(
    device: GPUDevice,
    input: GPUBuffer,
    output: GPUBuffer,
    n: number
  ): Promise<void> {
    const module = this.shaderModules.get('r2c')!;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });

    const paramsBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([n]));

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();
  }

  private async complexToReal(
    device: GPUDevice,
    input: GPUBuffer,
    output: GPUBuffer,
    n: number
  ): Promise<void> {
    const module = this.shaderModules.get('c2r')!;

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });

    const paramsBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([n]));

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    paramsBuffer.destroy();
  }
}

// Export singleton
export const cufftContext = CufftContext.getInstance();

// CUDA-like API exports
export function cufftPlan1d(
  nx: number,
  type: CufftType,
  batch?: number
): { plan: CufftHandle | null; result: CufftResult } {
  return cufftContext.cufftPlan1d(nx, type, batch);
}

export function cufftPlan2d(
  nx: number,
  ny: number,
  type: CufftType
): { plan: CufftHandle | null; result: CufftResult } {
  return cufftContext.cufftPlan2d(nx, ny, type);
}

export function cufftPlan3d(
  nx: number,
  ny: number,
  nz: number,
  type: CufftType
): { plan: CufftHandle | null; result: CufftResult } {
  return cufftContext.cufftPlan3d(nx, ny, nz, type);
}

export function cufftDestroy(plan: CufftHandle): CufftResult {
  return cufftContext.cufftDestroy(plan);
}

export function cufftExecC2C(
  plan: CufftHandle,
  idata: CudaPointer,
  odata: CudaPointer,
  direction: CufftDirection
): Promise<CufftResult> {
  return cufftContext.cufftExecC2C(plan, idata, odata, direction);
}

export function cufftExecR2C(
  plan: CufftHandle,
  idata: CudaPointer,
  odata: CudaPointer
): Promise<CufftResult> {
  return cufftContext.cufftExecR2C(plan, idata, odata);
}

export function cufftExecC2R(
  plan: CufftHandle,
  idata: CudaPointer,
  odata: CudaPointer
): Promise<CufftResult> {
  return cufftContext.cufftExecC2R(plan, idata, odata);
}
