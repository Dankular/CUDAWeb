/**
 * cuRAND-like Random Number Generation for WebGPU
 *
 * Provides CUDA cuRAND-compatible random number generation
 */

import { CudaError, cudaDevice } from '../runtime/cuda-device';
import { CudaPointer, getBuffer } from '../runtime/cuda-memory';
import { CudaStream } from '../runtime/cuda-stream';

export enum CurandStatus {
  CURAND_STATUS_SUCCESS = 0,
  CURAND_STATUS_VERSION_MISMATCH = 100,
  CURAND_STATUS_NOT_INITIALIZED = 101,
  CURAND_STATUS_ALLOCATION_FAILED = 102,
  CURAND_STATUS_TYPE_ERROR = 103,
  CURAND_STATUS_OUT_OF_RANGE = 104,
  CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
  CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
  CURAND_STATUS_LAUNCH_FAILURE = 201,
  CURAND_STATUS_PREEXISTING_FAILURE = 202,
  CURAND_STATUS_INITIALIZATION_FAILED = 203,
  CURAND_STATUS_ARCH_MISMATCH = 204,
  CURAND_STATUS_INTERNAL_ERROR = 999,
}

export enum CurandRngType {
  CURAND_RNG_TEST = 0,
  CURAND_RNG_PSEUDO_DEFAULT = 100,
  CURAND_RNG_PSEUDO_XORWOW = 101,
  CURAND_RNG_PSEUDO_MRG32K3A = 121,
  CURAND_RNG_PSEUDO_MTGP32 = 141,
  CURAND_RNG_PSEUDO_MT19937 = 142,
  CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
  CURAND_RNG_QUASI_DEFAULT = 200,
  CURAND_RNG_QUASI_SOBOL32 = 201,
  CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
  CURAND_RNG_QUASI_SOBOL64 = 203,
  CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204,
}

export enum CurandOrdering {
  CURAND_ORDERING_PSEUDO_BEST = 100,
  CURAND_ORDERING_PSEUDO_DEFAULT = 101,
  CURAND_ORDERING_PSEUDO_SEEDED = 102,
  CURAND_ORDERING_QUASI_DEFAULT = 201,
}

export interface CurandGenerator {
  id: number;
  type: CurandRngType;
  seed: bigint;
  offset: bigint;
  stream: CudaStream | null;
  stateBuffer: GPUBuffer | null;
}

// Philox random number generator in WGSL
const CURAND_PHILOX_WGSL = `
// Philox 4x32-10 RNG Implementation

fn mulhilo32(a: u32, b: u32) -> vec2<u32> {
    let product = u64(a) * u64(b);
    return vec2<u32>(u32(product), u32(product >> 32u));
}

fn philox_round(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    let PHILOX_M4x32_0: u32 = 0xD2511F53u;
    let PHILOX_M4x32_1: u32 = 0xCD9E8D57u;

    let lo0 = mulhilo32(PHILOX_M4x32_0, ctr.x);
    let lo1 = mulhilo32(PHILOX_M4x32_1, ctr.z);

    return vec4<u32>(
        lo1.y ^ ctr.y ^ key.x,
        lo1.x,
        lo0.y ^ ctr.w ^ key.y,
        lo0.x
    );
}

fn philox_bumpkey(key: vec2<u32>) -> vec2<u32> {
    let PHILOX_W32_0: u32 = 0x9E3779B9u;
    let PHILOX_W32_1: u32 = 0xBB67AE85u;
    return vec2<u32>(key.x + PHILOX_W32_0, key.y + PHILOX_W32_1);
}

fn philox4x32_10(ctr: vec4<u32>, key: vec2<u32>) -> vec4<u32> {
    var c = ctr;
    var k = key;

    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k); k = philox_bumpkey(k);
    c = philox_round(c, k);

    return c;
}

// Convert u32 to uniform float in [0, 1)
fn u32_to_uniform(x: u32) -> f32 {
    return f32(x) * 2.3283064365386963e-10f; // 1.0 / 2^32
}

// Box-Muller transform for normal distribution
fn box_muller(u1: f32, u2: f32) -> vec2<f32> {
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307179586 * u2;
    return vec2<f32>(r * cos(theta), r * sin(theta));
}
`;

const CURAND_UNIFORM_SHADER = `
${CURAND_PHILOX_WGSL}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>; // seed_lo, seed_hi, offset_lo, offset_hi
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }

    let seed = vec2<u32>(params.x, params.y);
    let offset = vec2<u32>(params.z, params.w);

    // Counter includes index and offset
    let counter = vec4<u32>(idx + offset.x, offset.y, 0u, 0u);
    let random = philox4x32_10(counter, seed);

    output[idx] = u32_to_uniform(random.x);
}
`;

const CURAND_UNIFORM_DOUBLE_SHADER = `
${CURAND_PHILOX_WGSL}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }

    let seed = vec2<u32>(params.x, params.y);
    let offset = vec2<u32>(params.z, params.w);

    let counter = vec4<u32>(idx + offset.x, offset.y, 0u, 0u);
    let random = philox4x32_10(counter, seed);

    // Use two 32-bit values for higher precision
    // WebGPU doesn't support f64, so we still output f32
    let combined = (f32(random.x) + f32(random.y) * 2.3283064365386963e-10) * 2.3283064365386963e-10;
    output[idx] = combined;
}
`;

const CURAND_NORMAL_SHADER = `
${CURAND_PHILOX_WGSL}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n / 2u) { return; }

    let seed = vec2<u32>(params.x, params.y);
    let offset = vec2<u32>(params.z, params.w);

    let counter = vec4<u32>(idx + offset.x, offset.y, 0u, 0u);
    let random = philox4x32_10(counter, seed);

    let u1 = max(u32_to_uniform(random.x), 1e-10f);
    let u2 = u32_to_uniform(random.y);
    let normals = box_muller(u1, u2);

    output[2u * idx] = normals.x;
    if (2u * idx + 1u < n) {
        output[2u * idx + 1u] = normals.y;
    }
}
`;

const CURAND_LOG_NORMAL_SHADER = `
${CURAND_PHILOX_WGSL}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
@group(0) @binding(2) var<uniform> dist_params: vec2<f32>; // mean, stddev
@group(0) @binding(3) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n / 2u) { return; }

    let seed = vec2<u32>(params.x, params.y);
    let offset = vec2<u32>(params.z, params.w);
    let mean = dist_params.x;
    let stddev = dist_params.y;

    let counter = vec4<u32>(idx + offset.x, offset.y, 0u, 0u);
    let random = philox4x32_10(counter, seed);

    let u1 = max(u32_to_uniform(random.x), 1e-10f);
    let u2 = u32_to_uniform(random.y);
    let normals = box_muller(u1, u2);

    output[2u * idx] = exp(mean + stddev * normals.x);
    if (2u * idx + 1u < n) {
        output[2u * idx + 1u] = exp(mean + stddev * normals.y);
    }
}
`;

const CURAND_POISSON_SHADER = `
${CURAND_PHILOX_WGSL}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
@group(0) @binding(2) var<uniform> lambda: f32;
@group(0) @binding(3) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }

    let seed = vec2<u32>(params.x, params.y);
    let offset = vec2<u32>(params.z, params.w);

    // Knuth algorithm for Poisson
    let L = exp(-lambda);
    var k: u32 = 0u;
    var p: f32 = 1.0;
    var counter_offset: u32 = 0u;

    loop {
        k = k + 1u;
        let counter = vec4<u32>(idx + offset.x, offset.y + counter_offset, 0u, 0u);
        let random = philox4x32_10(counter, seed);
        p = p * u32_to_uniform(random.x);
        counter_offset = counter_offset + 1u;

        if (p <= L || k > 1000u) {
            break;
        }
    }

    output[idx] = k - 1u;
}
`;

const CURAND_UINT_SHADER = `
${CURAND_PHILOX_WGSL}

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }

    let seed = vec2<u32>(params.x, params.y);
    let offset = vec2<u32>(params.z, params.w);

    let counter = vec4<u32>(idx + offset.x, offset.y, 0u, 0u);
    let random = philox4x32_10(counter, seed);

    output[idx] = random.x;
}
`;

class CurandContext {
  private static instance: CurandContext | null = null;
  private generators: Map<number, CurandGenerator> = new Map();
  private nextGenId: number = 1;
  private shaderModules: Map<string, GPUShaderModule> = new Map();

  private constructor() {}

  static getInstance(): CurandContext {
    if (!CurandContext.instance) {
      CurandContext.instance = new CurandContext();
    }
    return CurandContext.instance;
  }

  private getDevice(): GPUDevice | null {
    return cudaDevice.getDevice();
  }

  private initShaders(device: GPUDevice): void {
    if (this.shaderModules.size > 0) return;

    this.shaderModules.set('uniform', device.createShaderModule({ code: CURAND_UNIFORM_SHADER }));
    this.shaderModules.set('uniformDouble', device.createShaderModule({ code: CURAND_UNIFORM_DOUBLE_SHADER }));
    this.shaderModules.set('normal', device.createShaderModule({ code: CURAND_NORMAL_SHADER }));
    this.shaderModules.set('logNormal', device.createShaderModule({ code: CURAND_LOG_NORMAL_SHADER }));
    this.shaderModules.set('poisson', device.createShaderModule({ code: CURAND_POISSON_SHADER }));
    this.shaderModules.set('uint', device.createShaderModule({ code: CURAND_UINT_SHADER }));
  }

  curandCreateGenerator(type: CurandRngType): {
    generator: CurandGenerator | null;
    status: CurandStatus;
  } {
    const device = this.getDevice();
    if (!device) {
      return { generator: null, status: CurandStatus.CURAND_STATUS_NOT_INITIALIZED };
    }

    this.initShaders(device);

    const generator: CurandGenerator = {
      id: this.nextGenId++,
      type,
      seed: BigInt(0),
      offset: BigInt(0),
      stream: null,
      stateBuffer: null,
    };

    this.generators.set(generator.id, generator);
    return { generator, status: CurandStatus.CURAND_STATUS_SUCCESS };
  }

  curandDestroyGenerator(generator: CurandGenerator): CurandStatus {
    const gen = this.generators.get(generator.id);
    if (!gen) return CurandStatus.CURAND_STATUS_NOT_INITIALIZED;

    if (gen.stateBuffer) gen.stateBuffer.destroy();
    this.generators.delete(generator.id);
    return CurandStatus.CURAND_STATUS_SUCCESS;
  }

  curandSetPseudoRandomGeneratorSeed(
    generator: CurandGenerator,
    seed: bigint
  ): CurandStatus {
    const gen = this.generators.get(generator.id);
    if (!gen) return CurandStatus.CURAND_STATUS_NOT_INITIALIZED;

    gen.seed = seed;
    return CurandStatus.CURAND_STATUS_SUCCESS;
  }

  curandSetGeneratorOffset(
    generator: CurandGenerator,
    offset: bigint
  ): CurandStatus {
    const gen = this.generators.get(generator.id);
    if (!gen) return CurandStatus.CURAND_STATUS_NOT_INITIALIZED;

    gen.offset = offset;
    return CurandStatus.CURAND_STATUS_SUCCESS;
  }

  curandSetStream(
    generator: CurandGenerator,
    stream: CudaStream
  ): CurandStatus {
    const gen = this.generators.get(generator.id);
    if (!gen) return CurandStatus.CURAND_STATUS_NOT_INITIALIZED;

    gen.stream = stream;
    return CurandStatus.CURAND_STATUS_SUCCESS;
  }

  async curandGenerateUniform(
    generator: CurandGenerator,
    outputPtr: CudaPointer,
    num: number
  ): Promise<CurandStatus> {
    return this.generateRandom(generator, outputPtr, num, 'uniform');
  }

  async curandGenerateUniformDouble(
    generator: CurandGenerator,
    outputPtr: CudaPointer,
    num: number
  ): Promise<CurandStatus> {
    return this.generateRandom(generator, outputPtr, num, 'uniformDouble');
  }

  async curandGenerateNormal(
    generator: CurandGenerator,
    outputPtr: CudaPointer,
    num: number,
    mean: number = 0,
    stddev: number = 1
  ): Promise<CurandStatus> {
    return this.generateRandom(generator, outputPtr, num, 'normal', { mean, stddev });
  }

  async curandGenerateLogNormal(
    generator: CurandGenerator,
    outputPtr: CudaPointer,
    num: number,
    mean: number,
    stddev: number
  ): Promise<CurandStatus> {
    return this.generateRandom(generator, outputPtr, num, 'logNormal', { mean, stddev });
  }

  async curandGeneratePoisson(
    generator: CurandGenerator,
    outputPtr: CudaPointer,
    num: number,
    lambda: number
  ): Promise<CurandStatus> {
    return this.generateRandom(generator, outputPtr, num, 'poisson', { lambda });
  }

  async curandGenerate(
    generator: CurandGenerator,
    outputPtr: CudaPointer,
    num: number
  ): Promise<CurandStatus> {
    return this.generateRandom(generator, outputPtr, num, 'uint');
  }

  private async generateRandom(
    generator: CurandGenerator,
    outputPtr: CudaPointer,
    num: number,
    type: string,
    params?: { mean?: number; stddev?: number; lambda?: number }
  ): Promise<CurandStatus> {
    const device = this.getDevice();
    if (!device) return CurandStatus.CURAND_STATUS_NOT_INITIALIZED;

    const gen = this.generators.get(generator.id);
    if (!gen) return CurandStatus.CURAND_STATUS_NOT_INITIALIZED;

    const outputBuffer = getBuffer(outputPtr);
    if (!outputBuffer) return CurandStatus.CURAND_STATUS_ALLOCATION_FAILED;

    const module = this.shaderModules.get(type);
    if (!module) return CurandStatus.CURAND_STATUS_TYPE_ERROR;

    // Create params buffer
    const seedLo = Number(gen.seed & BigInt(0xFFFFFFFF));
    const seedHi = Number((gen.seed >> BigInt(32)) & BigInt(0xFFFFFFFF));
    const offsetLo = Number(gen.offset & BigInt(0xFFFFFFFF));
    const offsetHi = Number((gen.offset >> BigInt(32)) & BigInt(0xFFFFFFFF));

    const paramsData = new Uint32Array([seedLo, seedHi, offsetLo, offsetHi]);
    const paramsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);

    const nBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(nBuffer, 0, new Uint32Array([num]));

    // Build bind group layout based on type
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ];

    let distParamsBuffer: GPUBuffer | null = null;
    if (type === 'logNormal') {
      entries.push({ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
      distParamsBuffer = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(distParamsBuffer, 0, new Float32Array([params?.mean || 0, params?.stddev || 1]));
    } else if (type === 'poisson') {
      entries.push({ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
      distParamsBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(distParamsBuffer, 0, new Float32Array([params?.lambda || 1]));
    }

    const bindGroupLayout = device.createBindGroupLayout({ entries });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      compute: { module, entryPoint: 'main' },
    });

    const bindGroupEntries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: outputBuffer } },
      { binding: 1, resource: { buffer: paramsBuffer } },
      { binding: 2, resource: { buffer: nBuffer } },
    ];

    if (distParamsBuffer) {
      bindGroupEntries.push({ binding: 3, resource: { buffer: distParamsBuffer } });
    }

    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: bindGroupEntries,
    });

    const workgroups = type === 'normal' || type === 'logNormal'
      ? Math.ceil(num / 2 / 256)
      : Math.ceil(num / 256);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Update offset for next call
    gen.offset += BigInt(num);

    // Cleanup
    paramsBuffer.destroy();
    nBuffer.destroy();
    if (distParamsBuffer) distParamsBuffer.destroy();

    return CurandStatus.CURAND_STATUS_SUCCESS;
  }
}

// Export singleton
export const curandContext = CurandContext.getInstance();

// CUDA-like API exports
export function curandCreateGenerator(type: CurandRngType): {
  generator: CurandGenerator | null;
  status: CurandStatus;
} {
  return curandContext.curandCreateGenerator(type);
}

export function curandDestroyGenerator(generator: CurandGenerator): CurandStatus {
  return curandContext.curandDestroyGenerator(generator);
}

export function curandSetPseudoRandomGeneratorSeed(
  generator: CurandGenerator,
  seed: bigint
): CurandStatus {
  return curandContext.curandSetPseudoRandomGeneratorSeed(generator, seed);
}

export function curandSetGeneratorOffset(
  generator: CurandGenerator,
  offset: bigint
): CurandStatus {
  return curandContext.curandSetGeneratorOffset(generator, offset);
}

export function curandSetStream(
  generator: CurandGenerator,
  stream: CudaStream
): CurandStatus {
  return curandContext.curandSetStream(generator, stream);
}

export function curandGenerateUniform(
  generator: CurandGenerator,
  outputPtr: CudaPointer,
  num: number
): Promise<CurandStatus> {
  return curandContext.curandGenerateUniform(generator, outputPtr, num);
}

export function curandGenerateUniformDouble(
  generator: CurandGenerator,
  outputPtr: CudaPointer,
  num: number
): Promise<CurandStatus> {
  return curandContext.curandGenerateUniformDouble(generator, outputPtr, num);
}

export function curandGenerateNormal(
  generator: CurandGenerator,
  outputPtr: CudaPointer,
  num: number,
  mean?: number,
  stddev?: number
): Promise<CurandStatus> {
  return curandContext.curandGenerateNormal(generator, outputPtr, num, mean, stddev);
}

export function curandGenerateLogNormal(
  generator: CurandGenerator,
  outputPtr: CudaPointer,
  num: number,
  mean: number,
  stddev: number
): Promise<CurandStatus> {
  return curandContext.curandGenerateLogNormal(generator, outputPtr, num, mean, stddev);
}

export function curandGeneratePoisson(
  generator: CurandGenerator,
  outputPtr: CudaPointer,
  num: number,
  lambda: number
): Promise<CurandStatus> {
  return curandContext.curandGeneratePoisson(generator, outputPtr, num, lambda);
}

export function curandGenerate(
  generator: CurandGenerator,
  outputPtr: CudaPointer,
  num: number
): Promise<CurandStatus> {
  return curandContext.curandGenerate(generator, outputPtr, num);
}

// Export WGSL code for inclusion in custom shaders
export const CURAND_WGSL = CURAND_PHILOX_WGSL;

export function withCurand(shaderCode: string): string {
  return CURAND_PHILOX_WGSL + '\n\n' + shaderCode;
}
