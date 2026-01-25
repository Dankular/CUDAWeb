/**
 * CUDA to WGSL Transpiler Utilities
 *
 * Helps convert CUDA patterns to WebGPU/WGSL equivalents
 */

// CUDA to WGSL type mappings
export const TYPE_MAPPINGS: Record<string, string> = {
  // Scalar types
  'float': 'f32',
  'double': 'f32', // WebGPU doesn't have f64 in most implementations
  'int': 'i32',
  'unsigned int': 'u32',
  'uint': 'u32',
  'long': 'i32',
  'unsigned long': 'u32',
  'short': 'i32',
  'unsigned short': 'u32',
  'char': 'i32',
  'unsigned char': 'u32',
  'bool': 'bool',
  'size_t': 'u32',

  // Vector types
  'float2': 'vec2<f32>',
  'float3': 'vec3<f32>',
  'float4': 'vec4<f32>',
  'double2': 'vec2<f32>',
  'double3': 'vec3<f32>',
  'double4': 'vec4<f32>',
  'int2': 'vec2<i32>',
  'int3': 'vec3<i32>',
  'int4': 'vec4<i32>',
  'uint2': 'vec2<u32>',
  'uint3': 'vec3<u32>',
  'uint4': 'vec4<u32>',

  // Special types
  'dim3': 'vec3<u32>',
  'cudaStream_t': 'u32',
  'cudaEvent_t': 'u32',
};

// CUDA built-in variable mappings
export const BUILTIN_MAPPINGS: Record<string, string> = {
  'threadIdx.x': 'local_invocation_id.x',
  'threadIdx.y': 'local_invocation_id.y',
  'threadIdx.z': 'local_invocation_id.z',
  'blockIdx.x': 'workgroup_id.x',
  'blockIdx.y': 'workgroup_id.y',
  'blockIdx.z': 'workgroup_id.z',
  'blockDim.x': 'workgroup_size.x',
  'blockDim.y': 'workgroup_size.y',
  'blockDim.z': 'workgroup_size.z',
  'gridDim.x': 'num_workgroups.x',
  'gridDim.y': 'num_workgroups.y',
  'gridDim.z': 'num_workgroups.z',
  'warpSize': '32u',
};

// CUDA function to WGSL function mappings
export const FUNCTION_MAPPINGS: Record<string, string> = {
  // Math functions
  '__sinf': 'sin',
  '__cosf': 'cos',
  '__tanf': 'tan',
  'sinf': 'sin',
  'cosf': 'cos',
  'tanf': 'tan',
  'asinf': 'asin',
  'acosf': 'acos',
  'atanf': 'atan',
  'atan2f': 'atan2',
  'sinhf': 'sinh',
  'coshf': 'cosh',
  'tanhf': 'tanh',
  'asinhf': 'asinh',
  'acoshf': 'acosh',
  'atanhf': 'atanh',
  '__expf': 'exp',
  'expf': 'exp',
  'exp2f': 'exp2',
  '__logf': 'log',
  'logf': 'log',
  'log2f': 'log2',
  '__powf': 'pow',
  'powf': 'pow',
  'sqrtf': 'sqrt',
  '__sqrtf': 'sqrt',
  'rsqrtf': 'inverseSqrt',
  '__rsqrtf': 'inverseSqrt',
  'fabsf': 'abs',
  'fabs': 'abs',
  'floorf': 'floor',
  'ceilf': 'ceil',
  'truncf': 'trunc',
  'roundf': 'round',
  'fminf': 'min',
  'fmaxf': 'max',
  '__fmaf_rn': 'fma',
  'fmaf': 'fma',
  'copysignf': 'sign', // Not exact, needs adjustment

  // Vector construction
  'make_float2': 'vec2<f32>',
  'make_float3': 'vec3<f32>',
  'make_float4': 'vec4<f32>',
  'make_int2': 'vec2<i32>',
  'make_int3': 'vec3<i32>',
  'make_int4': 'vec4<i32>',
  'make_uint2': 'vec2<u32>',
  'make_uint3': 'vec3<u32>',
  'make_uint4': 'vec4<u32>',

  // Synchronization
  '__syncthreads': 'workgroupBarrier',
  '__threadfence_block': 'workgroupBarrier',
  '__threadfence': 'storageBarrier',

  // Atomic operations
  'atomicAdd': 'atomicAdd',
  'atomicSub': 'atomicSub',
  'atomicExch': 'atomicExchange',
  'atomicMin': 'atomicMin',
  'atomicMax': 'atomicMax',
  'atomicAnd': 'atomicAnd',
  'atomicOr': 'atomicOr',
  'atomicXor': 'atomicXor',
  'atomicCAS': 'atomicCompareExchangeWeak',

  // Integer bit operations
  '__clz': 'countLeadingZeros',
  '__ffs': 'firstLeadingBit',
  '__popc': 'countOneBits',
  '__brev': 'reverseBits',
};

// Memory space mappings
export const MEMORY_SPACE_MAPPINGS: Record<string, string> = {
  '__shared__': 'var<workgroup>',
  '__device__': 'var<private>', // Or storage depending on context
  '__constant__': 'var<uniform>',
  '__global__': '@compute @workgroup_size', // Kernel declaration
};

// Common CUDA patterns and their WGSL equivalents
export const PATTERN_EXAMPLES: Record<string, { cuda: string; wgsl: string }> = {
  kernelDeclaration: {
    cuda: `__global__ void myKernel(float* input, float* output, int n)`,
    wgsl: `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= n) { return; }
    // ...
}`,
  },

  sharedMemory: {
    cuda: `__shared__ float sharedData[256];`,
    wgsl: `var<workgroup> sharedData: array<f32, 256>;`,
  },

  threadIndexing: {
    cuda: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`,
    wgsl: `let idx = workgroup_id.x * workgroup_size.x + local_invocation_id.x;`,
  },

  atomicAddFloat: {
    cuda: `atomicAdd(&sum, value);`,
    wgsl: `
// Float atomics need CAS loop in WGSL
var old_bits: u32;
loop {
    old_bits = atomicLoad(&sum_bits);
    let old_val = bitcast<f32>(old_bits);
    let new_val = old_val + value;
    let new_bits = bitcast<u32>(new_val);
    let result = atomicCompareExchangeWeak(&sum_bits, old_bits, new_bits);
    if (result.exchanged) { break; }
}`,
  },

  warpReduce: {
    cuda: `
for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}`,
    wgsl: `
var<workgroup> warp_data: array<f32, 32>;
let lane = local_invocation_index % 32u;
let warp = local_invocation_index / 32u;
warp_data[local_invocation_index] = val;
workgroupBarrier();
for (var offset: u32 = 16u; offset > 0u; offset = offset / 2u) {
    if (lane < offset) {
        warp_data[warp * 32u + lane] += warp_data[warp * 32u + lane + offset];
    }
    workgroupBarrier();
}
val = warp_data[warp * 32u];`,
  },

  blockReduce: {
    cuda: `
__shared__ float sdata[256];
sdata[threadIdx.x] = myVal;
__syncthreads();
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
}`,
    wgsl: `
var<workgroup> sdata: array<f32, 256>;
sdata[local_invocation_id.x] = myVal;
workgroupBarrier();
for (var s: u32 = workgroup_size.x / 2u; s > 0u; s = s / 2u) {
    if (local_invocation_id.x < s) {
        sdata[local_invocation_id.x] += sdata[local_invocation_id.x + s];
    }
    workgroupBarrier();
}`,
  },

  textureRead: {
    cuda: `float4 color = tex2D(texture, u, v);`,
    wgsl: `let color = textureSample(t, s, vec2<f32>(u, v));`,
  },

  conditionalAssign: {
    cuda: `float result = condition ? a : b;`,
    wgsl: `let result = select(b, a, condition);`,
  },
};

/**
 * Simple CUDA to WGSL transpiler
 * Note: This is a basic pattern-based converter, not a full transpiler
 */
export function transpileCudaToWgsl(cudaCode: string): string {
  let wgslCode = cudaCode;

  // Replace types
  for (const [cudaType, wgslType] of Object.entries(TYPE_MAPPINGS)) {
    const regex = new RegExp(`\\b${cudaType}\\b`, 'g');
    wgslCode = wgslCode.replace(regex, wgslType);
  }

  // Replace built-in variables
  for (const [cudaVar, wgslVar] of Object.entries(BUILTIN_MAPPINGS)) {
    const escaped = cudaVar.replace(/\./g, '\\.');
    const regex = new RegExp(escaped, 'g');
    wgslCode = wgslCode.replace(regex, wgslVar);
  }

  // Replace functions
  for (const [cudaFunc, wgslFunc] of Object.entries(FUNCTION_MAPPINGS)) {
    const regex = new RegExp(`\\b${cudaFunc}\\b`, 'g');
    wgslCode = wgslCode.replace(regex, wgslFunc);
  }

  // Replace memory spaces
  wgslCode = wgslCode.replace(/__shared__\s+/g, 'var<workgroup> ');
  wgslCode = wgslCode.replace(/__device__\s+/g, '');
  wgslCode = wgslCode.replace(/__constant__\s+/g, 'var<uniform> ');

  // Replace common patterns
  wgslCode = wgslCode.replace(/\?.*?:.*?;/g, (match) => {
    // Ternary to select (simplified)
    return match.replace('?', '// TODO: Use select(');
  });

  // Replace pointer syntax
  wgslCode = wgslCode.replace(/\*(\w+)/g, '$1');
  wgslCode = wgslCode.replace(/&(\w+)/g, '&$1');

  // Replace array access
  wgslCode = wgslCode.replace(/\[(\d+)\]/g, '[$1u]');

  // Add note about manual review needed
  wgslCode = `// Auto-transpiled from CUDA - Manual review recommended\n// Some patterns may need adjustment\n\n${wgslCode}`;

  return wgslCode;
}

/**
 * Generate WGSL kernel template from CUDA-like specification
 */
export function generateKernelTemplate(config: {
  name: string;
  inputs: Array<{ name: string; type: string; access: 'read' | 'write' | 'read_write' }>;
  uniforms?: Array<{ name: string; type: string }>;
  workgroupSize: [number, number, number];
  sharedMemory?: Array<{ name: string; type: string; size: number }>;
}): string {
  let wgsl = '';

  // Bindings
  let binding = 0;
  for (const input of config.inputs) {
    const accessType = input.access === 'read' ? 'read' :
                       input.access === 'write' ? 'read_write' : 'read_write';
    const storageType = input.access === 'read' ? 'read' : 'read_write';
    wgsl += `@group(0) @binding(${binding}) var<storage, ${storageType}> ${input.name}: ${input.type};\n`;
    binding++;
  }

  if (config.uniforms) {
    for (const uniform of config.uniforms) {
      wgsl += `@group(0) @binding(${binding}) var<uniform> ${uniform.name}: ${uniform.type};\n`;
      binding++;
    }
  }

  wgsl += '\n';

  // Shared memory
  if (config.sharedMemory) {
    for (const shared of config.sharedMemory) {
      wgsl += `var<workgroup> ${shared.name}: array<${shared.type}, ${shared.size}>;\n`;
    }
    wgsl += '\n';
  }

  // Main function
  const [wgx, wgy, wgz] = config.workgroupSize;
  wgsl += `@compute @workgroup_size(${wgx}, ${wgy}, ${wgz})\n`;
  wgsl += `fn ${config.name}(\n`;
  wgsl += `    @builtin(global_invocation_id) global_id: vec3<u32>,\n`;
  wgsl += `    @builtin(local_invocation_id) local_id: vec3<u32>,\n`;
  wgsl += `    @builtin(workgroup_id) workgroup_id: vec3<u32>,\n`;
  wgsl += `    @builtin(local_invocation_index) local_index: u32\n`;
  wgsl += `) {\n`;
  wgsl += `    let idx = global_id.x;\n`;
  wgsl += `    // TODO: Add kernel logic here\n`;
  wgsl += `}\n`;

  return wgsl;
}

/**
 * Documentation of CUDA -> WebGPU conceptual mappings
 */
export const CONCEPT_MAPPINGS = {
  'CUDA Grid': 'WebGPU Dispatch (workgroup count)',
  'CUDA Block': 'WebGPU Workgroup',
  'CUDA Thread': 'WebGPU Invocation',
  'CUDA Warp (32 threads)': 'WebGPU Subgroup (size varies, typically 32)',
  'Shared Memory': 'Workgroup Variables (var<workgroup>)',
  'Global Memory': 'Storage Buffers',
  'Constant Memory': 'Uniform Buffers',
  'Texture Memory': 'Sampled Textures',
  'Surface Memory': 'Storage Textures',
  'cudaMemcpy': 'queue.writeBuffer / buffer copy',
  'cudaDeviceSynchronize': 'queue.onSubmittedWorkDone()',
  '__syncthreads()': 'workgroupBarrier()',
  '__threadfence()': 'storageBarrier()',
  'atomicAdd (float)': 'CAS loop with atomicCompareExchangeWeak',
};

/**
 * Limitations and differences documentation
 */
export const LIMITATIONS = [
  'WebGPU has no native double (f64) support in most implementations',
  'Warp-level primitives (shuffle, ballot) must be emulated with shared memory',
  'No dynamic parallelism (kernel launching kernels)',
  'No direct equivalent to CUDA streams (single queue model)',
  'Maximum workgroup size is typically 256 (vs CUDA 1024)',
  'Float atomics require emulation via CAS loops',
  'No hardware texture interpolation for compute (must use samplers)',
  'No system-wide memory fence',
  'No cooperative groups with grid-wide sync',
  'Shared memory size is more limited',
];
