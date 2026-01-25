/**
 * CUDA Atomic Operations for WebGPU
 *
 * Provides CUDA-compatible atomic operations
 */

// WGSL Atomic Operations Implementation
export const CUDA_ATOMIC_WGSL = `
// ============================================================================
// Atomic Operations
// WebGPU provides atomic operations on i32 and u32 types
// ============================================================================

// Atomic Add
fn atomicAdd_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicAdd(ptr, val);
}

fn atomicAdd_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicAdd(ptr, val);
}

// Workgroup atomics
fn atomicAdd_workgroup_i32(ptr: ptr<workgroup, atomic<i32>>, val: i32) -> i32 {
    return atomicAdd(ptr, val);
}

fn atomicAdd_workgroup_u32(ptr: ptr<workgroup, atomic<u32>>, val: u32) -> u32 {
    return atomicAdd(ptr, val);
}

// Atomic Sub
fn atomicSub_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicSub(ptr, val);
}

fn atomicSub_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicSub(ptr, val);
}

// Atomic Exchange
fn atomicExch_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicExchange(ptr, val);
}

fn atomicExch_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicExchange(ptr, val);
}

// Atomic Min
fn atomicMin_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicMin(ptr, val);
}

fn atomicMin_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicMin(ptr, val);
}

// Atomic Max
fn atomicMax_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicMax(ptr, val);
}

fn atomicMax_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicMax(ptr, val);
}

// Atomic And
fn atomicAnd_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicAnd(ptr, val);
}

fn atomicAnd_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicAnd(ptr, val);
}

// Atomic Or
fn atomicOr_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicOr(ptr, val);
}

fn atomicOr_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicOr(ptr, val);
}

// Atomic Xor
fn atomicXor_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) -> i32 {
    return atomicXor(ptr, val);
}

fn atomicXor_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) -> u32 {
    return atomicXor(ptr, val);
}

// Atomic Compare and Swap (CAS)
fn atomicCAS_i32(ptr: ptr<storage, atomic<i32>, read_write>, compare: i32, val: i32) -> i32 {
    return atomicCompareExchangeWeak(ptr, compare, val).old_value;
}

fn atomicCAS_u32(ptr: ptr<storage, atomic<u32>, read_write>, compare: u32, val: u32) -> u32 {
    return atomicCompareExchangeWeak(ptr, compare, val).old_value;
}

// Atomic Load
fn atomicLoad_i32(ptr: ptr<storage, atomic<i32>, read_write>) -> i32 {
    return atomicLoad(ptr);
}

fn atomicLoad_u32(ptr: ptr<storage, atomic<u32>, read_write>) -> u32 {
    return atomicLoad(ptr);
}

// Atomic Store
fn atomicStore_i32(ptr: ptr<storage, atomic<i32>, read_write>, val: i32) {
    atomicStore(ptr, val);
}

fn atomicStore_u32(ptr: ptr<storage, atomic<u32>, read_write>, val: u32) {
    atomicStore(ptr, val);
}

// ============================================================================
// Emulated Float Atomics (using CAS loop)
// WebGPU doesn't have native float atomics, so we emulate them
// ============================================================================

// Note: These require the atomic to be stored as u32 (reinterpreted float bits)
// The caller must use bitcast<f32>(atomicLoad(...)) to read the float value

fn atomicAdd_f32_emulated(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) -> f32 {
    var old_bits: u32;
    var new_bits: u32;
    var old_val: f32;

    loop {
        old_bits = atomicLoad(ptr);
        old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        new_bits = bitcast<u32>(new_val);

        let result = atomicCompareExchangeWeak(ptr, old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
    }

    return old_val;
}

fn atomicMax_f32_emulated(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) -> f32 {
    var old_bits: u32;
    var old_val: f32;

    loop {
        old_bits = atomicLoad(ptr);
        old_val = bitcast<f32>(old_bits);

        if (val <= old_val) {
            break;
        }

        let new_bits = bitcast<u32>(val);
        let result = atomicCompareExchangeWeak(ptr, old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
    }

    return old_val;
}

fn atomicMin_f32_emulated(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) -> f32 {
    var old_bits: u32;
    var old_val: f32;

    loop {
        old_bits = atomicLoad(ptr);
        old_val = bitcast<f32>(old_bits);

        if (val >= old_val) {
            break;
        }

        let new_bits = bitcast<u32>(val);
        let result = atomicCompareExchangeWeak(ptr, old_bits, new_bits);
        if (result.exchanged) {
            break;
        }
    }

    return old_val;
}

fn atomicExch_f32_emulated(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) -> f32 {
    let new_bits = bitcast<u32>(val);
    let old_bits = atomicExchange(ptr, new_bits);
    return bitcast<f32>(old_bits);
}

// ============================================================================
// Double-Wide Atomics (emulated with two u32 atomics)
// For 64-bit values, we use a lock-based approach
// ============================================================================

var<workgroup> atomic_lock: atomic<u32>;

fn acquire_lock() {
    loop {
        let result = atomicCompareExchangeWeak(&atomic_lock, 0u, 1u);
        if (result.exchanged) {
            break;
        }
    }
}

fn release_lock() {
    atomicStore(&atomic_lock, 0u);
}

// Note: For 64-bit atomics, consider using two 32-bit operations
// with proper synchronization, or restructure data to use 32-bit atomics

// ============================================================================
// System-Level Atomics (for inter-workgroup communication)
// ============================================================================

// These require careful use with memory barriers
fn threadfence() {
    storageBarrier();
}

fn threadfence_block() {
    workgroupBarrier();
}

fn threadfence_system() {
    // WebGPU doesn't have system-wide fence
    // storageBarrier provides device-scope ordering
    storageBarrier();
}

// ============================================================================
// Atomic Counter Helpers
// ============================================================================

fn atomicInc_u32(ptr: ptr<storage, atomic<u32>, read_write>, max_val: u32) -> u32 {
    var old_val: u32;
    loop {
        old_val = atomicLoad(ptr);
        let new_val = select(0u, old_val + 1u, old_val < max_val);

        let result = atomicCompareExchangeWeak(ptr, old_val, new_val);
        if (result.exchanged) {
            break;
        }
    }
    return old_val;
}

fn atomicDec_u32(ptr: ptr<storage, atomic<u32>, read_write>, max_val: u32) -> u32 {
    var old_val: u32;
    loop {
        old_val = atomicLoad(ptr);
        let new_val = select(max_val, old_val - 1u, old_val != 0u && old_val <= max_val);

        let result = atomicCompareExchangeWeak(ptr, old_val, new_val);
        if (result.exchanged) {
            break;
        }
    }
    return old_val;
}

// ============================================================================
// Reduction Helpers with Atomics
// ============================================================================

fn atomicAddGlobal_f32(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    // Simple global atomic add for float
    atomicAdd_f32_emulated(ptr, val);
}

fn atomicMaxGlobal_f32(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    atomicMax_f32_emulated(ptr, val);
}

fn atomicMinGlobal_f32(ptr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    atomicMin_f32_emulated(ptr, val);
}
`;

// TypeScript exports
export function withCudaAtomic(shaderCode: string): string {
  return CUDA_ATOMIC_WGSL + '\n\n' + shaderCode;
}

export const ATOMIC_OPERATIONS_INFO = {
  native: {
    types: ['i32', 'u32'],
    operations: [
      'atomicAdd', 'atomicSub', 'atomicExchange',
      'atomicMin', 'atomicMax',
      'atomicAnd', 'atomicOr', 'atomicXor',
      'atomicCompareExchangeWeak',
      'atomicLoad', 'atomicStore'
    ],
  },
  emulated: {
    types: ['f32'],
    operations: ['atomicAdd_f32', 'atomicMax_f32', 'atomicMin_f32', 'atomicExch_f32'],
    notes: 'Float atomics are emulated using CAS loops on u32 representation',
  },
  barriers: {
    storageBarrier: 'Ensures memory operations are visible across workgroups',
    workgroupBarrier: 'Synchronizes threads within a workgroup',
  },
  limitations: [
    'No native 64-bit atomics in WebGPU',
    'Float atomics require emulation via CAS loops (slower)',
    'No native system-wide memory ordering',
  ],
};

// Helper class for managing atomic buffers
export class AtomicBuffer {
  private device: GPUDevice;
  private buffer: GPUBuffer;
  private size: number;

  constructor(device: GPUDevice, count: number, type: 'i32' | 'u32' = 'u32') {
    this.device = device;
    this.size = count * 4; // 4 bytes per atomic
    this.buffer = device.createBuffer({
      size: this.size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  getBuffer(): GPUBuffer {
    return this.buffer;
  }

  async reset(value: number = 0): Promise<void> {
    const data = new Uint32Array(this.size / 4);
    data.fill(value);
    this.device.queue.writeBuffer(this.buffer, 0, data);
  }

  async readValues(): Promise<Uint32Array> {
    const staging = this.device.createBuffer({
      size: this.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, this.size);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    return result;
  }

  destroy(): void {
    this.buffer.destroy();
  }
}
