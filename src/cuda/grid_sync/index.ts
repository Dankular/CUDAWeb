/**
 * Grid-Wide Synchronization for WebGPU
 *
 * Provides CUDA-compatible grid-wide synchronization primitives.
 *
 * Unlike CUDA, WebGPU cannot synchronize across workgroups within a single
 * dispatch. This module implements grid-wide sync using:
 *
 * 1. Atomic counters for arrival/departure barriers
 * 2. Multi-dispatch patterns with explicit GPU synchronization
 * 3. Producer-consumer patterns for data exchange
 *
 * Usage requires a global sync buffer bound to the shader.
 */

// WGSL Grid Synchronization Implementation
export const CUDA_GRID_SYNC_WGSL = `
// ============================================================================
// Grid Synchronization Constants and State
// ============================================================================

// Grid sync buffer layout:
// [0]: arrival_count - Number of workgroups that have arrived
// [1]: departure_count - Number of workgroups that have departed
// [2]: sense - Barrier sense (toggles each barrier)
// [3]: generation - Barrier generation counter
// [4-7]: reserved for future use
// [8+]: Per-workgroup flags or data

const GRID_SYNC_ARRIVAL: u32 = 0u;
const GRID_SYNC_DEPARTURE: u32 = 1u;
const GRID_SYNC_SENSE: u32 = 2u;
const GRID_SYNC_GENERATION: u32 = 3u;
const GRID_SYNC_WORKGROUP_BASE: u32 = 8u;

// Thread-local tracking of barrier sense
var<private> grid_barrier_sense: u32 = 0u;

// ============================================================================
// Grid Barrier Implementation
// ============================================================================

// Initialize grid sync state (call from host before first use)
// This function is called by ONE thread (global_id = 0)
fn grid_sync_init(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>) {
    if (global_invocation_id.x == 0u && global_invocation_id.y == 0u && global_invocation_id.z == 0u) {
        atomicStore(&(*sync_buffer)[GRID_SYNC_ARRIVAL], 0u);
        atomicStore(&(*sync_buffer)[GRID_SYNC_DEPARTURE], 0u);
        atomicStore(&(*sync_buffer)[GRID_SYNC_SENSE], 0u);
        atomicStore(&(*sync_buffer)[GRID_SYNC_GENERATION], 0u);
    }
}

// Dissemination barrier for grid-wide synchronization
// All workgroups must call this; requires expected_workgroups parameter
fn grid_sync(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
             expected_workgroups: u32) {
    // Only thread 0 of each workgroup participates in grid barrier
    if (local_invocation_index != 0u) {
        workgroupBarrier();
        return;
    }

    let my_sense = grid_barrier_sense ^ 1u;

    // Arrival phase
    let arrived = atomicAdd(&(*sync_buffer)[GRID_SYNC_ARRIVAL], 1u) + 1u;

    // Last workgroup to arrive resets arrival and signals
    if (arrived == expected_workgroups) {
        atomicStore(&(*sync_buffer)[GRID_SYNC_ARRIVAL], 0u);
        storageBarrier();
        atomicStore(&(*sync_buffer)[GRID_SYNC_SENSE], my_sense);
    } else {
        // Wait for sense to change
        loop {
            let current_sense = atomicLoad(&(*sync_buffer)[GRID_SYNC_SENSE]);
            if (current_sense == my_sense) {
                break;
            }
            // Spin - storage barrier ensures visibility
            storageBarrier();
        }
    }

    grid_barrier_sense = my_sense;
    storageBarrier();
    workgroupBarrier();
}

// ============================================================================
// Named Barrier for Partial Grid Synchronization
// ============================================================================

struct named_barrier {
    barrier_id: u32,
    expected_count: u32,
}

fn create_named_barrier(id: u32, count: u32) -> named_barrier {
    var nb: named_barrier;
    nb.barrier_id = id;
    nb.expected_count = count;
    return nb;
}

fn named_barrier_arrive(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                         barrier: named_barrier) {
    if (local_invocation_index != 0u) {
        return;
    }

    let offset = GRID_SYNC_WORKGROUP_BASE + barrier.barrier_id * 2u;
    atomicAdd(&(*sync_buffer)[offset], 1u);
}

fn named_barrier_wait(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                       barrier: named_barrier) {
    if (local_invocation_index != 0u) {
        workgroupBarrier();
        return;
    }

    let offset = GRID_SYNC_WORKGROUP_BASE + barrier.barrier_id * 2u;

    loop {
        let count = atomicLoad(&(*sync_buffer)[offset]);
        if (count >= barrier.expected_count) {
            break;
        }
        storageBarrier();
    }

    workgroupBarrier();
}

fn named_barrier_arrive_and_wait(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                                  barrier: named_barrier) {
    named_barrier_arrive(sync_buffer, barrier);
    named_barrier_wait(sync_buffer, barrier);
}

fn named_barrier_reset(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                        barrier: named_barrier) {
    if (local_invocation_index == 0u) {
        let offset = GRID_SYNC_WORKGROUP_BASE + barrier.barrier_id * 2u;
        atomicStore(&(*sync_buffer)[offset], 0u);
    }
    workgroupBarrier();
}

// ============================================================================
// Producer-Consumer Pattern
// ============================================================================

// Signal that this workgroup has produced data
fn grid_signal_produced(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                         producer_id: u32) {
    if (local_invocation_index == 0u) {
        let offset = GRID_SYNC_WORKGROUP_BASE + producer_id;
        atomicStore(&(*sync_buffer)[offset], 1u);
        storageBarrier();
    }
    workgroupBarrier();
}

// Wait for a producer to signal completion
fn grid_wait_for_producer(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                           producer_id: u32) {
    if (local_invocation_index == 0u) {
        let offset = GRID_SYNC_WORKGROUP_BASE + producer_id;
        loop {
            let ready = atomicLoad(&(*sync_buffer)[offset]);
            if (ready != 0u) {
                break;
            }
            storageBarrier();
        }
    }
    workgroupBarrier();
}

// ============================================================================
// Grid-Wide Reduction
// ============================================================================

// Two-phase reduction: local reduce + atomic global reduce
fn grid_reduce_sum_f32(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                        local_sum: f32,
                        result_index: u32) -> f32 {
    // First do workgroup reduction
    var<workgroup> wg_sum: f32;

    if (local_invocation_index == 0u) {
        wg_sum = 0.0;
    }
    workgroupBarrier();

    // Each thread adds to workgroup sum (simplified - use tree reduction in practice)
    atomicAdd(&sync_buffer[result_index], bitcast<u32>(local_sum));

    workgroupBarrier();
    return bitcast<f32>(atomicLoad(&(*sync_buffer)[result_index]));
}

// Grid-wide max reduction
fn grid_reduce_max_f32(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                        local_val: f32,
                        result_index: u32) -> f32 {
    if (local_invocation_index == 0u) {
        var old_bits: u32;
        let val_bits = bitcast<u32>(local_val);

        loop {
            old_bits = atomicLoad(&(*sync_buffer)[result_index]);
            let old_val = bitcast<f32>(old_bits);

            if (local_val <= old_val) {
                break;
            }

            let result = atomicCompareExchangeWeak(&(*sync_buffer)[result_index], old_bits, val_bits);
            if (result.exchanged) {
                break;
            }
        }
    }

    workgroupBarrier();
    return bitcast<f32>(atomicLoad(&(*sync_buffer)[result_index]));
}

// Grid-wide min reduction
fn grid_reduce_min_f32(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                        local_val: f32,
                        result_index: u32) -> f32 {
    if (local_invocation_index == 0u) {
        var old_bits: u32;
        let val_bits = bitcast<u32>(local_val);

        loop {
            old_bits = atomicLoad(&(*sync_buffer)[result_index]);
            let old_val = bitcast<f32>(old_bits);

            if (local_val >= old_val) {
                break;
            }

            let result = atomicCompareExchangeWeak(&(*sync_buffer)[result_index], old_bits, val_bits);
            if (result.exchanged) {
                break;
            }
        }
    }

    workgroupBarrier();
    return bitcast<f32>(atomicLoad(&(*sync_buffer)[result_index]));
}

// ============================================================================
// Grid Progress Tracking
// ============================================================================

struct grid_progress {
    total_workgroups: u32,
    completed_workgroups: u32,
    current_phase: u32,
}

fn get_grid_progress(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                      total_workgroups: u32) -> grid_progress {
    var progress: grid_progress;
    progress.total_workgroups = total_workgroups;
    progress.completed_workgroups = atomicLoad(&(*sync_buffer)[GRID_SYNC_ARRIVAL]);
    progress.current_phase = atomicLoad(&(*sync_buffer)[GRID_SYNC_GENERATION]);
    return progress;
}

fn mark_workgroup_complete(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>) {
    if (local_invocation_index == 0u) {
        atomicAdd(&(*sync_buffer)[GRID_SYNC_ARRIVAL], 1u);
        storageBarrier();
    }
}

fn increment_phase(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>) {
    if (local_invocation_index == 0u) {
        atomicAdd(&(*sync_buffer)[GRID_SYNC_GENERATION], 1u);
        atomicStore(&(*sync_buffer)[GRID_SYNC_ARRIVAL], 0u);
        storageBarrier();
    }
    workgroupBarrier();
}

// ============================================================================
// Multi-Pass Coordination
// ============================================================================

// Check if this is the last workgroup to reach this point
fn is_last_workgroup(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>,
                      expected_workgroups: u32) -> bool {
    var is_last = false;

    if (local_invocation_index == 0u) {
        let count = atomicAdd(&(*sync_buffer)[GRID_SYNC_ARRIVAL], 1u) + 1u;
        is_last = (count == expected_workgroups);
    }

    // Broadcast result to all threads in workgroup
    var<workgroup> result: u32;
    if (local_invocation_index == 0u) {
        result = select(0u, 1u, is_last);
    }
    workgroupBarrier();

    return result == 1u;
}

// Reset synchronization state for next iteration
fn grid_sync_reset(sync_buffer: ptr<storage, array<atomic<u32>>, read_write>) {
    if (local_invocation_index == 0u) {
        atomicStore(&(*sync_buffer)[GRID_SYNC_ARRIVAL], 0u);
        atomicStore(&(*sync_buffer)[GRID_SYNC_DEPARTURE], 0u);
        storageBarrier();
    }
    workgroupBarrier();
}
`;

// TypeScript exports
export function withGridSync(shaderCode: string): string {
  return CUDA_GRID_SYNC_WGSL + '\n\n' + shaderCode;
}

export const GRID_SYNC_INFO = {
  name: 'Grid-Wide Synchronization',
  description: 'Atomic counter-based grid synchronization for WebGPU',
  bufferLayout: {
    0: 'arrival_count - Workgroups arrived at barrier',
    1: 'departure_count - Workgroups departed from barrier',
    2: 'sense - Barrier sense flag (toggles each barrier)',
    3: 'generation - Barrier generation counter',
    '4-7': 'Reserved for future use',
    '8+': 'Per-workgroup flags and named barrier data',
  },
  functions: {
    'grid_sync_init': 'Initialize synchronization state (call once from thread 0)',
    'grid_sync': 'Full grid barrier - all workgroups synchronize',
    'create_named_barrier': 'Create a barrier for subset of workgroups',
    'named_barrier_arrive': 'Signal arrival at named barrier',
    'named_barrier_wait': 'Wait for all participants at named barrier',
    'named_barrier_arrive_and_wait': 'Arrive and wait in one call',
    'grid_signal_produced': 'Signal data production complete',
    'grid_wait_for_producer': 'Wait for producer to complete',
    'grid_reduce_sum_f32': 'Grid-wide sum reduction',
    'grid_reduce_max_f32': 'Grid-wide max reduction',
    'grid_reduce_min_f32': 'Grid-wide min reduction',
    'is_last_workgroup': 'Check if this is the last workgroup to arrive',
    'grid_sync_reset': 'Reset synchronization state for next iteration',
  },
  patterns: {
    'basic_barrier': `
// All workgroups must reach this point before any proceed
grid_sync(&sync_buffer, expected_workgroups);`,
    'producer_consumer': `
// Producer workgroup
process_data();
grid_signal_produced(&sync_buffer, workgroup_id.x);

// Consumer workgroup
grid_wait_for_producer(&sync_buffer, producer_id);
use_data();`,
    'phased_execution': `
// Phase 1
do_phase1_work();
grid_sync(&sync_buffer, num_workgroups);

// Phase 2
do_phase2_work();
grid_sync(&sync_buffer, num_workgroups);`,
  },
  limitations: [
    'Spin-waiting consumes GPU resources',
    'Requires all workgroups to participate (no early exit)',
    'Potential for deadlock if workgroup count mismatch',
    'Performance degrades with high workgroup counts',
    'Named barriers share buffer space - limit concurrent barriers',
  ],
  recommendations: [
    'Prefer multi-dispatch over grid sync when possible',
    'Keep workgroup counts reasonable (< 1024)',
    'Use named barriers for subset synchronization',
    'Initialize sync buffer from host before kernel launch',
    'Consider producer-consumer pattern for pipelines',
  ],
};

/**
 * Calculates the required sync buffer size in u32 elements
 * @param maxWorkgroups Maximum number of workgroups that will synchronize
 * @param maxNamedBarriers Maximum number of concurrent named barriers
 * @returns Required buffer size in u32 elements
 */
export function calculateSyncBufferSize(maxWorkgroups: number, maxNamedBarriers: number = 16): number {
  // Base: 8 u32 for core sync state
  // Plus 2 u32 per named barrier
  // Plus 1 u32 per workgroup for flags
  return 8 + (maxNamedBarriers * 2) + maxWorkgroups;
}

/**
 * Helper class for managing grid synchronization buffer
 */
export class GridSyncBuffer {
  private device: GPUDevice;
  private buffer: GPUBuffer;
  private maxWorkgroups: number;
  private maxNamedBarriers: number;

  constructor(device: GPUDevice, maxWorkgroups: number, maxNamedBarriers: number = 16) {
    this.device = device;
    this.maxWorkgroups = maxWorkgroups;
    this.maxNamedBarriers = maxNamedBarriers;

    const size = calculateSyncBufferSize(maxWorkgroups, maxNamedBarriers) * 4;

    this.buffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  getBuffer(): GPUBuffer {
    return this.buffer;
  }

  getMaxWorkgroups(): number {
    return this.maxWorkgroups;
  }

  /**
   * Reset sync buffer to initial state (call before each dispatch sequence)
   */
  async reset(): Promise<void> {
    const size = calculateSyncBufferSize(this.maxWorkgroups, this.maxNamedBarriers);
    const data = new Uint32Array(size);
    data.fill(0);
    // Initialize min reduction slots to max float
    // data[4] = 0x7F7FFFFF; // Float max for min reduction
    this.device.queue.writeBuffer(this.buffer, 0, data);
  }

  /**
   * Read current sync state (for debugging)
   */
  async readState(): Promise<{
    arrival: number;
    departure: number;
    sense: number;
    generation: number;
  }> {
    const staging = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer, 0, staging, 0, 16);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    return {
      arrival: data[0],
      departure: data[1],
      sense: data[2],
      generation: data[3],
    };
  }

  destroy(): void {
    this.buffer.destroy();
  }
}

/**
 * Multi-pass kernel execution helper for complex grid sync patterns
 */
export class MultiPassExecutor {
  private device: GPUDevice;
  private syncBuffer: GridSyncBuffer;

  constructor(device: GPUDevice, maxWorkgroups: number) {
    this.device = device;
    this.syncBuffer = new GridSyncBuffer(device, maxWorkgroups);
  }

  getSyncBuffer(): GPUBuffer {
    return this.syncBuffer.getBuffer();
  }

  /**
   * Execute multiple passes with synchronization between them
   */
  async executeWithSync(
    passes: Array<{
      pipeline: GPUComputePipeline;
      bindGroups: GPUBindGroup[];
      workgroupCount: [number, number, number];
    }>
  ): Promise<void> {
    await this.syncBuffer.reset();

    for (const pass of passes) {
      const encoder = this.device.createCommandEncoder();
      const computePass = encoder.beginComputePass();

      computePass.setPipeline(pass.pipeline);
      for (let i = 0; i < pass.bindGroups.length; i++) {
        computePass.setBindGroup(i, pass.bindGroups[i]);
      }
      computePass.dispatchWorkgroups(...pass.workgroupCount);
      computePass.end();

      this.device.queue.submit([encoder.finish()]);

      // Ensure pass completes before next
      await this.device.queue.onSubmittedWorkDone();
    }
  }

  destroy(): void {
    this.syncBuffer.destroy();
  }
}
