/**
 * CUDA Warp-Level Primitives for WebGPU
 *
 * Provides CUDA-compatible warp-level operations using WebGPU subgroups
 * Note: WebGPU subgroup support is still experimental
 */

// WGSL Warp-Level Primitives
// These require the subgroups feature in WebGPU (when available)
export const CUDA_WARP_WGSL = `
// ============================================================================
// Warp Configuration
// ============================================================================

const WARP_SIZE: u32 = 32u;

// Get lane ID within warp
fn __laneId() -> u32 {
    return local_invocation_index % WARP_SIZE;
}

// Get warp ID within block
fn __warpId() -> u32 {
    return local_invocation_index / WARP_SIZE;
}

// ============================================================================
// Warp Vote Functions (emulated via shared memory)
// ============================================================================

var<workgroup> warp_vote_result: array<u32, 32>;
var<workgroup> warp_ballot_result: array<u32, 32>;

// All threads in warp agree on predicate
fn __all_sync(mask: u32, predicate: bool) -> bool {
    let lane = __laneId();
    let warp = __warpId();

    // Store vote
    if (predicate) {
        atomicOr(&warp_vote_result[warp], 1u << lane);
    }
    workgroupBarrier();

    // Check if all lanes voted true
    let result = warp_vote_result[warp] == mask;
    workgroupBarrier();

    // Clear for next use
    if (lane == 0u) {
        warp_vote_result[warp] = 0u;
    }
    workgroupBarrier();

    return result;
}

// Any thread in warp has predicate true
fn __any_sync(mask: u32, predicate: bool) -> bool {
    let lane = __laneId();
    let warp = __warpId();

    if (predicate) {
        atomicOr(&warp_vote_result[warp], 1u << lane);
    }
    workgroupBarrier();

    let result = warp_vote_result[warp] != 0u;
    workgroupBarrier();

    if (lane == 0u) {
        warp_vote_result[warp] = 0u;
    }
    workgroupBarrier();

    return result;
}

// Ballot - return mask of threads with predicate true
fn __ballot_sync(mask: u32, predicate: bool) -> u32 {
    let lane = __laneId();
    let warp = __warpId();

    if (predicate) {
        atomicOr(&warp_ballot_result[warp], 1u << lane);
    }
    workgroupBarrier();

    let result = warp_ballot_result[warp];
    workgroupBarrier();

    if (lane == 0u) {
        warp_ballot_result[warp] = 0u;
    }
    workgroupBarrier();

    return result & mask;
}

// Population count of ballot
fn __popc_ballot(ballot: u32) -> u32 {
    return countOneBits(ballot);
}

// ============================================================================
// Warp Shuffle Functions (emulated via shared memory)
// ============================================================================

var<workgroup> warp_shuffle_data: array<f32, 1024>; // Up to 32 warps * 32 lanes
var<workgroup> warp_shuffle_data_i32: array<i32, 1024>;
var<workgroup> warp_shuffle_data_u32: array<u32, 1024>;

// Shuffle - exchange data between lanes
fn __shfl_sync_f32(mask: u32, val: f32, srcLane: u32, width: u32) -> f32 {
    let lane = __laneId();
    let warp = __warpId();
    let idx = warp * WARP_SIZE + lane;

    warp_shuffle_data[idx] = val;
    workgroupBarrier();

    let srcIdx = warp * WARP_SIZE + (srcLane % width) + (lane & ~(width - 1u));
    let result = warp_shuffle_data[srcIdx];
    workgroupBarrier();

    return result;
}

fn __shfl_sync_i32(mask: u32, val: i32, srcLane: u32, width: u32) -> i32 {
    let lane = __laneId();
    let warp = __warpId();
    let idx = warp * WARP_SIZE + lane;

    warp_shuffle_data_i32[idx] = val;
    workgroupBarrier();

    let srcIdx = warp * WARP_SIZE + (srcLane % width) + (lane & ~(width - 1u));
    let result = warp_shuffle_data_i32[srcIdx];
    workgroupBarrier();

    return result;
}

fn __shfl_sync_u32(mask: u32, val: u32, srcLane: u32, width: u32) -> u32 {
    let lane = __laneId();
    let warp = __warpId();
    let idx = warp * WARP_SIZE + lane;

    warp_shuffle_data_u32[idx] = val;
    workgroupBarrier();

    let srcIdx = warp * WARP_SIZE + (srcLane % width) + (lane & ~(width - 1u));
    let result = warp_shuffle_data_u32[srcIdx];
    workgroupBarrier();

    return result;
}

// Shuffle up - get value from lane - delta
fn __shfl_up_sync_f32(mask: u32, val: f32, delta: u32, width: u32) -> f32 {
    let lane = __laneId();
    let srcLane = select(lane, lane - delta, lane >= delta);
    return __shfl_sync_f32(mask, val, srcLane, width);
}

fn __shfl_up_sync_i32(mask: u32, val: i32, delta: u32, width: u32) -> i32 {
    let lane = __laneId();
    let srcLane = select(lane, lane - delta, lane >= delta);
    return __shfl_sync_i32(mask, val, srcLane, width);
}

// Shuffle down - get value from lane + delta
fn __shfl_down_sync_f32(mask: u32, val: f32, delta: u32, width: u32) -> f32 {
    let lane = __laneId();
    let srcLane = select(lane, lane + delta, lane + delta < width);
    return __shfl_sync_f32(mask, val, srcLane, width);
}

fn __shfl_down_sync_i32(mask: u32, val: i32, delta: u32, width: u32) -> i32 {
    let lane = __laneId();
    let srcLane = select(lane, lane + delta, lane + delta < width);
    return __shfl_sync_i32(mask, val, srcLane, width);
}

// Shuffle XOR - exchange with lane XOR srcLane
fn __shfl_xor_sync_f32(mask: u32, val: f32, laneMask: u32, width: u32) -> f32 {
    let lane = __laneId();
    let srcLane = lane ^ laneMask;
    return __shfl_sync_f32(mask, val, srcLane, width);
}

fn __shfl_xor_sync_i32(mask: u32, val: i32, laneMask: u32, width: u32) -> i32 {
    let lane = __laneId();
    let srcLane = lane ^ laneMask;
    return __shfl_sync_i32(mask, val, srcLane, width);
}

// ============================================================================
// Warp Reduction Functions
// ============================================================================

// Warp-level sum reduction using shuffle
fn __warp_reduce_sum_f32(val: f32) -> f32 {
    var result = val;
    result = result + __shfl_xor_sync_f32(0xFFFFFFFFu, result, 16u, 32u);
    result = result + __shfl_xor_sync_f32(0xFFFFFFFFu, result, 8u, 32u);
    result = result + __shfl_xor_sync_f32(0xFFFFFFFFu, result, 4u, 32u);
    result = result + __shfl_xor_sync_f32(0xFFFFFFFFu, result, 2u, 32u);
    result = result + __shfl_xor_sync_f32(0xFFFFFFFFu, result, 1u, 32u);
    return result;
}

fn __warp_reduce_sum_i32(val: i32) -> i32 {
    var result = val;
    result = result + __shfl_xor_sync_i32(0xFFFFFFFFu, result, 16u, 32u);
    result = result + __shfl_xor_sync_i32(0xFFFFFFFFu, result, 8u, 32u);
    result = result + __shfl_xor_sync_i32(0xFFFFFFFFu, result, 4u, 32u);
    result = result + __shfl_xor_sync_i32(0xFFFFFFFFu, result, 2u, 32u);
    result = result + __shfl_xor_sync_i32(0xFFFFFFFFu, result, 1u, 32u);
    return result;
}

// Warp-level max reduction
fn __warp_reduce_max_f32(val: f32) -> f32 {
    var result = val;
    result = max(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 16u, 32u));
    result = max(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 8u, 32u));
    result = max(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 4u, 32u));
    result = max(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 2u, 32u));
    result = max(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 1u, 32u));
    return result;
}

fn __warp_reduce_max_i32(val: i32) -> i32 {
    var result = val;
    result = max(result, __shfl_xor_sync_i32(0xFFFFFFFFu, result, 16u, 32u));
    result = max(result, __shfl_xor_sync_i32(0xFFFFFFFFu, result, 8u, 32u));
    result = max(result, __shfl_xor_sync_i32(0xFFFFFFFFu, result, 4u, 32u));
    result = max(result, __shfl_xor_sync_i32(0xFFFFFFFFu, result, 2u, 32u));
    result = max(result, __shfl_xor_sync_i32(0xFFFFFFFFu, result, 1u, 32u));
    return result;
}

// Warp-level min reduction
fn __warp_reduce_min_f32(val: f32) -> f32 {
    var result = val;
    result = min(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 16u, 32u));
    result = min(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 8u, 32u));
    result = min(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 4u, 32u));
    result = min(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 2u, 32u));
    result = min(result, __shfl_xor_sync_f32(0xFFFFFFFFu, result, 1u, 32u));
    return result;
}

// ============================================================================
// Warp Scan (Prefix Sum) Functions
// ============================================================================

// Inclusive scan
fn __warp_scan_inclusive_sum_f32(val: f32) -> f32 {
    var result = val;
    var temp: f32;

    temp = __shfl_up_sync_f32(0xFFFFFFFFu, result, 1u, 32u);
    if (__laneId() >= 1u) { result = result + temp; }

    temp = __shfl_up_sync_f32(0xFFFFFFFFu, result, 2u, 32u);
    if (__laneId() >= 2u) { result = result + temp; }

    temp = __shfl_up_sync_f32(0xFFFFFFFFu, result, 4u, 32u);
    if (__laneId() >= 4u) { result = result + temp; }

    temp = __shfl_up_sync_f32(0xFFFFFFFFu, result, 8u, 32u);
    if (__laneId() >= 8u) { result = result + temp; }

    temp = __shfl_up_sync_f32(0xFFFFFFFFu, result, 16u, 32u);
    if (__laneId() >= 16u) { result = result + temp; }

    return result;
}

// Exclusive scan
fn __warp_scan_exclusive_sum_f32(val: f32) -> f32 {
    let inclusive = __warp_scan_inclusive_sum_f32(val);
    return inclusive - val;
}

fn __warp_scan_inclusive_sum_i32(val: i32) -> i32 {
    var result = val;
    var temp: i32;

    temp = __shfl_up_sync_i32(0xFFFFFFFFu, result, 1u, 32u);
    if (__laneId() >= 1u) { result = result + temp; }

    temp = __shfl_up_sync_i32(0xFFFFFFFFu, result, 2u, 32u);
    if (__laneId() >= 2u) { result = result + temp; }

    temp = __shfl_up_sync_i32(0xFFFFFFFFu, result, 4u, 32u);
    if (__laneId() >= 4u) { result = result + temp; }

    temp = __shfl_up_sync_i32(0xFFFFFFFFu, result, 8u, 32u);
    if (__laneId() >= 8u) { result = result + temp; }

    temp = __shfl_up_sync_i32(0xFFFFFFFFu, result, 16u, 32u);
    if (__laneId() >= 16u) { result = result + temp; }

    return result;
}

fn __warp_scan_exclusive_sum_i32(val: i32) -> i32 {
    let inclusive = __warp_scan_inclusive_sum_i32(val);
    return inclusive - val;
}

// ============================================================================
// Active Mask Functions
// ============================================================================

fn __activemask() -> u32 {
    // In compute shaders, typically all lanes in a workgroup are active
    return 0xFFFFFFFFu;
}

// Match any/all - find lanes with matching values
fn __match_any_sync_u32(mask: u32, val: u32) -> u32 {
    var result: u32 = 0u;
    let lane = __laneId();
    let warp = __warpId();

    // Broadcast each lane's value and compare
    for (var i: u32 = 0u; i < WARP_SIZE; i = i + 1u) {
        if ((mask & (1u << i)) != 0u) {
            let other_val = __shfl_sync_u32(mask, val, i, WARP_SIZE);
            if (other_val == val) {
                result = result | (1u << i);
            }
        }
    }

    return result;
}

fn __match_all_sync_u32(mask: u32, val: u32, pred: ptr<function, bool>) -> u32 {
    let match_mask = __match_any_sync_u32(mask, val);
    *pred = (match_mask == mask);
    return match_mask;
}

// ============================================================================
// Warp Synchronization
// ============================================================================

fn __syncwarp(mask: u32) {
    // In WGSL, workgroupBarrier synchronizes the entire workgroup
    // For warp-level sync, we use this as an approximation
    workgroupBarrier();
}
`;

// TypeScript exports
export const WARP_SIZE = 32;

export function withCudaWarp(shaderCode: string): string {
  return CUDA_WARP_WGSL + '\n\n' + shaderCode;
}

export const CUDA_WARP_INFO = {
  warpSize: WARP_SIZE,
  functions: {
    voting: ['__all_sync', '__any_sync', '__ballot_sync'],
    shuffle: [
      '__shfl_sync', '__shfl_up_sync', '__shfl_down_sync', '__shfl_xor_sync'
    ],
    reduction: [
      '__warp_reduce_sum', '__warp_reduce_max', '__warp_reduce_min'
    ],
    scan: [
      '__warp_scan_inclusive_sum', '__warp_scan_exclusive_sum'
    ],
    utility: [
      '__laneId', '__warpId', '__activemask', '__syncwarp',
      '__match_any_sync', '__match_all_sync'
    ],
  },
  notes: [
    'Warp operations are emulated using shared memory and workgroup barriers',
    'Performance may vary compared to native CUDA warp intrinsics',
    'WebGPU subgroups feature will provide native support when available',
  ],
};
