/**
 * CUDA Cooperative Groups for WebGPU
 *
 * Provides CUDA-compatible cooperative groups functionality
 */

// WGSL Cooperative Groups Implementation
export const CUDA_COOPERATIVE_GROUPS_WGSL = `
// ============================================================================
// Cooperative Groups Types and Constants
// ============================================================================

const CG_WARP_SIZE: u32 = 32u;

// ============================================================================
// Thread Block (Workgroup) Group
// ============================================================================

struct thread_block {
    size: u32,
    thread_rank: u32,
}

fn this_thread_block() -> thread_block {
    var tb: thread_block;
    // Workgroup size in 1D
    tb.size = workgroup_size.x * workgroup_size.y * workgroup_size.z;
    tb.thread_rank = local_invocation_index;
    return tb;
}

fn thread_block_size(g: thread_block) -> u32 {
    return g.size;
}

fn thread_block_thread_rank(g: thread_block) -> u32 {
    return g.thread_rank;
}

fn thread_block_sync(g: thread_block) {
    workgroupBarrier();
}

fn thread_block_dim(g: thread_block) -> vec3<u32> {
    return workgroup_size;
}

fn thread_block_group_index() -> vec3<u32> {
    return workgroup_id;
}

fn thread_block_thread_index() -> vec3<u32> {
    return local_invocation_id;
}

// ============================================================================
// Grid Group (Compute Dispatch)
// ============================================================================

struct grid_group {
    size: u32,
    thread_rank: u32,
    block_rank: u32,
    num_blocks: u32,
}

fn this_grid() -> grid_group {
    var gg: grid_group;
    let block_size = workgroup_size.x * workgroup_size.y * workgroup_size.z;
    let num_workgroups = num_workgroups.x * num_workgroups.y * num_workgroups.z;

    gg.size = block_size * num_workgroups;
    gg.block_rank = workgroup_id.x +
                    workgroup_id.y * num_workgroups.x +
                    workgroup_id.z * num_workgroups.x * num_workgroups.y;
    gg.thread_rank = gg.block_rank * block_size + local_invocation_index;
    gg.num_blocks = num_workgroups;

    return gg;
}

fn grid_group_size(g: grid_group) -> u32 {
    return g.size;
}

fn grid_group_thread_rank(g: grid_group) -> u32 {
    return g.thread_rank;
}

fn grid_group_block_rank(g: grid_group) -> u32 {
    return g.block_rank;
}

fn grid_group_num_blocks(g: grid_group) -> u32 {
    return g.num_blocks;
}

fn grid_group_num_threads(g: grid_group) -> u32 {
    return g.size;
}

fn grid_group_block_dim(g: grid_group) -> vec3<u32> {
    return workgroup_size;
}

fn grid_group_dim_blocks(g: grid_group) -> vec3<u32> {
    return num_workgroups;
}

// Note: Grid-wide sync is not directly supported in WebGPU
// This would require multiple dispatch calls with synchronization
fn grid_group_sync(g: grid_group) {
    // Cannot synchronize across workgroups in a single dispatch
    // This is a no-op placeholder
    workgroupBarrier();
}

// ============================================================================
// Tile Group (Subset of Thread Block)
// ============================================================================

struct thread_block_tile_32 {
    size: u32,
    thread_rank: u32,
    meta_group_rank: u32,
}

struct thread_block_tile_16 {
    size: u32,
    thread_rank: u32,
    meta_group_rank: u32,
}

struct thread_block_tile_8 {
    size: u32,
    thread_rank: u32,
    meta_group_rank: u32,
}

struct thread_block_tile_4 {
    size: u32,
    thread_rank: u32,
    meta_group_rank: u32,
}

struct thread_block_tile_2 {
    size: u32,
    thread_rank: u32,
    meta_group_rank: u32,
}

struct thread_block_tile_1 {
    size: u32,
    thread_rank: u32,
    meta_group_rank: u32,
}

fn tiled_partition_32(g: thread_block) -> thread_block_tile_32 {
    var tile: thread_block_tile_32;
    tile.size = 32u;
    tile.thread_rank = g.thread_rank % 32u;
    tile.meta_group_rank = g.thread_rank / 32u;
    return tile;
}

fn tiled_partition_16(g: thread_block) -> thread_block_tile_16 {
    var tile: thread_block_tile_16;
    tile.size = 16u;
    tile.thread_rank = g.thread_rank % 16u;
    tile.meta_group_rank = g.thread_rank / 16u;
    return tile;
}

fn tiled_partition_8(g: thread_block) -> thread_block_tile_8 {
    var tile: thread_block_tile_8;
    tile.size = 8u;
    tile.thread_rank = g.thread_rank % 8u;
    tile.meta_group_rank = g.thread_rank / 8u;
    return tile;
}

fn tiled_partition_4(g: thread_block) -> thread_block_tile_4 {
    var tile: thread_block_tile_4;
    tile.size = 4u;
    tile.thread_rank = g.thread_rank % 4u;
    tile.meta_group_rank = g.thread_rank / 4u;
    return tile;
}

fn tiled_partition_2(g: thread_block) -> thread_block_tile_2 {
    var tile: thread_block_tile_2;
    tile.size = 2u;
    tile.thread_rank = g.thread_rank % 2u;
    tile.meta_group_rank = g.thread_rank / 2u;
    return tile;
}

fn tiled_partition_1(g: thread_block) -> thread_block_tile_1 {
    var tile: thread_block_tile_1;
    tile.size = 1u;
    tile.thread_rank = 0u;
    tile.meta_group_rank = g.thread_rank;
    return tile;
}

// Tile sync (barrier within tile)
fn tile_sync_32(tile: thread_block_tile_32) {
    workgroupBarrier();
}

fn tile_sync_16(tile: thread_block_tile_16) {
    workgroupBarrier();
}

fn tile_sync_8(tile: thread_block_tile_8) {
    workgroupBarrier();
}

// ============================================================================
// Coalesced Group (Active Threads)
// ============================================================================

struct coalesced_group {
    size: u32,
    thread_rank: u32,
    mask: u32,
}

fn coalesced_threads() -> coalesced_group {
    var cg: coalesced_group;
    // In compute shaders, typically all threads in workgroup are active
    cg.size = workgroup_size.x * workgroup_size.y * workgroup_size.z;
    cg.thread_rank = local_invocation_index;
    cg.mask = 0xFFFFFFFFu; // Assume all active
    return cg;
}

fn coalesced_group_size(g: coalesced_group) -> u32 {
    return g.size;
}

fn coalesced_group_thread_rank(g: coalesced_group) -> u32 {
    return g.thread_rank;
}

fn coalesced_group_sync(g: coalesced_group) {
    workgroupBarrier();
}

// ============================================================================
// Binary Partition
// ============================================================================

struct binary_partition {
    size: u32,
    thread_rank: u32,
    mask: u32,
}

fn binary_partition_create(g: coalesced_group, pred: bool) -> binary_partition {
    var bp: binary_partition;
    bp.size = g.size / 2u; // Approximate
    bp.thread_rank = g.thread_rank;
    bp.mask = select(0x55555555u, 0xAAAAAAAAu, pred);
    return bp;
}

// ============================================================================
// Collective Operations
// ============================================================================

// Shared memory for collective operations
var<workgroup> cg_shared_data_f32: array<f32, 1024>;
var<workgroup> cg_shared_data_i32: array<i32, 1024>;
var<workgroup> cg_shared_data_u32: array<u32, 1024>;

// Reduce within thread block
fn reduce_sum_f32(g: thread_block, val: f32) -> f32 {
    let idx = g.thread_rank;
    cg_shared_data_f32[idx] = val;
    workgroupBarrier();

    // Tree reduction
    var stride = g.size / 2u;
    while (stride > 0u) {
        if (idx < stride) {
            cg_shared_data_f32[idx] = cg_shared_data_f32[idx] + cg_shared_data_f32[idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return cg_shared_data_f32[0];
}

fn reduce_max_f32(g: thread_block, val: f32) -> f32 {
    let idx = g.thread_rank;
    cg_shared_data_f32[idx] = val;
    workgroupBarrier();

    var stride = g.size / 2u;
    while (stride > 0u) {
        if (idx < stride) {
            cg_shared_data_f32[idx] = max(cg_shared_data_f32[idx], cg_shared_data_f32[idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return cg_shared_data_f32[0];
}

fn reduce_min_f32(g: thread_block, val: f32) -> f32 {
    let idx = g.thread_rank;
    cg_shared_data_f32[idx] = val;
    workgroupBarrier();

    var stride = g.size / 2u;
    while (stride > 0u) {
        if (idx < stride) {
            cg_shared_data_f32[idx] = min(cg_shared_data_f32[idx], cg_shared_data_f32[idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return cg_shared_data_f32[0];
}

fn reduce_sum_i32(g: thread_block, val: i32) -> i32 {
    let idx = g.thread_rank;
    cg_shared_data_i32[idx] = val;
    workgroupBarrier();

    var stride = g.size / 2u;
    while (stride > 0u) {
        if (idx < stride) {
            cg_shared_data_i32[idx] = cg_shared_data_i32[idx] + cg_shared_data_i32[idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return cg_shared_data_i32[0];
}

// Inclusive scan within thread block
fn inclusive_scan_sum_f32(g: thread_block, val: f32) -> f32 {
    let idx = g.thread_rank;
    cg_shared_data_f32[idx] = val;
    workgroupBarrier();

    // Hillis-Steele scan
    var offset = 1u;
    while (offset < g.size) {
        var temp: f32 = 0.0;
        if (idx >= offset) {
            temp = cg_shared_data_f32[idx - offset];
        }
        workgroupBarrier();
        if (idx >= offset) {
            cg_shared_data_f32[idx] = cg_shared_data_f32[idx] + temp;
        }
        workgroupBarrier();
        offset = offset * 2u;
    }

    return cg_shared_data_f32[idx];
}

fn exclusive_scan_sum_f32(g: thread_block, val: f32) -> f32 {
    let inclusive = inclusive_scan_sum_f32(g, val);
    return inclusive - val;
}

// Broadcast value from a specific thread
fn shfl_f32(g: thread_block, val: f32, srcRank: u32) -> f32 {
    let idx = g.thread_rank;
    cg_shared_data_f32[idx] = val;
    workgroupBarrier();
    return cg_shared_data_f32[srcRank];
}

fn shfl_i32(g: thread_block, val: i32, srcRank: u32) -> i32 {
    let idx = g.thread_rank;
    cg_shared_data_i32[idx] = val;
    workgroupBarrier();
    return cg_shared_data_i32[srcRank];
}

fn shfl_u32(g: thread_block, val: u32, srcRank: u32) -> u32 {
    let idx = g.thread_rank;
    cg_shared_data_u32[idx] = val;
    workgroupBarrier();
    return cg_shared_data_u32[srcRank];
}

// All/any predicates
fn all_predicate(g: thread_block, pred: bool) -> bool {
    let idx = g.thread_rank;
    cg_shared_data_u32[idx] = select(0u, 1u, pred);
    workgroupBarrier();

    // Reduce with AND
    var stride = g.size / 2u;
    while (stride > 0u) {
        if (idx < stride) {
            cg_shared_data_u32[idx] = cg_shared_data_u32[idx] & cg_shared_data_u32[idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return cg_shared_data_u32[0] != 0u;
}

fn any_predicate(g: thread_block, pred: bool) -> bool {
    let idx = g.thread_rank;
    cg_shared_data_u32[idx] = select(0u, 1u, pred);
    workgroupBarrier();

    // Reduce with OR
    var stride = g.size / 2u;
    while (stride > 0u) {
        if (idx < stride) {
            cg_shared_data_u32[idx] = cg_shared_data_u32[idx] | cg_shared_data_u32[idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    return cg_shared_data_u32[0] != 0u;
}

// Ballot
fn ballot_predicate(g: thread_block, pred: bool) -> u32 {
    let idx = g.thread_rank;
    let lane = idx % 32u;
    let warp = idx / 32u;

    // Only supports up to 32 threads per ballot
    cg_shared_data_u32[warp] = 0u;
    workgroupBarrier();

    if (pred) {
        atomicOr(&cg_shared_data_u32[warp], 1u << lane);
    }
    workgroupBarrier();

    return cg_shared_data_u32[warp];
}

// ============================================================================
// Wait/Arrive Barrier (simplified)
// ============================================================================

var<workgroup> cg_barrier_count: atomic<u32>;

fn barrier_arrive(count: u32) {
    atomicAdd(&cg_barrier_count, 1u);
}

fn barrier_wait(count: u32) {
    while (atomicLoad(&cg_barrier_count) < count) {
        // Busy wait
    }
    workgroupBarrier();
}

fn barrier_arrive_and_wait(count: u32) {
    atomicAdd(&cg_barrier_count, 1u);
    while (atomicLoad(&cg_barrier_count) < count) {
        // Busy wait
    }
    workgroupBarrier();
}
`;

// TypeScript exports
export function withCooperativeGroups(shaderCode: string): string {
  return CUDA_COOPERATIVE_GROUPS_WGSL + '\n\n' + shaderCode;
}

export const COOPERATIVE_GROUPS_INFO = {
  groups: {
    thread_block: 'Workgroup-level group (all threads in workgroup)',
    grid_group: 'Grid-level group (all threads in dispatch) - limited sync support',
    thread_block_tile: 'Fixed-size partition of thread block (32, 16, 8, 4, 2, 1)',
    coalesced_group: 'Group of currently active threads',
    binary_partition: 'Binary split of a group based on predicate',
  },
  operations: {
    sync: 'Synchronize threads within group',
    size: 'Get number of threads in group',
    thread_rank: 'Get thread index within group',
    reduce: 'Collective reduction (sum, max, min)',
    scan: 'Prefix scan (inclusive/exclusive)',
    shfl: 'Exchange data between threads',
    vote: 'Collective voting (all, any, ballot)',
  },
  limitations: [
    'Grid-wide synchronization requires multiple dispatch calls',
    'Tile sync is approximated with workgroup barrier',
    'Maximum shared memory limits group collective operations',
  ],
};
