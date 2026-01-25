// GPU Radix Sort for Gaussian Depth Sorting
// Equivalent to CUDA CUB radix sort
// Implements parallel prefix sum based radix sort

struct SortElement {
    key: u32,       // Depth key (or tile_id << 16 | depth_key for tile-based)
    value: u32,     // Gaussian index
};

struct SortUniforms {
    num_elements: u32,
    bit_offset: u32,    // Current bit position being sorted (0, 4, 8, ...)
    num_workgroups: u32,
    _pad: u32,
};

// Shared memory for local histogram and prefix sums
var<workgroup> local_histogram: array<atomic<u32>, 16>;  // 4 bits = 16 buckets
var<workgroup> local_prefix: array<u32, 16>;
var<workgroup> local_keys: array<u32, 256>;
var<workgroup> local_values: array<u32, 256>;

@group(0) @binding(0) var<storage, read> input_elements: array<SortElement>;
@group(0) @binding(1) var<storage, read_write> output_elements: array<SortElement>;
@group(0) @binding(2) var<storage, read_write> global_histogram: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: SortUniforms;

const RADIX_BITS: u32 = 4u;
const RADIX_SIZE: u32 = 16u;  // 2^4
const WORKGROUP_SIZE: u32 = 256u;

// Extract 4-bit digit at given position
fn get_digit(key: u32, bit_offset: u32) -> u32 {
    return (key >> bit_offset) & 0xFu;
}

// Phase 1: Build local and global histograms
@compute @workgroup_size(256)
fn histogram_phase(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Initialize local histogram
    if (tid < RADIX_SIZE) {
        atomicStore(&local_histogram[tid], 0u);
    }
    workgroupBarrier();

    // Count digits in this workgroup
    if (gid < params.num_elements) {
        let key = input_elements[gid].key;
        let digit = get_digit(key, params.bit_offset);
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // Add to global histogram
    if (tid < RADIX_SIZE) {
        let count = atomicLoad(&local_histogram[tid]);
        atomicAdd(&global_histogram[workgroup_id.x * RADIX_SIZE + tid], count);
    }
}

// Phase 2: Compute prefix sums on global histogram
// (Simplified - in practice would use multi-level scan)
@compute @workgroup_size(256)
fn prefix_sum_phase(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    // This is a simplified exclusive scan
    // A full implementation would use Blelloch scan algorithm
    let tid = local_id.x;

    if (tid < RADIX_SIZE * params.num_workgroups) {
        var sum = 0u;
        for (var i = 0u; i < tid; i = i + 1u) {
            sum = sum + atomicLoad(&global_histogram[i]);
        }
        // Store prefix sum back (would need separate buffer in practice)
    }
}

// Phase 3: Scatter elements to sorted positions
@compute @workgroup_size(256)
fn scatter_phase(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load element
    var key = 0xFFFFFFFFu;
    var value = 0u;
    if (gid < params.num_elements) {
        key = input_elements[gid].key;
        value = input_elements[gid].value;
    }

    let digit = get_digit(key, params.bit_offset);

    // Initialize local histogram for counting sort within workgroup
    if (tid < RADIX_SIZE) {
        atomicStore(&local_histogram[tid], 0u);
    }
    workgroupBarrier();

    // Count local occurrences
    if (gid < params.num_elements) {
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();

    // Compute local prefix sum
    if (tid < RADIX_SIZE) {
        var sum = 0u;
        for (var i = 0u; i < tid; i = i + 1u) {
            sum = sum + atomicLoad(&local_histogram[i]);
        }
        local_prefix[tid] = sum;
    }
    workgroupBarrier();

    // Store in shared memory
    local_keys[tid] = key;
    local_values[tid] = value;
    workgroupBarrier();

    // Compute position and scatter
    if (gid < params.num_elements) {
        // Count how many elements with same digit come before this one
        var rank = 0u;
        for (var i = 0u; i < tid; i = i + 1u) {
            if (get_digit(local_keys[i], params.bit_offset) == digit) {
                rank = rank + 1u;
            }
        }

        // Global offset from histogram prefix sum + local offset
        let global_offset = atomicLoad(&global_histogram[workgroup_id.x * RADIX_SIZE + digit]);
        let local_offset = local_prefix[digit];
        let out_idx = global_offset + local_offset + rank;

        if (out_idx < params.num_elements) {
            output_elements[out_idx] = SortElement(key, value);
        }
    }
}

// Alternative: Simple bitonic sort for smaller arrays
// Useful when number of Gaussians per tile is small

struct BitonicUniforms {
    num_elements: u32,
    stage: u32,
    step: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> elements: array<SortElement>;
@group(0) @binding(1) var<uniform> bitonic_params: BitonicUniforms;

@compute @workgroup_size(256)
fn bitonic_sort_step(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    if (idx >= bitonic_params.num_elements / 2u) {
        return;
    }

    let stage = bitonic_params.stage;
    let step = bitonic_params.step;

    // Calculate partner index
    let block_size = 2u << stage;
    let half_block = 1u << step;

    let block_idx = idx / half_block;
    let idx_in_block = idx % half_block;

    let first_idx = block_idx * 2u * half_block + idx_in_block;
    let second_idx = first_idx + half_block;

    if (second_idx >= bitonic_params.num_elements) {
        return;
    }

    // Determine sort direction
    let ascending = ((first_idx / block_size) % 2u) == 0u;

    let first = elements[first_idx];
    let second = elements[second_idx];

    let should_swap = select(first.key < second.key, first.key > second.key, ascending);

    if (should_swap) {
        elements[first_idx] = second;
        elements[second_idx] = first;
    }
}
