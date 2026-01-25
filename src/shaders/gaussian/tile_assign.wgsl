// Tile Assignment Shader
// Assigns Gaussians to screen tiles for tile-based rendering
// Equivalent to CUDA tile assignment kernel

struct Gaussian2D {
    position: vec2<f32>,
    depth: f32,
    radius: f32,
    conic: vec3<f32>,
    opacity: f32,
    color: vec4<f32>,
};

struct TileUniforms {
    tile_size: u32,      // Usually 16
    num_tiles_x: u32,
    num_tiles_y: u32,
    num_gaussians: u32,
    width: u32,
    height: u32,
    _pad: vec2<u32>,
};

// Key-value pair for sorting: key = (tile_id << 32) | depth_bits
struct TileGaussianPair {
    tile_id: u32,
    gaussian_idx: u32,
    depth_key: u32,      // Depth encoded for sorting
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> gaussians_2d: array<Gaussian2D>;
@group(0) @binding(1) var<storage, read_write> tile_gaussian_pairs: array<TileGaussianPair>;
@group(0) @binding(2) var<storage, read_write> tile_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> total_pairs: atomic<u32>;
@group(0) @binding(4) var<uniform> params: TileUniforms;

// Convert float depth to sortable uint (preserves ordering)
fn depth_to_sort_key(depth: f32) -> u32 {
    let bits = bitcast<u32>(depth);
    // Flip sign bit and conditionally flip all bits to make sortable
    let mask = select(0x80000000u, 0xFFFFFFFFu, (bits & 0x80000000u) != 0u);
    return bits ^ mask;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_gaussians) {
        return;
    }

    let gaussian = gaussians_2d[idx];

    // Skip invalid Gaussians
    if (gaussian.opacity <= 0.0 || gaussian.radius <= 0.0) {
        return;
    }

    let pos = gaussian.position;
    let radius = gaussian.radius;

    // Compute tile bounds for this Gaussian
    let tile_size_f = f32(params.tile_size);
    let min_tile_x = max(0u, u32(floor((pos.x - radius) / tile_size_f)));
    let max_tile_x = min(params.num_tiles_x - 1u, u32(floor((pos.x + radius) / tile_size_f)));
    let min_tile_y = max(0u, u32(floor((pos.y - radius) / tile_size_f)));
    let max_tile_y = min(params.num_tiles_y - 1u, u32(floor((pos.y + radius) / tile_size_f)));

    let depth_key = depth_to_sort_key(gaussian.depth);

    // Assign to all overlapping tiles
    for (var ty = min_tile_y; ty <= max_tile_y; ty = ty + 1u) {
        for (var tx = min_tile_x; tx <= max_tile_x; tx = tx + 1u) {
            let tile_id = ty * params.num_tiles_x + tx;

            // Allocate slot in output array
            let pair_idx = atomicAdd(&total_pairs, 1u);

            // Increment tile count
            atomicAdd(&tile_counts[tile_id], 1u);

            // Write pair
            tile_gaussian_pairs[pair_idx] = TileGaussianPair(
                tile_id,
                idx,
                depth_key,
                0u
            );
        }
    }
}

// Second pass: compute tile offsets using prefix sum
@compute @workgroup_size(256)
fn compute_tile_offsets(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    // This would typically be done with a parallel prefix sum
    // For simplicity, showing the concept
    let idx = global_id.x;
    let total_tiles = params.num_tiles_x * params.num_tiles_y;

    if (idx >= total_tiles) {
        return;
    }

    // Prefix sum would be computed here
    // tile_offsets[idx] = sum of tile_counts[0..idx-1]
}
