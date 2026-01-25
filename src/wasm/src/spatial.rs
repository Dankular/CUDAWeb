//! Spatial data structures for efficient queries
//! - Morton codes for Z-order sorting
//! - Octree construction
//! - KD-tree for nearest neighbor queries

use wasm_bindgen::prelude::*;

/// Compute Morton code (Z-order curve) for 3D coordinates
/// Used for spatial locality in GPU processing
#[wasm_bindgen]
pub fn compute_morton_codes(
    positions: &[f32],
    min_x: f32,
    min_y: f32,
    min_z: f32,
    max_x: f32,
    max_y: f32,
    max_z: f32,
    output: &mut [u32],
) {
    let scale_x = if max_x > min_x {
        1023.0 / (max_x - min_x)
    } else {
        0.0
    };
    let scale_y = if max_y > min_y {
        1023.0 / (max_y - min_y)
    } else {
        0.0
    };
    let scale_z = if max_z > min_z {
        1023.0 / (max_z - min_z)
    } else {
        0.0
    };

    let num_points = positions.len() / 3;

    for i in 0..num_points {
        let x = ((positions[i * 3] - min_x) * scale_x).clamp(0.0, 1023.0) as u32;
        let y = ((positions[i * 3 + 1] - min_y) * scale_y).clamp(0.0, 1023.0) as u32;
        let z = ((positions[i * 3 + 2] - min_z) * scale_z).clamp(0.0, 1023.0) as u32;

        output[i] = morton_encode_3d(x, y, z);
    }
}

/// Encode 3D coordinates into Morton code (30-bit)
fn morton_encode_3d(x: u32, y: u32, z: u32) -> u32 {
    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

/// Expand 10-bit integer to 30 bits by inserting 2 zeros between each bit
fn expand_bits(v: u32) -> u32 {
    let mut v = v & 0x3FF; // Take only 10 bits
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

/// Decode Morton code back to 3D coordinates
#[wasm_bindgen]
pub fn morton_decode(code: u32) -> Vec<u32> {
    vec![
        compact_bits(code),
        compact_bits(code >> 1),
        compact_bits(code >> 2),
    ]
}

fn compact_bits(v: u32) -> u32 {
    let mut v = v & 0x09249249;
    v = (v | (v >> 2)) & 0x030C30C3;
    v = (v | (v >> 4)) & 0x0300F00F;
    v = (v | (v >> 8)) & 0x030000FF;
    v = (v | (v >> 16)) & 0x000003FF;
    v
}

/// Simple octree node for hierarchical culling
#[derive(Clone, Debug)]
pub struct OctreeNode {
    pub center: [f32; 3],
    pub half_size: f32,
    pub children: [Option<Box<OctreeNode>>; 8],
    pub point_indices: Vec<u32>,
    pub is_leaf: bool,
}

impl OctreeNode {
    pub fn new(center: [f32; 3], half_size: f32) -> Self {
        Self {
            center,
            half_size,
            children: Default::default(),
            point_indices: Vec::new(),
            is_leaf: true,
        }
    }

    fn get_child_index(&self, point: [f32; 3]) -> usize {
        let mut index = 0;
        if point[0] >= self.center[0] {
            index |= 1;
        }
        if point[1] >= self.center[1] {
            index |= 2;
        }
        if point[2] >= self.center[2] {
            index |= 4;
        }
        index
    }
}

/// Octree builder for spatial organization
#[wasm_bindgen]
pub struct Octree {
    root: Option<OctreeNode>,
    max_points_per_node: usize,
    max_depth: usize,
}

#[wasm_bindgen]
impl Octree {
    #[wasm_bindgen(constructor)]
    pub fn new(max_points_per_node: usize, max_depth: usize) -> Self {
        Self {
            root: None,
            max_points_per_node,
            max_depth,
        }
    }

    /// Build octree from positions
    #[wasm_bindgen]
    pub fn build(
        &mut self,
        positions: &[f32],
        min_x: f32,
        min_y: f32,
        min_z: f32,
        max_x: f32,
        max_y: f32,
        max_z: f32,
    ) {
        let center = [
            (min_x + max_x) * 0.5,
            (min_y + max_y) * 0.5,
            (min_z + max_z) * 0.5,
        ];

        let half_size = ((max_x - min_x).max(max_y - min_y).max(max_z - min_z)) * 0.5;

        let mut root = OctreeNode::new(center, half_size);

        // Insert all points
        let num_points = positions.len() / 3;
        for i in 0..num_points {
            let point = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
            self.insert_point(&mut root, point, i as u32, 0);
        }

        self.root = Some(root);
    }

    fn insert_point(&self, node: &mut OctreeNode, point: [f32; 3], index: u32, depth: usize) {
        if node.is_leaf {
            node.point_indices.push(index);

            // Split if necessary
            if node.point_indices.len() > self.max_points_per_node && depth < self.max_depth {
                self.split_node(node, depth);
            }
        } else {
            let child_idx = node.get_child_index(point);
            if let Some(ref mut child) = node.children[child_idx] {
                self.insert_point(child, point, index, depth + 1);
            }
        }
    }

    fn split_node(&self, node: &mut OctreeNode, _depth: usize) {
        node.is_leaf = false;
        let new_half_size = node.half_size * 0.5;

        // Create children
        for i in 0..8 {
            let offset_x = if i & 1 != 0 { new_half_size } else { -new_half_size };
            let offset_y = if i & 2 != 0 { new_half_size } else { -new_half_size };
            let offset_z = if i & 4 != 0 { new_half_size } else { -new_half_size };

            let child_center = [
                node.center[0] + offset_x,
                node.center[1] + offset_y,
                node.center[2] + offset_z,
            ];

            node.children[i] = Some(Box::new(OctreeNode::new(child_center, new_half_size)));
        }

        // Note: In a full implementation, we would redistribute points to children
    }

    /// Get total number of nodes
    #[wasm_bindgen]
    pub fn node_count(&self) -> usize {
        self.root.as_ref().map_or(0, |r| self.count_nodes(r))
    }

    fn count_nodes(&self, node: &OctreeNode) -> usize {
        let mut count = 1;
        for child in &node.children {
            if let Some(ref c) = child {
                count += self.count_nodes(c);
            }
        }
        count
    }

    /// Query points within a bounding box
    #[wasm_bindgen]
    pub fn query_box(
        &self,
        min_x: f32,
        min_y: f32,
        min_z: f32,
        max_x: f32,
        max_y: f32,
        max_z: f32,
    ) -> Vec<u32> {
        let mut result = Vec::new();

        if let Some(ref root) = self.root {
            self.query_box_recursive(root, min_x, min_y, min_z, max_x, max_y, max_z, &mut result);
        }

        result
    }

    fn query_box_recursive(
        &self,
        node: &OctreeNode,
        min_x: f32,
        min_y: f32,
        min_z: f32,
        max_x: f32,
        max_y: f32,
        max_z: f32,
        result: &mut Vec<u32>,
    ) {
        // Check if node intersects query box
        let node_min_x = node.center[0] - node.half_size;
        let node_max_x = node.center[0] + node.half_size;
        let node_min_y = node.center[1] - node.half_size;
        let node_max_y = node.center[1] + node.half_size;
        let node_min_z = node.center[2] - node.half_size;
        let node_max_z = node.center[2] + node.half_size;

        if node_max_x < min_x
            || node_min_x > max_x
            || node_max_y < min_y
            || node_min_y > max_y
            || node_max_z < min_z
            || node_min_z > max_z
        {
            return; // No intersection
        }

        if node.is_leaf {
            // Add all points (in full implementation, would filter by actual position)
            result.extend(&node.point_indices);
        } else {
            for child in &node.children {
                if let Some(ref c) = child {
                    self.query_box_recursive(c, min_x, min_y, min_z, max_x, max_y, max_z, result);
                }
            }
        }
    }
}

/// Compute tile assignments for screen-space tiling
#[wasm_bindgen]
pub fn compute_tile_assignments(
    screen_positions: &[f32], // [x, y, radius] per Gaussian
    tile_size: u32,
    screen_width: u32,
    screen_height: u32,
) -> Vec<u32> {
    let num_tiles_x = (screen_width + tile_size - 1) / tile_size;
    let num_tiles_y = (screen_height + tile_size - 1) / tile_size;
    let num_gaussians = screen_positions.len() / 3;

    // First pass: count assignments per Gaussian
    let mut total_assignments = 0;
    let mut assignments_per_gaussian = vec![0u32; num_gaussians];

    for i in 0..num_gaussians {
        let x = screen_positions[i * 3];
        let y = screen_positions[i * 3 + 1];
        let radius = screen_positions[i * 3 + 2];

        let min_tile_x = ((x - radius).max(0.0) / tile_size as f32).floor() as u32;
        let max_tile_x = ((x + radius).min(screen_width as f32) / tile_size as f32).floor() as u32;
        let min_tile_y = ((y - radius).max(0.0) / tile_size as f32).floor() as u32;
        let max_tile_y = ((y + radius).min(screen_height as f32) / tile_size as f32).floor() as u32;

        let count = (max_tile_x - min_tile_x + 1) * (max_tile_y - min_tile_y + 1);
        assignments_per_gaussian[i] = count;
        total_assignments += count;
    }

    // Second pass: compute offsets and fill assignments
    // Format: [tile_id, gaussian_idx, tile_id, gaussian_idx, ...]
    let mut result = vec![0u32; (total_assignments * 2) as usize];
    let mut offset = 0;

    for i in 0..num_gaussians {
        let x = screen_positions[i * 3];
        let y = screen_positions[i * 3 + 1];
        let radius = screen_positions[i * 3 + 2];

        let min_tile_x = ((x - radius).max(0.0) / tile_size as f32).floor() as u32;
        let max_tile_x = ((x + radius).min(screen_width as f32) / tile_size as f32).floor() as u32;
        let min_tile_y = ((y - radius).max(0.0) / tile_size as f32).floor() as u32;
        let max_tile_y = ((y + radius).min(screen_height as f32) / tile_size as f32).floor() as u32;

        for ty in min_tile_y..=max_tile_y.min(num_tiles_y - 1) {
            for tx in min_tile_x..=max_tile_x.min(num_tiles_x - 1) {
                let tile_id = ty * num_tiles_x + tx;
                result[offset] = tile_id;
                result[offset + 1] = i as u32;
                offset += 2;
            }
        }
    }

    result
}
