//! Sorting algorithms for Gaussian depth ordering
//! - Radix sort for depth keys
//! - Counting sort for tile assignment

use wasm_bindgen::prelude::*;

/// Radix sort key-value pairs by key
/// Used for sorting Gaussians by depth
#[wasm_bindgen]
pub fn radix_sort_u32(keys: &mut [u32], values: &mut [u32]) {
    let n = keys.len();
    if n <= 1 {
        return;
    }

    let mut keys_tmp = vec![0u32; n];
    let mut values_tmp = vec![0u32; n];

    // Process 8 bits at a time (4 passes for 32-bit keys)
    for shift in (0..32).step_by(8) {
        // Count occurrences
        let mut count = [0usize; 256];
        for &key in keys.iter() {
            let digit = ((key >> shift) & 0xFF) as usize;
            count[digit] += 1;
        }

        // Compute prefix sum
        let mut prefix = [0usize; 256];
        for i in 1..256 {
            prefix[i] = prefix[i - 1] + count[i - 1];
        }

        // Scatter to temporary arrays
        for i in 0..n {
            let digit = ((keys[i] >> shift) & 0xFF) as usize;
            let pos = prefix[digit];
            keys_tmp[pos] = keys[i];
            values_tmp[pos] = values[i];
            prefix[digit] += 1;
        }

        // Copy back
        keys.copy_from_slice(&keys_tmp);
        values.copy_from_slice(&values_tmp);
    }
}

/// Convert float depth to sortable uint (preserves ordering)
#[wasm_bindgen]
pub fn depth_to_sort_key(depth: f32) -> u32 {
    let bits = depth.to_bits();
    // Flip sign bit and conditionally flip all bits
    if (bits & 0x80000000) != 0 {
        !bits
    } else {
        bits ^ 0x80000000
    }
}

/// Convert sort key back to float depth
#[wasm_bindgen]
pub fn sort_key_to_depth(key: u32) -> f32 {
    let bits = if (key & 0x80000000) != 0 {
        key ^ 0x80000000
    } else {
        !key
    };
    f32::from_bits(bits)
}

/// Sort Gaussians by depth (front to back)
#[wasm_bindgen]
pub fn sort_by_depth(depths: &[f32], indices: &mut [u32]) {
    let n = depths.len();
    if n <= 1 {
        return;
    }

    // Convert depths to sortable keys
    let mut keys: Vec<u32> = depths.iter().map(|&d| depth_to_sort_key(d)).collect();

    // Initialize indices
    for (i, idx) in indices.iter_mut().enumerate() {
        *idx = i as u32;
    }

    // Sort
    radix_sort_u32(&mut keys, indices);
}

/// Counting sort for small key ranges (like tile IDs)
#[wasm_bindgen]
pub fn counting_sort(
    keys: &[u32],
    values: &[u32],
    max_key: u32,
    sorted_keys: &mut [u32],
    sorted_values: &mut [u32],
) {
    let n = keys.len();

    // Count occurrences
    let mut count = vec![0usize; max_key as usize + 1];
    for &key in keys {
        count[key as usize] += 1;
    }

    // Compute prefix sums
    let mut prefix = vec![0usize; max_key as usize + 2];
    for i in 0..=max_key as usize {
        prefix[i + 1] = prefix[i] + count[i];
    }

    // Scatter (stable sort - preserves order within same key)
    for i in 0..n {
        let key = keys[i] as usize;
        let pos = prefix[key];
        sorted_keys[pos] = keys[i];
        sorted_values[pos] = values[i];
        prefix[key] += 1;
    }
}

/// Compute prefix sums (exclusive scan)
#[wasm_bindgen]
pub fn prefix_sum(input: &[u32], output: &mut [u32]) {
    let mut sum = 0u32;
    for i in 0..input.len() {
        output[i] = sum;
        sum += input[i];
    }
}

/// Compute tile ranges from sorted tile assignments
/// Returns [start, count] pairs for each tile
#[wasm_bindgen]
pub fn compute_tile_ranges(
    sorted_tile_ids: &[u32],
    num_tiles: u32,
) -> Vec<u32> {
    let mut ranges = vec![0u32; num_tiles as usize * 2];

    if sorted_tile_ids.is_empty() {
        return ranges;
    }

    let mut current_tile = sorted_tile_ids[0];
    let mut start = 0;

    for i in 0..sorted_tile_ids.len() {
        if sorted_tile_ids[i] != current_tile {
            // End of previous tile
            ranges[current_tile as usize * 2] = start as u32;
            ranges[current_tile as usize * 2 + 1] = (i - start) as u32;

            // Start of new tile
            current_tile = sorted_tile_ids[i];
            start = i;
        }
    }

    // Handle last tile
    ranges[current_tile as usize * 2] = start as u32;
    ranges[current_tile as usize * 2 + 1] = (sorted_tile_ids.len() - start) as u32;

    ranges
}

/// Argsort: return indices that would sort the array
#[wasm_bindgen]
pub fn argsort_f32(values: &[f32]) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..values.len() as u32).collect();
    indices.sort_by(|&a, &b| {
        values[a as usize]
            .partial_cmp(&values[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// Parallel merge sort for large arrays (single-threaded fallback)
#[wasm_bindgen]
pub fn merge_sort_with_indices(values: &mut [f32], indices: &mut [u32]) {
    let n = values.len();
    if n <= 1 {
        return;
    }

    // Initialize indices
    for (i, idx) in indices.iter_mut().enumerate() {
        *idx = i as u32;
    }

    let mut tmp_values = vec![0.0f32; n];
    let mut tmp_indices = vec![0u32; n];

    merge_sort_impl(values, indices, &mut tmp_values, &mut tmp_indices, 0, n);
}

fn merge_sort_impl(
    values: &mut [f32],
    indices: &mut [u32],
    tmp_values: &mut [f32],
    tmp_indices: &mut [u32],
    start: usize,
    end: usize,
) {
    if end - start <= 1 {
        return;
    }

    let mid = start + (end - start) / 2;

    merge_sort_impl(values, indices, tmp_values, tmp_indices, start, mid);
    merge_sort_impl(values, indices, tmp_values, tmp_indices, mid, end);

    // Merge
    let mut i = start;
    let mut j = mid;
    let mut k = start;

    while i < mid && j < end {
        if values[i] <= values[j] {
            tmp_values[k] = values[i];
            tmp_indices[k] = indices[i];
            i += 1;
        } else {
            tmp_values[k] = values[j];
            tmp_indices[k] = indices[j];
            j += 1;
        }
        k += 1;
    }

    while i < mid {
        tmp_values[k] = values[i];
        tmp_indices[k] = indices[i];
        i += 1;
        k += 1;
    }

    while j < end {
        tmp_values[k] = values[j];
        tmp_indices[k] = indices[j];
        j += 1;
        k += 1;
    }

    values[start..end].copy_from_slice(&tmp_values[start..end]);
    indices[start..end].copy_from_slice(&tmp_indices[start..end]);
}
