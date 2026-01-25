//! CUDAWeb WASM Module
//! CPU-intensive operations for Gaussian Splatting
//!
//! This module provides:
//! - Spatial data structures (KD-trees, octrees)
//! - Point cloud preprocessing
//! - Morton code computation
//! - Bounding volume hierarchy construction

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

pub mod gaussian;
pub mod spatial;
pub mod sort;
pub mod math;

#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better error messages
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Initialize thread pool for parallel operations
#[wasm_bindgen]
pub fn init_thread_pool(num_threads: usize) -> Result<(), JsValue> {
    // Note: wasm-bindgen-rayon handles thread pool initialization
    // This is a placeholder for when SharedArrayBuffer is available
    Ok(())
}

/// Get version info
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Memory management utilities
#[wasm_bindgen]
pub struct WasmMemory {
    // Internal memory tracking
}

#[wasm_bindgen]
impl WasmMemory {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Allocate a Float32Array in WASM memory
    #[wasm_bindgen]
    pub fn alloc_f32(&self, size: usize) -> js_sys::Float32Array {
        let vec = vec![0.0f32; size];
        js_sys::Float32Array::from(vec.as_slice())
    }

    /// Allocate a Uint32Array in WASM memory
    #[wasm_bindgen]
    pub fn alloc_u32(&self, size: usize) -> js_sys::Uint32Array {
        let vec = vec![0u32; size];
        js_sys::Uint32Array::from(vec.as_slice())
    }
}

impl Default for WasmMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Bounding box representation
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct BoundingBox {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
}

#[wasm_bindgen]
impl BoundingBox {
    #[wasm_bindgen(constructor)]
    pub fn new(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) -> Self {
        Self {
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
        }
    }

    #[wasm_bindgen]
    pub fn center(&self) -> Vec<f32> {
        vec![
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        ]
    }

    #[wasm_bindgen]
    pub fn size(&self) -> Vec<f32> {
        vec![
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        ]
    }

    #[wasm_bindgen]
    pub fn contains(&self, x: f32, y: f32, z: f32) -> bool {
        x >= self.min_x
            && x <= self.max_x
            && y >= self.min_y
            && y <= self.max_y
            && z >= self.min_z
            && z <= self.max_z
    }

    #[wasm_bindgen]
    pub fn expand(&mut self, x: f32, y: f32, z: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.min_z = self.min_z.min(z);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
        self.max_z = self.max_z.max(z);
    }
}

/// Compute bounding box from positions array
#[wasm_bindgen]
pub fn compute_bounding_box(positions: &[f32]) -> BoundingBox {
    let mut bbox = BoundingBox::new(
        f32::INFINITY,
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );

    for i in (0..positions.len()).step_by(3) {
        bbox.expand(positions[i], positions[i + 1], positions[i + 2]);
    }

    bbox
}

/// Performance measurement
#[wasm_bindgen]
pub struct PerfTimer {
    start: f64,
}

#[wasm_bindgen]
impl PerfTimer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            start: js_sys::Date::now(),
        }
    }

    #[wasm_bindgen]
    pub fn elapsed_ms(&self) -> f64 {
        js_sys::Date::now() - self.start
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.start = js_sys::Date::now();
    }
}

impl Default for PerfTimer {
    fn default() -> Self {
        Self::new()
    }
}
