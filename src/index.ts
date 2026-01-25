/**
 * CUDAWeb - WebGPU Gaussian Splatting
 * Main entry point and public API
 */

// Core WebGPU infrastructure
export {
  GPUContext,
  checkWebGPUSupport,
  BufferManager,
  createComputePipeline,
  createRenderPipeline,
  type GPUContextOptions,
  type GPUContextInfo,
  type BufferAllocation,
  type ChunkedUploadOptions,
} from './core';

// Model loaders
export {
  WeightLoader,
  PLYLoader,
  SplatLoader,
  type WeightMetadata,
  type ModelManifest,
  type LoadProgress,
  type ProgressCallback,
  type WeightBuffer,
  type GaussianModelWeights,
  type GaussianData,
  type SplatData,
} from './loaders';

// Rendering
export {
  Camera,
  OrbitControls,
  GaussianRenderer,
  type CameraState,
  type GaussianRendererConfig,
  type GaussianBuffers,
  type RenderStats,
} from './renderer';

// Main application class
export { GaussianSplattingApp, type AppConfig } from './app';
