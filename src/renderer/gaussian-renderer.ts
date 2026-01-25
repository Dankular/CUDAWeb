/**
 * Gaussian Splatting Renderer
 * Complete WebGPU rendering pipeline for 3D Gaussian Splatting
 */

import { GPUContext, BufferManager, createComputePipeline, createRenderPipeline } from '../core';
import { Camera } from './camera';

// Import shaders as strings (Vite handles this)
import preprocessShader from '../shaders/gaussian/preprocess.wgsl?raw';
import tileAssignShader from '../shaders/gaussian/tile_assign.wgsl?raw';
import radixSortShader from '../shaders/sorting/radix_sort.wgsl?raw';
import rasterShader from '../shaders/rasterize/gaussian_raster.wgsl?raw';
import renderShader from '../shaders/rasterize/render.wgsl?raw';

export interface GaussianRendererConfig {
  tileSize?: number;
  maxGaussians?: number;
  shDegree?: number;
  backgroundColor?: [number, number, number];
}

export interface GaussianBuffers {
  positions: GPUBuffer;
  scales: GPUBuffer;
  rotations: GPUBuffer;
  opacities: GPUBuffer;
  shCoefficients: GPUBuffer;
}

export interface RenderStats {
  numGaussians: number;
  visibleGaussians: number;
  preprocessTimeMs: number;
  sortTimeMs: number;
  rasterTimeMs: number;
  totalTimeMs: number;
}

export class GaussianRenderer {
  private gpu: GPUContext;
  private bufferManager: BufferManager;
  private camera: Camera;

  // Configuration
  private tileSize: number = 16;
  private maxGaussians: number = 0;
  private shDegree: number = 3;
  private backgroundColor: [number, number, number] = [1, 1, 1];

  // Dimensions
  private width: number = 0;
  private height: number = 0;
  private numTilesX: number = 0;
  private numTilesY: number = 0;

  // Pipelines
  private preprocessPipeline!: GPUComputePipeline;
  private preprocessBindGroupLayout!: GPUBindGroupLayout;

  private tileAssignPipeline!: GPUComputePipeline;
  private tileAssignBindGroupLayout!: GPUBindGroupLayout;

  private sortPipeline!: GPUComputePipeline;
  private sortBindGroupLayout!: GPUBindGroupLayout;

  private rasterPipeline!: GPUComputePipeline;
  private rasterBindGroupLayout!: GPUBindGroupLayout;

  private renderPipeline!: GPURenderPipeline;
  private renderBindGroupLayout!: GPUBindGroupLayout;

  // Buffers
  private gaussianBuffers: GaussianBuffers | null = null;
  private gaussian2DBuffer!: GPUBuffer;
  private visibleIndicesBuffer!: GPUBuffer;
  private visibleCountBuffer!: GPUBuffer;
  private tileGaussianPairsBuffer!: GPUBuffer;
  private tileCountsBuffer!: GPUBuffer;
  private totalPairsBuffer!: GPUBuffer;
  private sortedIndicesBuffer!: GPUBuffer;
  private tileRangesBuffer!: GPUBuffer;
  private cameraUniformBuffer!: GPUBuffer;
  private paramsUniformBuffer!: GPUBuffer;
  private rasterUniformBuffer!: GPUBuffer;

  // Output
  private outputTexture!: GPUTexture;
  private outputTextureView!: GPUTextureView;
  private depthTexture!: GPUTexture;
  private depthTextureView!: GPUTextureView;
  private sampler!: GPUSampler;

  // Stats
  private lastStats: RenderStats = {
    numGaussians: 0,
    visibleGaussians: 0,
    preprocessTimeMs: 0,
    sortTimeMs: 0,
    rasterTimeMs: 0,
    totalTimeMs: 0,
  };

  private initialized = false;

  constructor(gpu: GPUContext, config: GaussianRendererConfig = {}) {
    this.gpu = gpu;
    this.bufferManager = new BufferManager(gpu);
    this.camera = new Camera();

    if (config.tileSize) this.tileSize = config.tileSize;
    if (config.maxGaussians) this.maxGaussians = config.maxGaussians;
    if (config.shDegree) this.shDegree = config.shDegree;
    if (config.backgroundColor) this.backgroundColor = config.backgroundColor;
  }

  async initialize(width: number, height: number): Promise<void> {
    this.width = width;
    this.height = height;
    this.numTilesX = Math.ceil(width / this.tileSize);
    this.numTilesY = Math.ceil(height / this.tileSize);

    this.camera.setAspect(width / height);
    this.camera.setFocalFromDimensions(width, height);

    await this.createPipelines();
    this.createBuffers();
    this.createTextures();

    this.initialized = true;
  }

  private async createPipelines(): Promise<void> {
    // Preprocess pipeline
    const preprocess = createComputePipeline(this.gpu)
      .setShader(preprocessShader, 'main')
      .setLabel('preprocess')
      .addStorageBuffer(0, true)   // gaussians_3d (read-only)
      .addStorageBuffer(1, true)   // sh_coefficients (read-only)
      .addStorageBuffer(2)          // gaussians_2d
      .addStorageBuffer(3)          // visible_indices
      .addStorageBuffer(4)          // visible_count
      .addUniformBuffer(5)          // camera
      .addUniformBuffer(6)          // params
      .build();
    this.preprocessPipeline = preprocess.pipeline;
    this.preprocessBindGroupLayout = preprocess.bindGroupLayout;

    // Tile assignment pipeline
    const tileAssign = createComputePipeline(this.gpu)
      .setShader(tileAssignShader, 'main')
      .setLabel('tile-assign')
      .addStorageBuffer(0, true)   // gaussians_2d
      .addStorageBuffer(1)          // tile_gaussian_pairs
      .addStorageBuffer(2)          // tile_counts
      .addStorageBuffer(3)          // total_pairs
      .addUniformBuffer(4)          // params
      .build();
    this.tileAssignPipeline = tileAssign.pipeline;
    this.tileAssignBindGroupLayout = tileAssign.bindGroupLayout;

    // Sort pipeline (bitonic sort for simplicity)
    const sort = createComputePipeline(this.gpu)
      .setShader(radixSortShader, 'bitonic_sort_step')
      .setLabel('sort')
      .addStorageBuffer(0)          // elements
      .addUniformBuffer(1)          // params
      .build();
    this.sortPipeline = sort.pipeline;
    this.sortBindGroupLayout = sort.bindGroupLayout;

    // Rasterization pipeline
    const raster = createComputePipeline(this.gpu)
      .setShader(rasterShader, 'main')
      .setLabel('raster')
      .addStorageBuffer(0, true)   // gaussians_2d
      .addStorageBuffer(1, true)   // sorted_indices
      .addStorageBuffer(2, true)   // tile_ranges
      .addUniformBuffer(3)          // params
      .addStorageTexture(4, 'rgba8unorm', 'write-only')
      .build();
    this.rasterPipeline = raster.pipeline;
    this.rasterBindGroupLayout = raster.bindGroupLayout;

    // Final render pipeline
    const render = createRenderPipeline(this.gpu)
      .setShader(renderShader, 'vs_fullscreen')
      .setFragmentShader(renderShader, 'fs_fullscreen')
      .setLabel('render')
      .addTexture(0, GPUShaderStage.FRAGMENT)
      .addSampler(1, GPUShaderStage.FRAGMENT)
      .build();
    this.renderPipeline = render.pipeline;
    this.renderBindGroupLayout = render.bindGroupLayout;
  }

  private createBuffers(): void {
    const maxGaussians = this.maxGaussians || 1000000;
    const numTiles = this.numTilesX * this.numTilesY;

    // Gaussian 2D data (position, depth, radius, conic, opacity, color)
    // = 2 + 1 + 1 + 3 + 1 + 4 = 12 floats per Gaussian
    this.gaussian2DBuffer = this.bufferManager.createStorageBuffer(
      maxGaussians * 12 * 4,
      'gaussian-2d'
    );

    // Visible indices
    this.visibleIndicesBuffer = this.bufferManager.createStorageBuffer(
      maxGaussians * 4,
      'visible-indices'
    );

    // Visible count (single u32)
    this.visibleCountBuffer = this.bufferManager.createStorageBuffer(
      4,
      'visible-count'
    );

    // Tile-Gaussian pairs (worst case: each Gaussian touches many tiles)
    // Estimate 4 tiles average per Gaussian
    this.tileGaussianPairsBuffer = this.bufferManager.createStorageBuffer(
      maxGaussians * 4 * 4 * 4, // 4 tiles * 4 values per pair
      'tile-gaussian-pairs'
    );

    // Tile counts
    this.tileCountsBuffer = this.bufferManager.createStorageBuffer(
      numTiles * 4,
      'tile-counts'
    );

    // Total pairs count
    this.totalPairsBuffer = this.bufferManager.createStorageBuffer(
      4,
      'total-pairs'
    );

    // Sorted indices
    this.sortedIndicesBuffer = this.bufferManager.createStorageBuffer(
      maxGaussians * 4 * 4, // Same as pairs
      'sorted-indices'
    );

    // Tile ranges ([start, count] per tile)
    this.tileRangesBuffer = this.bufferManager.createStorageBuffer(
      numTiles * 2 * 4,
      'tile-ranges'
    );

    // Uniform buffers
    this.cameraUniformBuffer = this.bufferManager.createUniformBuffer(
      256, // Camera uniforms
      'camera-uniforms'
    );

    this.paramsUniformBuffer = this.bufferManager.createUniformBuffer(
      32,
      'params-uniforms'
    );

    this.rasterUniformBuffer = this.bufferManager.createUniformBuffer(
      64,
      'raster-uniforms'
    );

    this.sampler = this.gpu.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  private createTextures(): void {
    this.outputTexture = this.gpu.createTexture({
      size: { width: this.width, height: this.height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING |
             GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_SRC,
      label: 'output-texture',
    });
    this.outputTextureView = this.outputTexture.createView();

    this.depthTexture = this.gpu.createTexture({
      size: { width: this.width, height: this.height },
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'depth-texture',
    });
    this.depthTextureView = this.depthTexture.createView();
  }

  /**
   * Set Gaussian data buffers
   */
  setGaussians(buffers: GaussianBuffers, numGaussians: number): void {
    this.gaussianBuffers = buffers;
    this.maxGaussians = numGaussians;
    this.lastStats.numGaussians = numGaussians;
  }

  /**
   * Update camera
   */
  getCamera(): Camera {
    return this.camera;
  }

  /**
   * Render frame
   */
  render(): void {
    if (!this.initialized || !this.gaussianBuffers) {
      return;
    }

    const startTime = performance.now();

    // Update camera uniforms
    const cameraUniforms = this.camera.getUniforms(this.width, this.height);
    this.gpu.writeBuffer(this.cameraUniformBuffer, 0, cameraUniforms);

    // Update params uniforms
    const paramsUniforms = new Uint32Array([
      this.maxGaussians,
      this.shDegree,
    ]);
    const paramsUniformsFloat = new Float32Array(paramsUniforms.buffer);
    // Add scale modifier
    const fullParams = new Float32Array([
      ...new Float32Array(paramsUniforms.buffer),
      1.0, // scale_modifier
      0,   // padding
    ]);
    this.gpu.writeBuffer(this.paramsUniformBuffer, 0, fullParams);

    const encoder = this.gpu.createCommandEncoder();

    // Clear visible count
    this.bufferManager.clearBuffer(this.visibleCountBuffer);

    // Preprocess pass
    const preprocessBindGroup = this.gpu.device.createBindGroup({
      layout: this.preprocessBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.gaussianBuffers.positions } },
        { binding: 1, resource: { buffer: this.gaussianBuffers.shCoefficients } },
        { binding: 2, resource: { buffer: this.gaussian2DBuffer } },
        { binding: 3, resource: { buffer: this.visibleIndicesBuffer } },
        { binding: 4, resource: { buffer: this.visibleCountBuffer } },
        { binding: 5, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 6, resource: { buffer: this.paramsUniformBuffer } },
      ],
    });

    const preprocessPass = encoder.beginComputePass();
    preprocessPass.setPipeline(this.preprocessPipeline);
    preprocessPass.setBindGroup(0, preprocessBindGroup);
    preprocessPass.dispatchWorkgroups(Math.ceil(this.maxGaussians / 256));
    preprocessPass.end();

    // Rasterization pass (simplified - direct compute rasterization)
    const rasterUniforms = new Float32Array([
      this.tileSize,
      this.numTilesX,
      this.numTilesY,
      this.width,
      this.height,
      this.backgroundColor[0],
      this.backgroundColor[1],
      this.backgroundColor[2],
    ]);
    // Convert first 5 to u32
    const rasterUniformsU32 = new Uint32Array(rasterUniforms.buffer);
    rasterUniformsU32[0] = this.tileSize;
    rasterUniformsU32[1] = this.numTilesX;
    rasterUniformsU32[2] = this.numTilesY;
    rasterUniformsU32[3] = this.width;
    rasterUniformsU32[4] = this.height;
    this.gpu.writeBuffer(this.rasterUniformBuffer, 0, new Float32Array(rasterUniforms));

    const rasterBindGroup = this.gpu.device.createBindGroup({
      layout: this.rasterBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.gaussian2DBuffer } },
        { binding: 1, resource: { buffer: this.sortedIndicesBuffer } },
        { binding: 2, resource: { buffer: this.tileRangesBuffer } },
        { binding: 3, resource: { buffer: this.rasterUniformBuffer } },
        { binding: 4, resource: this.outputTextureView },
      ],
    });

    const rasterPass = encoder.beginComputePass();
    rasterPass.setPipeline(this.rasterPipeline);
    rasterPass.setBindGroup(0, rasterBindGroup);
    rasterPass.dispatchWorkgroups(this.numTilesX, this.numTilesY);
    rasterPass.end();

    // Final render to canvas
    const renderBindGroup = this.gpu.device.createBindGroup({
      layout: this.renderBindGroupLayout,
      entries: [
        { binding: 0, resource: this.outputTextureView },
        { binding: 1, resource: this.sampler },
      ],
    });

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: this.gpu.getCurrentTexture().createView(),
        clearValue: { r: this.backgroundColor[0], g: this.backgroundColor[1], b: this.backgroundColor[2], a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    };

    const renderPass = encoder.beginRenderPass(renderPassDescriptor);
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(3); // Fullscreen triangle
    renderPass.end();

    // Submit
    this.gpu.submit([encoder.finish()]);

    // Update stats
    this.lastStats.totalTimeMs = performance.now() - startTime;
  }

  /**
   * Resize renderer
   */
  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) {
      return;
    }

    this.width = width;
    this.height = height;
    this.numTilesX = Math.ceil(width / this.tileSize);
    this.numTilesY = Math.ceil(height / this.tileSize);

    this.camera.setAspect(width / height);
    this.camera.setFocalFromDimensions(width, height);

    // Recreate textures
    this.outputTexture?.destroy();
    this.depthTexture?.destroy();
    this.createTextures();

    // Recreate tile-related buffers
    const numTiles = this.numTilesX * this.numTilesY;
    this.bufferManager.destroyBuffer(this.tileCountsBuffer);
    this.tileCountsBuffer = this.bufferManager.createStorageBuffer(
      numTiles * 4,
      'tile-counts'
    );

    this.bufferManager.destroyBuffer(this.tileRangesBuffer);
    this.tileRangesBuffer = this.bufferManager.createStorageBuffer(
      numTiles * 2 * 4,
      'tile-ranges'
    );
  }

  /**
   * Get rendering statistics
   */
  getStats(): RenderStats {
    return { ...this.lastStats };
  }

  /**
   * Set background color
   */
  setBackgroundColor(r: number, g: number, b: number): void {
    this.backgroundColor = [r, g, b];
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.outputTexture?.destroy();
    this.depthTexture?.destroy();
    this.bufferManager.clearPools();
  }
}
