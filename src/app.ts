/**
 * Gaussian Splatting Application
 * High-level API for loading and rendering Gaussian Splatting models
 */

import { GPUContext, checkWebGPUSupport, BufferManager } from './core';
import { PLYLoader, SplatLoader, WeightLoader, type LoadProgress } from './loaders';
import { GaussianRenderer, Camera, OrbitControls, type RenderStats } from './renderer';

export interface AppConfig {
  canvas: HTMLCanvasElement;
  backgroundColor?: [number, number, number];
  tileSize?: number;
  shDegree?: number;
  autoResize?: boolean;
}

export class GaussianSplattingApp {
  // Core components
  private gpu!: GPUContext;
  private bufferManager!: BufferManager;
  private renderer!: GaussianRenderer;
  private controls!: OrbitControls;

  // Loaders
  private plyLoader!: PLYLoader;
  private splatLoader!: SplatLoader;
  private weightLoader!: WeightLoader;

  // Configuration
  private canvas: HTMLCanvasElement;
  private config: AppConfig;

  // State
  private initialized = false;
  private isRendering = false;
  private animationFrameId: number | null = null;
  private resizeObserver: ResizeObserver | null = null;

  // Model info
  private numGaussians = 0;

  // Callbacks
  private onLoadProgress?: (progress: LoadProgress) => void;
  private onRenderStats?: (stats: RenderStats) => void;
  private onError?: (error: Error) => void;

  constructor(config: AppConfig) {
    this.canvas = config.canvas;
    this.config = {
      backgroundColor: [1, 1, 1],
      tileSize: 16,
      shDegree: 3,
      autoResize: true,
      ...config,
    };
  }

  /**
   * Initialize the application
   */
  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    // Check WebGPU support
    const support = await checkWebGPUSupport();
    if (!support.supported) {
      throw new Error(`WebGPU not supported: ${support.reason}`);
    }

    // Initialize GPU context
    this.gpu = await GPUContext.create({
      powerPreference: 'high-performance',
    });
    this.gpu.configureCanvas(this.canvas);

    // Initialize managers
    this.bufferManager = new BufferManager(this.gpu);

    // Initialize loaders
    this.plyLoader = new PLYLoader(this.gpu, this.bufferManager);
    this.splatLoader = new SplatLoader(this.gpu, this.bufferManager);
    this.weightLoader = new WeightLoader(this.gpu, this.bufferManager);

    // Initialize renderer
    this.renderer = new GaussianRenderer(this.gpu, {
      tileSize: this.config.tileSize,
      shDegree: this.config.shDegree,
      backgroundColor: this.config.backgroundColor,
    });

    const { width, height } = this.getCanvasSize();
    await this.renderer.initialize(width, height);

    // Initialize controls
    this.controls = new OrbitControls(this.renderer.getCamera(), this.canvas);

    // Set up auto-resize
    if (this.config.autoResize) {
      this.setupResizeObserver();
    }

    this.initialized = true;
    console.log('Gaussian Splatting App initialized');
  }

  /**
   * Load a Gaussian Splatting model from URL
   */
  async loadModel(
    url: string,
    onProgress?: (progress: LoadProgress) => void
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    this.onLoadProgress = onProgress;

    try {
      const extension = url.split('.').pop()?.toLowerCase();

      let gaussianData;

      if (extension === 'ply') {
        // Load PLY format
        const data = await this.plyLoader.load(url, {
          onProgress: (loaded, total) => {
            this.onLoadProgress?.({
              loaded,
              total,
              phase: 'weights',
            });
          },
        });
        gaussianData = data;
        this.numGaussians = data.count;
      } else if (extension === 'splat' || extension === 'ksplat') {
        // Load splat format
        const data = await this.splatLoader.load(url, {
          onProgress: (loaded, total) => {
            this.onLoadProgress?.({
              loaded,
              total,
              phase: 'weights',
            });
          },
        });
        gaussianData = data;
        this.numGaussians = data.count;
      } else if (extension === 'json' || extension === 'bin') {
        // Load weight-based model
        const weights = await this.weightLoader.loadGaussianModel(url, (progress) => {
          this.onLoadProgress?.(progress);
        });
        this.numGaussians = weights.numGaussians;

        // Set buffers directly from weight loader
        this.renderer.setGaussians(
          {
            positions: weights.positions.buffer,
            scales: weights.scales.buffer,
            rotations: weights.rotations.buffer,
            opacities: weights.opacities.buffer,
            shCoefficients: weights.shCoefficients.buffer,
          },
          weights.numGaussians
        );

        this.onLoadProgress?.({
          loaded: 1,
          total: 1,
          phase: 'complete',
        });

        return;
      } else {
        throw new Error(`Unsupported file format: ${extension}`);
      }

      // Upload to GPU and set buffers
      const buffers = this.plyLoader.uploadToGPU(gaussianData);
      this.renderer.setGaussians(buffers, this.numGaussians);

      this.onLoadProgress?.({
        loaded: 1,
        total: 1,
        phase: 'complete',
      });

      // Auto-focus on model center
      if (gaussianData.positions && gaussianData.positions.length > 0) {
        this.autoFocus(gaussianData.positions);
      }

      console.log(`Loaded ${this.numGaussians} Gaussians`);
    } catch (error) {
      this.onError?.(error as Error);
      throw error;
    }
  }

  /**
   * Load model from File object
   */
  async loadModelFromFile(
    file: File,
    onProgress?: (progress: LoadProgress) => void
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    this.onLoadProgress = onProgress;

    try {
      const extension = file.name.split('.').pop()?.toLowerCase();

      let gaussianData;

      if (extension === 'ply') {
        gaussianData = await this.plyLoader.loadFromFile(file, {
          onProgress: (loaded, total) => {
            this.onLoadProgress?.({
              loaded,
              total,
              phase: 'weights',
            });
          },
        });
      } else if (extension === 'splat' || extension === 'ksplat') {
        gaussianData = await this.splatLoader.loadFromFile(file, {
          onProgress: (loaded, total) => {
            this.onLoadProgress?.({
              loaded,
              total,
              phase: 'weights',
            });
          },
        });
      } else {
        throw new Error(`Unsupported file format: ${extension}`);
      }

      this.numGaussians = gaussianData.count;

      // Upload to GPU and set buffers
      const buffers = this.plyLoader.uploadToGPU(gaussianData);
      this.renderer.setGaussians(buffers, this.numGaussians);

      this.onLoadProgress?.({
        loaded: 1,
        total: 1,
        phase: 'complete',
      });

      // Auto-focus on model center
      this.autoFocus(gaussianData.positions);

      console.log(`Loaded ${this.numGaussians} Gaussians from file`);
    } catch (error) {
      this.onError?.(error as Error);
      throw error;
    }
  }

  /**
   * Auto-focus camera on model center
   */
  private autoFocus(positions: Float32Array): void {
    // Compute bounding box
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < positions.length; i += 3) {
      minX = Math.min(minX, positions[i]);
      minY = Math.min(minY, positions[i + 1]);
      minZ = Math.min(minZ, positions[i + 2]);
      maxX = Math.max(maxX, positions[i]);
      maxY = Math.max(maxY, positions[i + 1]);
      maxZ = Math.max(maxZ, positions[i + 2]);
    }

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;

    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ);

    // Set camera to look at center from a reasonable distance
    const distance = maxSize * 1.5;
    this.controls.setTarget(centerX, centerY, centerZ);
    this.controls.setOrbit(Math.PI / 4, Math.PI / 3, distance);
  }

  /**
   * Start rendering loop
   */
  startRendering(): void {
    if (this.isRendering) {
      return;
    }

    this.isRendering = true;
    this.renderLoop();
  }

  /**
   * Stop rendering loop
   */
  stopRendering(): void {
    this.isRendering = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  /**
   * Render a single frame
   */
  renderFrame(): void {
    if (!this.initialized) {
      return;
    }

    this.renderer.render();

    if (this.onRenderStats) {
      this.onRenderStats(this.renderer.getStats());
    }
  }

  private renderLoop(): void {
    if (!this.isRendering) {
      return;
    }

    this.renderFrame();
    this.animationFrameId = requestAnimationFrame(() => this.renderLoop());
  }

  /**
   * Get camera for manual control
   */
  getCamera(): Camera {
    return this.renderer.getCamera();
  }

  /**
   * Get orbit controls
   */
  getControls(): OrbitControls {
    return this.controls;
  }

  /**
   * Set background color
   */
  setBackgroundColor(r: number, g: number, b: number): void {
    this.renderer.setBackgroundColor(r, g, b);
  }

  /**
   * Get current render statistics
   */
  getStats(): RenderStats {
    return this.renderer.getStats();
  }

  /**
   * Get number of loaded Gaussians
   */
  getNumGaussians(): number {
    return this.numGaussians;
  }

  /**
   * Set callback for load progress
   */
  onProgress(callback: (progress: LoadProgress) => void): void {
    this.onLoadProgress = callback;
  }

  /**
   * Set callback for render stats
   */
  onStats(callback: (stats: RenderStats) => void): void {
    this.onRenderStats = callback;
  }

  /**
   * Set error callback
   */
  onErrorCallback(callback: (error: Error) => void): void {
    this.onError = callback;
  }

  private getCanvasSize(): { width: number; height: number } {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    return {
      width: Math.floor(rect.width * dpr),
      height: Math.floor(rect.height * dpr),
    };
  }

  private setupResizeObserver(): void {
    this.resizeObserver = new ResizeObserver(() => {
      const { width, height } = this.getCanvasSize();
      this.canvas.width = width;
      this.canvas.height = height;
      this.renderer.resize(width, height);
    });

    this.resizeObserver.observe(this.canvas);
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stopRendering();

    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }

    this.controls?.dispose();
    this.renderer?.dispose();
    this.weightLoader?.dispose();
    this.bufferManager?.clearPools();
    this.gpu?.destroy();

    this.initialized = false;
  }
}

// Convenience function to create and initialize app
export async function createGaussianSplattingApp(
  canvas: HTMLCanvasElement,
  config?: Omit<AppConfig, 'canvas'>
): Promise<GaussianSplattingApp> {
  const app = new GaussianSplattingApp({ canvas, ...config });
  await app.initialize();
  return app;
}
