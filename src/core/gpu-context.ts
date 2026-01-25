/**
 * WebGPU Context Manager
 * Handles device initialization, adapter selection, and resource management
 */

export interface GPUContextOptions {
  powerPreference?: GPUPowerPreference;
  requiredFeatures?: GPUFeatureName[];
  requiredLimits?: Record<string, number>;
}

export interface GPUContextInfo {
  vendor: string;
  architecture: string;
  maxBufferSize: number;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupsPerDimension: number;
  maxComputeInvocationsPerWorkgroup: number;
}

export class GPUContext {
  private static instance: GPUContext | null = null;

  public device!: GPUDevice;
  public adapter!: GPUAdapter;
  public context!: GPUCanvasContext;
  public format!: GPUTextureFormat;
  public info!: GPUContextInfo;

  private canvas: HTMLCanvasElement | null = null;
  private initialized = false;

  private constructor() {}

  static async create(options: GPUContextOptions = {}): Promise<GPUContext> {
    if (GPUContext.instance?.initialized) {
      return GPUContext.instance;
    }

    const instance = new GPUContext();
    await instance.initialize(options);
    GPUContext.instance = instance;
    return instance;
  }

  static getInstance(): GPUContext {
    if (!GPUContext.instance?.initialized) {
      throw new Error('GPUContext not initialized. Call GPUContext.create() first.');
    }
    return GPUContext.instance;
  }

  private async initialize(options: GPUContextOptions): Promise<void> {
    if (!navigator.gpu) {
      throw new Error('WebGPU is not supported in this browser');
    }

    // Request adapter with high-performance GPU preference
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: options.powerPreference ?? 'high-performance',
    });

    if (!adapter) {
      throw new Error('Failed to get WebGPU adapter');
    }

    this.adapter = adapter;

    // Get adapter info for debugging
    const adapterInfo = await adapter.requestAdapterInfo();

    // Define required limits for Gaussian Splatting
    const requiredLimits: Record<string, number> = {
      maxStorageBufferBindingSize: Math.min(
        1024 * 1024 * 1024, // 1GB target
        adapter.limits.maxStorageBufferBindingSize
      ),
      maxBufferSize: Math.min(
        2 * 1024 * 1024 * 1024, // 2GB target for large models
        adapter.limits.maxBufferSize
      ),
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxBindGroups: adapter.limits.maxBindGroups,
      ...options.requiredLimits,
    };

    // Required features for advanced compute operations
    const requiredFeatures: GPUFeatureName[] = [
      ...(adapter.features.has('timestamp-query') ? ['timestamp-query' as GPUFeatureName] : []),
      ...(adapter.features.has('shader-f16') ? ['shader-f16' as GPUFeatureName] : []),
      ...(options.requiredFeatures ?? []),
    ];

    // Request device with our requirements
    this.device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits,
    });

    // Set up error handling
    this.device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
      this.initialized = false;
      if (info.reason !== 'destroyed') {
        // Attempt to reinitialize
        this.initialize(options);
      }
    });

    this.device.onuncapturederror = (event) => {
      console.error('WebGPU uncaptured error:', event.error);
    };

    this.info = {
      vendor: adapterInfo.vendor,
      architecture: adapterInfo.architecture,
      maxBufferSize: this.device.limits.maxBufferSize,
      maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: this.device.limits.maxComputeWorkgroupsPerDimension,
      maxComputeInvocationsPerWorkgroup: this.device.limits.maxComputeInvocationsPerWorkgroup,
    };

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.initialized = true;

    console.log('WebGPU initialized:', this.info);
  }

  configureCanvas(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    this.context = canvas.getContext('webgpu') as GPUCanvasContext;

    if (!this.context) {
      throw new Error('Failed to get WebGPU canvas context');
    }

    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
  }

  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    return this.device.createBuffer(descriptor);
  }

  createTexture(descriptor: GPUTextureDescriptor): GPUTexture {
    return this.device.createTexture(descriptor);
  }

  createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor): GPUBindGroupLayout {
    return this.device.createBindGroupLayout(descriptor);
  }

  createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor): GPUPipelineLayout {
    return this.device.createPipelineLayout(descriptor);
  }

  createComputePipeline(descriptor: GPUComputePipelineDescriptor): GPUComputePipeline {
    return this.device.createComputePipeline(descriptor);
  }

  async createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor): Promise<GPUComputePipeline> {
    return this.device.createComputePipelineAsync(descriptor);
  }

  createRenderPipeline(descriptor: GPURenderPipelineDescriptor): GPURenderPipeline {
    return this.device.createRenderPipeline(descriptor);
  }

  createCommandEncoder(descriptor?: GPUCommandEncoderDescriptor): GPUCommandEncoder {
    return this.device.createCommandEncoder(descriptor);
  }

  createShaderModule(code: string, label?: string): GPUShaderModule {
    return this.device.createShaderModule({
      code,
      label,
    });
  }

  submit(commandBuffers: GPUCommandBuffer[]): void {
    this.device.queue.submit(commandBuffers);
  }

  writeBuffer(buffer: GPUBuffer, data: BufferSource | SharedArrayBuffer, bufferOffset = 0, dataOffset?: number, size?: number): void {
    this.device.queue.writeBuffer(buffer, bufferOffset, data, dataOffset, size);
  }

  writeTexture(
    destination: GPUImageCopyTexture,
    data: BufferSource | SharedArrayBuffer,
    dataLayout: GPUImageDataLayout,
    size: GPUExtent3DStrict
  ): void {
    this.device.queue.writeTexture(destination, data, dataLayout, size);
  }

  getCurrentTexture(): GPUTexture {
    if (!this.context) {
      throw new Error('Canvas not configured');
    }
    return this.context.getCurrentTexture();
  }

  getCanvasSize(): { width: number; height: number } {
    if (!this.canvas) {
      throw new Error('Canvas not configured');
    }
    return {
      width: this.canvas.width,
      height: this.canvas.height,
    };
  }

  destroy(): void {
    this.device?.destroy();
    this.initialized = false;
    GPUContext.instance = null;
  }
}

// Utility function to check WebGPU support
export async function checkWebGPUSupport(): Promise<{
  supported: boolean;
  reason?: string;
}> {
  if (!navigator.gpu) {
    return { supported: false, reason: 'WebGPU is not available in this browser' };
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return { supported: false, reason: 'No WebGPU adapter available' };
  }

  return { supported: true };
}
