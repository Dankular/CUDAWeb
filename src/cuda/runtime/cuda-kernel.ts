/**
 * CUDA Kernel Launch API for WebGPU
 *
 * Provides CUDA-compatible kernel launch and management functions
 */

import { CudaError, cudaDevice } from './cuda-device';
import { CudaPointer, getBuffer } from './cuda-memory';
import { CudaStream, cudaStreamManager } from './cuda-stream';

export interface Dim3 {
  x: number;
  y: number;
  z: number;
}

export interface CudaKernel {
  id: number;
  name: string;
  shaderModule: GPUShaderModule;
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  workgroupSize: Dim3;
  sharedMemorySize: number;
}

export interface KernelLaunchConfig {
  gridDim: Dim3;
  blockDim: Dim3;
  sharedMemBytes?: number;
  stream?: CudaStream | null;
}

export interface KernelBinding {
  buffer?: CudaPointer | GPUBuffer;
  texture?: GPUTexture;
  sampler?: GPUSampler;
  offset?: number;
  size?: number;
}

export type KernelArg = CudaPointer | GPUBuffer | ArrayBuffer | number | Float32Array | Int32Array | Uint32Array;

class CudaKernelManager {
  private static instance: CudaKernelManager | null = null;
  private kernels: Map<number, CudaKernel> = new Map();
  private kernelsByName: Map<string, CudaKernel> = new Map();
  private nextKernelId: number = 1;
  private uniformBuffers: Map<number, GPUBuffer> = new Map();
  private shaderCache: Map<string, GPUShaderModule> = new Map();

  private constructor() {}

  static getInstance(): CudaKernelManager {
    if (!CudaKernelManager.instance) {
      CudaKernelManager.instance = new CudaKernelManager();
    }
    return CudaKernelManager.instance;
  }

  private getDevice(): GPUDevice | null {
    return cudaDevice.getDevice();
  }

  // Kernel creation

  async createKernel(
    name: string,
    wgslCode: string,
    workgroupSize: Dim3 = { x: 256, y: 1, z: 1 },
    bindings: GPUBindGroupLayoutEntry[] = []
  ): Promise<{ kernel: CudaKernel | null; error: CudaError }> {
    const device = this.getDevice();
    if (!device) {
      return { kernel: null, error: CudaError.cudaErrorInitializationError };
    }

    try {
      // Check shader cache
      let shaderModule = this.shaderCache.get(wgslCode);
      if (!shaderModule) {
        shaderModule = device.createShaderModule({
          code: wgslCode,
          label: name,
        });
        this.shaderCache.set(wgslCode, shaderModule);
      }

      // Create default bindings if none provided
      if (bindings.length === 0) {
        bindings = this.createDefaultBindings();
      }

      const bindGroupLayout = device.createBindGroupLayout({
        entries: bindings,
        label: `${name}_layout`,
      });

      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
        label: `${name}_pipeline_layout`,
      });

      const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
        label: name,
      });

      const kernel: CudaKernel = {
        id: this.nextKernelId++,
        name,
        shaderModule,
        pipeline,
        bindGroupLayout,
        workgroupSize,
        sharedMemorySize: 0,
      };

      this.kernels.set(kernel.id, kernel);
      this.kernelsByName.set(name, kernel);

      return { kernel, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('Kernel creation error:', error);
      return { kernel: null, error: CudaError.cudaErrorInvalidDeviceFunction };
    }
  }

  private createDefaultBindings(): GPUBindGroupLayoutEntry[] {
    // Default: 16 storage buffer bindings
    const bindings: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < 16; i++) {
      bindings.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      });
    }
    return bindings;
  }

  // Kernel launch

  async launchKernel(
    kernel: CudaKernel,
    config: KernelLaunchConfig,
    args: KernelArg[]
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) {
      return CudaError.cudaErrorInitializationError;
    }

    try {
      // Create bind group entries from arguments
      const entries: GPUBindGroupEntry[] = [];
      let bindingIndex = 0;

      for (const arg of args) {
        const entry = this.createBindGroupEntry(bindingIndex, arg, device);
        if (entry) {
          entries.push(entry);
          bindingIndex++;
        }
      }

      const bindGroup = device.createBindGroup({
        layout: kernel.bindGroupLayout,
        entries,
        label: `${kernel.name}_bindgroup`,
      });

      // Create and submit command buffer
      const commandEncoder = device.createCommandEncoder({
        label: `${kernel.name}_encoder`,
      });

      const computePass = commandEncoder.beginComputePass({
        label: `${kernel.name}_pass`,
      });

      computePass.setPipeline(kernel.pipeline);
      computePass.setBindGroup(0, bindGroup);

      // Calculate dispatch dimensions
      const workgroupsX = Math.ceil(config.gridDim.x);
      const workgroupsY = Math.ceil(config.gridDim.y);
      const workgroupsZ = Math.ceil(config.gridDim.z);

      computePass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
      computePass.end();

      const commandBuffer = commandEncoder.finish();

      // Submit to queue
      device.queue.submit([commandBuffer]);

      // Track in stream if provided
      if (config.stream) {
        cudaStreamManager.addCommandBuffer(config.stream, commandBuffer);
        cudaStreamManager.addPendingOperation(
          config.stream,
          device.queue.onSubmittedWorkDone()
        );
      }

      return CudaError.cudaSuccess;
    } catch (error) {
      console.error('Kernel launch error:', error);
      return CudaError.cudaErrorLaunchFailure;
    }
  }

  private createBindGroupEntry(
    binding: number,
    arg: KernelArg,
    device: GPUDevice
  ): GPUBindGroupEntry | null {
    // Handle CudaPointer
    if ((arg as CudaPointer).id !== undefined) {
      const buffer = getBuffer(arg as CudaPointer);
      if (buffer) {
        return {
          binding,
          resource: { buffer },
        };
      }
      return null;
    }

    // Handle GPUBuffer directly
    if ((arg as GPUBuffer).size !== undefined && (arg as GPUBuffer).usage !== undefined) {
      return {
        binding,
        resource: { buffer: arg as GPUBuffer },
      };
    }

    // Handle ArrayBuffer or TypedArray (create uniform buffer)
    if (arg instanceof ArrayBuffer || ArrayBuffer.isView(arg)) {
      const data = arg instanceof ArrayBuffer ? new Uint8Array(arg) : new Uint8Array(arg.buffer);
      const size = Math.ceil(data.byteLength / 4) * 4; // Align to 4 bytes

      const uniformBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: `uniform_${binding}`,
      });

      device.queue.writeBuffer(uniformBuffer, 0, data);
      this.uniformBuffers.set(binding, uniformBuffer);

      return {
        binding,
        resource: { buffer: uniformBuffer },
      };
    }

    // Handle scalar number (create small uniform buffer)
    if (typeof arg === 'number') {
      const data = new Float32Array([arg]);
      const uniformBuffer = device.createBuffer({
        size: 16, // Minimum size with padding
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: `scalar_${binding}`,
      });

      device.queue.writeBuffer(uniformBuffer, 0, data);
      this.uniformBuffers.set(binding, uniformBuffer);

      return {
        binding,
        resource: { buffer: uniformBuffer },
      };
    }

    return null;
  }

  // Convenient launch syntax similar to CUDA <<<>>>
  async launch(
    kernel: CudaKernel | string,
    gridDim: Dim3 | number,
    blockDim: Dim3 | number,
    args: KernelArg[],
    sharedMemBytes: number = 0,
    stream: CudaStream | null = null
  ): Promise<CudaError> {
    let k: CudaKernel | undefined;

    if (typeof kernel === 'string') {
      k = this.kernelsByName.get(kernel);
      if (!k) {
        return CudaError.cudaErrorInvalidDeviceFunction;
      }
    } else {
      k = kernel;
    }

    const grid = typeof gridDim === 'number'
      ? { x: gridDim, y: 1, z: 1 }
      : gridDim;
    const block = typeof blockDim === 'number'
      ? { x: blockDim, y: 1, z: 1 }
      : blockDim;

    return this.launchKernel(k, {
      gridDim: grid,
      blockDim: block,
      sharedMemBytes,
      stream,
    }, args);
  }

  // Get kernel by name
  getKernel(name: string): CudaKernel | null {
    return this.kernelsByName.get(name) || null;
  }

  // Kernel occupancy calculations

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    kernel: CudaKernel,
    blockSize: number,
    dynamicSMemSize: number = 0
  ): { numBlocks: number; error: CudaError } {
    // WebGPU doesn't expose this directly, estimate based on limits
    const device = this.getDevice();
    if (!device) {
      return { numBlocks: 0, error: CudaError.cudaErrorInitializationError };
    }

    const maxWorkgroupSize = device.limits.maxComputeInvocationsPerWorkgroup;
    const sharedMemPerBlock = device.limits.maxComputeWorkgroupStorageSize;

    // Estimate: how many blocks can run concurrently
    const blocksLimitedByThreads = Math.floor(maxWorkgroupSize / blockSize);
    const blocksLimitedBySharedMem =
      dynamicSMemSize > 0
        ? Math.floor(sharedMemPerBlock / dynamicSMemSize)
        : Infinity;

    const numBlocks = Math.min(blocksLimitedByThreads, blocksLimitedBySharedMem, 32);

    return { numBlocks: Math.max(1, numBlocks), error: CudaError.cudaSuccess };
  }

  cudaOccupancyMaxPotentialBlockSize(
    kernel: CudaKernel,
    dynamicSMemSize: number = 0
  ): { minGridSize: number; blockSize: number; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { minGridSize: 0, blockSize: 0, error: CudaError.cudaErrorInitializationError };
    }

    // Choose optimal block size based on limits
    const maxWorkgroupSize = device.limits.maxComputeInvocationsPerWorkgroup;

    // Common optimal sizes: 256, 512, 128
    let blockSize = 256;
    if (maxWorkgroupSize >= 512) {
      blockSize = 512;
    } else if (maxWorkgroupSize < 256) {
      blockSize = Math.min(128, maxWorkgroupSize);
    }

    // Estimate minimum grid size for full occupancy
    const { props } = cudaDevice.cudaGetDeviceProperties(0);
    const smCount = props?.multiProcessorCount || 1;
    const minGridSize = smCount * 2; // 2 blocks per SM

    return { minGridSize, blockSize, error: CudaError.cudaSuccess };
  }

  // Cooperative kernel launch

  async cudaLaunchCooperativeKernel(
    kernel: CudaKernel,
    config: KernelLaunchConfig,
    args: KernelArg[]
  ): Promise<CudaError> {
    // Cooperative launch is similar to regular launch in WebGPU
    // The difference is that all workgroups can synchronize
    return this.launchKernel(kernel, config, args);
  }

  // Multi-GPU cooperative launch (simulated)

  async cudaLaunchCooperativeKernelMultiDevice(
    launchParams: Array<{
      kernel: CudaKernel;
      config: KernelLaunchConfig;
      args: KernelArg[];
    }>
  ): Promise<CudaError> {
    // In WebGPU, we typically only have one device
    // Execute all kernels sequentially
    for (const params of launchParams) {
      const error = await this.launchKernel(params.kernel, params.config, params.args);
      if (error !== CudaError.cudaSuccess) {
        return error;
      }
    }
    return CudaError.cudaSuccess;
  }

  // Function attributes

  cudaFuncGetAttributes(kernel: CudaKernel): {
    attributes: CudaFuncAttributes;
    error: CudaError;
  } {
    const device = this.getDevice();
    if (!device) {
      return {
        attributes: this.getDefaultFuncAttributes(),
        error: CudaError.cudaErrorInitializationError,
      };
    }

    const attributes: CudaFuncAttributes = {
      binaryVersion: 80,
      cacheModeCA: false,
      constSizeBytes: 0,
      localSizeBytes: 0,
      maxDynamicSharedSizeBytes: device.limits.maxComputeWorkgroupStorageSize,
      maxThreadsPerBlock: device.limits.maxComputeInvocationsPerWorkgroup,
      numRegs: 32,
      preferredShmemCarveout: 0,
      ptxVersion: 80,
      sharedSizeBytes: kernel.sharedMemorySize,
    };

    return { attributes, error: CudaError.cudaSuccess };
  }

  private getDefaultFuncAttributes(): CudaFuncAttributes {
    return {
      binaryVersion: 80,
      cacheModeCA: false,
      constSizeBytes: 0,
      localSizeBytes: 0,
      maxDynamicSharedSizeBytes: 49152,
      maxThreadsPerBlock: 256,
      numRegs: 32,
      preferredShmemCarveout: 0,
      ptxVersion: 80,
      sharedSizeBytes: 0,
    };
  }

  // Cleanup

  destroyKernel(kernel: CudaKernel): CudaError {
    this.kernels.delete(kernel.id);
    this.kernelsByName.delete(kernel.name);
    return CudaError.cudaSuccess;
  }

  clearAll(): void {
    for (const [, buffer] of this.uniformBuffers) {
      buffer.destroy();
    }
    this.kernels.clear();
    this.kernelsByName.clear();
    this.uniformBuffers.clear();
    this.shaderCache.clear();
  }
}

export interface CudaFuncAttributes {
  binaryVersion: number;
  cacheModeCA: boolean;
  constSizeBytes: number;
  localSizeBytes: number;
  maxDynamicSharedSizeBytes: number;
  maxThreadsPerBlock: number;
  numRegs: number;
  preferredShmemCarveout: number;
  ptxVersion: number;
  sharedSizeBytes: number;
}

// Export singleton
export const cudaKernelManager = CudaKernelManager.getInstance();

// Helper function for dim3 creation
export function dim3(x: number, y: number = 1, z: number = 1): Dim3 {
  return { x, y, z };
}

// Export CUDA-like API functions

export async function cudaCreateKernel(
  name: string,
  wgslCode: string,
  workgroupSize?: Dim3,
  bindings?: GPUBindGroupLayoutEntry[]
): Promise<{ kernel: CudaKernel | null; error: CudaError }> {
  return cudaKernelManager.createKernel(name, wgslCode, workgroupSize, bindings);
}

export async function cudaLaunchKernel(
  kernel: CudaKernel,
  gridDim: Dim3,
  blockDim: Dim3,
  args: KernelArg[],
  sharedMemBytes?: number,
  stream?: CudaStream | null
): Promise<CudaError> {
  return cudaKernelManager.launchKernel(
    kernel,
    { gridDim, blockDim, sharedMemBytes, stream },
    args
  );
}

export async function cudaLaunch(
  kernel: CudaKernel | string,
  gridDim: Dim3 | number,
  blockDim: Dim3 | number,
  args: KernelArg[],
  sharedMemBytes?: number,
  stream?: CudaStream | null
): Promise<CudaError> {
  return cudaKernelManager.launch(kernel, gridDim, blockDim, args, sharedMemBytes, stream);
}

export function cudaGetKernel(name: string): CudaKernel | null {
  return cudaKernelManager.getKernel(name);
}

export function cudaOccupancyMaxActiveBlocksPerMultiprocessor(
  kernel: CudaKernel,
  blockSize: number,
  dynamicSMemSize?: number
): { numBlocks: number; error: CudaError } {
  return cudaKernelManager.cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    kernel,
    blockSize,
    dynamicSMemSize
  );
}

export function cudaOccupancyMaxPotentialBlockSize(
  kernel: CudaKernel,
  dynamicSMemSize?: number
): { minGridSize: number; blockSize: number; error: CudaError } {
  return cudaKernelManager.cudaOccupancyMaxPotentialBlockSize(kernel, dynamicSMemSize);
}

export async function cudaLaunchCooperativeKernel(
  kernel: CudaKernel,
  gridDim: Dim3,
  blockDim: Dim3,
  args: KernelArg[],
  sharedMemBytes?: number,
  stream?: CudaStream | null
): Promise<CudaError> {
  return cudaKernelManager.cudaLaunchCooperativeKernel(
    kernel,
    { gridDim, blockDim, sharedMemBytes, stream },
    args
  );
}

export function cudaFuncGetAttributes(kernel: CudaKernel): {
  attributes: CudaFuncAttributes;
  error: CudaError;
} {
  return cudaKernelManager.cudaFuncGetAttributes(kernel);
}

export function cudaDestroyKernel(kernel: CudaKernel): CudaError {
  return cudaKernelManager.destroyKernel(kernel);
}
