/**
 * CUDA Memory Management API for WebGPU
 *
 * Provides CUDA-compatible memory allocation and transfer functions
 */

import { CudaDevice, CudaError, CudaMemcpyKind, cudaDevice } from './cuda-device';

export interface CudaPointer {
  id: number;
  buffer: GPUBuffer;
  size: number;
  type: 'device' | 'host' | 'managed' | 'pinned';
  offset: number;
}

export interface CudaArray {
  id: number;
  texture: GPUTexture;
  width: number;
  height: number;
  depth: number;
  format: GPUTextureFormat;
}

export interface CudaPitchedPtr {
  ptr: CudaPointer;
  pitch: number;
  xsize: number;
  ysize: number;
}

export interface CudaMemcpy3DParms {
  srcArray?: CudaArray;
  srcPos: { x: number; y: number; z: number };
  srcPtr?: CudaPitchedPtr;
  dstArray?: CudaArray;
  dstPos: { x: number; y: number; z: number };
  dstPtr?: CudaPitchedPtr;
  extent: { width: number; height: number; depth: number };
  kind: CudaMemcpyKind;
}

export enum CudaMemoryType {
  cudaMemoryTypeUnregistered = 0,
  cudaMemoryTypeHost = 1,
  cudaMemoryTypeDevice = 2,
  cudaMemoryTypeManaged = 3,
}

export interface CudaPointerAttributes {
  type: CudaMemoryType;
  device: number;
  devicePointer: CudaPointer | null;
  hostPointer: ArrayBuffer | null;
}

class CudaMemoryManager {
  private static instance: CudaMemoryManager | null = null;
  private pointers: Map<number, CudaPointer> = new Map();
  private hostBuffers: Map<number, ArrayBuffer> = new Map();
  private arrays: Map<number, CudaArray> = new Map();
  private nextPointerId: number = 1;
  private nextArrayId: number = 1;
  private totalAllocatedMemory: number = 0;
  private bufferPool: Map<number, GPUBuffer[]> = new Map(); // Size -> available buffers

  private constructor() {}

  static getInstance(): CudaMemoryManager {
    if (!CudaMemoryManager.instance) {
      CudaMemoryManager.instance = new CudaMemoryManager();
    }
    return CudaMemoryManager.instance;
  }

  private getDevice(): GPUDevice | null {
    return cudaDevice.getDevice();
  }

  private alignSize(size: number, alignment: number = 256): number {
    return Math.ceil(size / alignment) * alignment;
  }

  // Core allocation functions

  cudaMalloc(size: number): { ptr: CudaPointer | null; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { ptr: null, error: CudaError.cudaErrorInitializationError };
    }

    if (size <= 0) {
      return { ptr: null, error: CudaError.cudaErrorInvalidValue };
    }

    try {
      const alignedSize = this.alignSize(size);

      // Check buffer pool first
      const pooledBuffer = this.getPooledBuffer(alignedSize);
      let buffer: GPUBuffer;

      if (pooledBuffer) {
        buffer = pooledBuffer;
      } else {
        buffer = device.createBuffer({
          size: alignedSize,
          usage:
            GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_SRC |
            GPUBufferUsage.COPY_DST,
          mappedAtCreation: false,
        });
      }

      const ptr: CudaPointer = {
        id: this.nextPointerId++,
        buffer,
        size: alignedSize,
        type: 'device',
        offset: 0,
      };

      this.pointers.set(ptr.id, ptr);
      this.totalAllocatedMemory += alignedSize;

      return { ptr, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('cudaMalloc error:', error);
      return { ptr: null, error: CudaError.cudaErrorMemoryAllocation };
    }
  }

  cudaMallocPitch(
    width: number,
    height: number,
    elementSize: number
  ): { ptr: CudaPitchedPtr | null; error: CudaError } {
    // Calculate pitch (aligned row width)
    const minPitch = width * elementSize;
    const pitch = this.alignSize(minPitch, 256);
    const totalSize = pitch * height;

    const { ptr, error } = this.cudaMalloc(totalSize);
    if (error !== CudaError.cudaSuccess || !ptr) {
      return { ptr: null, error };
    }

    const pitchedPtr: CudaPitchedPtr = {
      ptr,
      pitch,
      xsize: width,
      ysize: height,
    };

    return { ptr: pitchedPtr, error: CudaError.cudaSuccess };
  }

  cudaMalloc3D(
    extent: { width: number; height: number; depth: number },
    elementSize: number
  ): { ptr: CudaPitchedPtr | null; error: CudaError } {
    const minPitch = extent.width * elementSize;
    const pitch = this.alignSize(minPitch, 256);
    const totalSize = pitch * extent.height * extent.depth;

    const { ptr, error } = this.cudaMalloc(totalSize);
    if (error !== CudaError.cudaSuccess || !ptr) {
      return { ptr: null, error };
    }

    const pitchedPtr: CudaPitchedPtr = {
      ptr,
      pitch,
      xsize: extent.width,
      ysize: extent.height,
    };

    return { ptr: pitchedPtr, error: CudaError.cudaSuccess };
  }

  cudaMallocHost(size: number): { ptr: CudaPointer | null; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { ptr: null, error: CudaError.cudaErrorInitializationError };
    }

    if (size <= 0) {
      return { ptr: null, error: CudaError.cudaErrorInvalidValue };
    }

    try {
      const alignedSize = this.alignSize(size);

      // Create mappable buffer for host access
      const buffer = device.createBuffer({
        size: alignedSize,
        usage:
          GPUBufferUsage.MAP_READ |
          GPUBufferUsage.MAP_WRITE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
      });

      const ptr: CudaPointer = {
        id: this.nextPointerId++,
        buffer,
        size: alignedSize,
        type: 'pinned',
        offset: 0,
      };

      this.pointers.set(ptr.id, ptr);
      this.totalAllocatedMemory += alignedSize;

      return { ptr, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('cudaMallocHost error:', error);
      return { ptr: null, error: CudaError.cudaErrorMemoryAllocation };
    }
  }

  cudaHostAlloc(
    size: number,
    flags: number = 0
  ): { ptr: CudaPointer | null; error: CudaError } {
    // In WebGPU, all host allocations are essentially pinned
    return this.cudaMallocHost(size);
  }

  cudaMallocManaged(size: number): { ptr: CudaPointer | null; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { ptr: null, error: CudaError.cudaErrorInitializationError };
    }

    if (size <= 0) {
      return { ptr: null, error: CudaError.cudaErrorInvalidValue };
    }

    try {
      const alignedSize = this.alignSize(size);

      // Managed memory: accessible from both CPU and GPU
      const buffer = device.createBuffer({
        size: alignedSize,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
      });

      // Also create host-side ArrayBuffer
      const hostBuffer = new ArrayBuffer(alignedSize);
      const hostId = this.nextPointerId;
      this.hostBuffers.set(hostId, hostBuffer);

      const ptr: CudaPointer = {
        id: this.nextPointerId++,
        buffer,
        size: alignedSize,
        type: 'managed',
        offset: 0,
      };

      this.pointers.set(ptr.id, ptr);
      this.totalAllocatedMemory += alignedSize;

      return { ptr, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('cudaMallocManaged error:', error);
      return { ptr: null, error: CudaError.cudaErrorMemoryAllocation };
    }
  }

  cudaFree(ptr: CudaPointer | null): CudaError {
    if (!ptr) {
      return CudaError.cudaSuccess; // cudaFree(nullptr) is valid
    }

    const pointer = this.pointers.get(ptr.id);
    if (!pointer) {
      return CudaError.cudaErrorInvalidValue;
    }

    // Return to pool or destroy
    const poolKey = pointer.size;
    if (pointer.type === 'device') {
      const pool = this.bufferPool.get(poolKey) || [];
      if (pool.length < 10) {
        // Keep up to 10 buffers of each size
        pool.push(pointer.buffer);
        this.bufferPool.set(poolKey, pool);
      } else {
        pointer.buffer.destroy();
      }
    } else {
      pointer.buffer.destroy();
    }

    this.pointers.delete(ptr.id);
    this.hostBuffers.delete(ptr.id);
    this.totalAllocatedMemory -= pointer.size;

    return CudaError.cudaSuccess;
  }

  cudaFreeHost(ptr: CudaPointer | null): CudaError {
    return this.cudaFree(ptr);
  }

  private getPooledBuffer(size: number): GPUBuffer | null {
    const pool = this.bufferPool.get(size);
    if (pool && pool.length > 0) {
      return pool.pop()!;
    }
    return null;
  }

  // Memory transfer functions

  async cudaMemcpy(
    dst: CudaPointer | ArrayBuffer,
    src: CudaPointer | ArrayBuffer,
    count: number,
    kind: CudaMemcpyKind
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) {
      return CudaError.cudaErrorInitializationError;
    }

    try {
      switch (kind) {
        case CudaMemcpyKind.cudaMemcpyHostToDevice:
          return await this.copyHostToDevice(
            dst as CudaPointer,
            src as ArrayBuffer,
            count
          );

        case CudaMemcpyKind.cudaMemcpyDeviceToHost:
          return await this.copyDeviceToHost(
            dst as ArrayBuffer,
            src as CudaPointer,
            count
          );

        case CudaMemcpyKind.cudaMemcpyDeviceToDevice:
          return await this.copyDeviceToDevice(
            dst as CudaPointer,
            src as CudaPointer,
            count
          );

        case CudaMemcpyKind.cudaMemcpyHostToHost:
          return this.copyHostToHost(
            dst as ArrayBuffer,
            src as ArrayBuffer,
            count
          );

        case CudaMemcpyKind.cudaMemcpyDefault:
          // Auto-detect based on pointer types
          return await this.cudaMemcpyDefault(dst, src, count);

        default:
          return CudaError.cudaErrorInvalidMemcpyDirection;
      }
    } catch (error) {
      console.error('cudaMemcpy error:', error);
      return CudaError.cudaErrorUnknown;
    }
  }

  private async copyHostToDevice(
    dst: CudaPointer,
    src: ArrayBuffer,
    count: number
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const pointer = this.pointers.get(dst.id);
    if (!pointer) return CudaError.cudaErrorInvalidValue;

    const srcData = new Uint8Array(src, 0, count);
    device.queue.writeBuffer(pointer.buffer, dst.offset, srcData);

    return CudaError.cudaSuccess;
  }

  private async copyDeviceToHost(
    dst: ArrayBuffer,
    src: CudaPointer,
    count: number
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const pointer = this.pointers.get(src.id);
    if (!pointer) return CudaError.cudaErrorInvalidValue;

    // Create staging buffer for readback
    const stagingBuffer = device.createBuffer({
      size: count,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      pointer.buffer,
      src.offset,
      stagingBuffer,
      0,
      count
    );
    device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const mappedData = new Uint8Array(stagingBuffer.getMappedRange());
    const dstArray = new Uint8Array(dst);
    dstArray.set(mappedData.subarray(0, count));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return CudaError.cudaSuccess;
  }

  private async copyDeviceToDevice(
    dst: CudaPointer,
    src: CudaPointer,
    count: number
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const srcPointer = this.pointers.get(src.id);
    const dstPointer = this.pointers.get(dst.id);
    if (!srcPointer || !dstPointer) return CudaError.cudaErrorInvalidValue;

    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      srcPointer.buffer,
      src.offset,
      dstPointer.buffer,
      dst.offset,
      count
    );
    device.queue.submit([commandEncoder.finish()]);

    return CudaError.cudaSuccess;
  }

  private copyHostToHost(
    dst: ArrayBuffer,
    src: ArrayBuffer,
    count: number
  ): CudaError {
    const srcArray = new Uint8Array(src, 0, count);
    const dstArray = new Uint8Array(dst);
    dstArray.set(srcArray);
    return CudaError.cudaSuccess;
  }

  private async cudaMemcpyDefault(
    dst: CudaPointer | ArrayBuffer,
    src: CudaPointer | ArrayBuffer,
    count: number
  ): Promise<CudaError> {
    const dstIsDevice = this.isDevicePointer(dst);
    const srcIsDevice = this.isDevicePointer(src);

    if (srcIsDevice && dstIsDevice) {
      return await this.copyDeviceToDevice(
        dst as CudaPointer,
        src as CudaPointer,
        count
      );
    } else if (!srcIsDevice && dstIsDevice) {
      return await this.copyHostToDevice(
        dst as CudaPointer,
        src as ArrayBuffer,
        count
      );
    } else if (srcIsDevice && !dstIsDevice) {
      return await this.copyDeviceToHost(
        dst as ArrayBuffer,
        src as CudaPointer,
        count
      );
    } else {
      return this.copyHostToHost(dst as ArrayBuffer, src as ArrayBuffer, count);
    }
  }

  private isDevicePointer(ptr: CudaPointer | ArrayBuffer): ptr is CudaPointer {
    return (ptr as CudaPointer).id !== undefined;
  }

  // Async memory operations

  cudaMemcpyAsync(
    dst: CudaPointer | ArrayBuffer,
    src: CudaPointer | ArrayBuffer,
    count: number,
    kind: CudaMemcpyKind,
    stream?: CudaStream
  ): Promise<CudaError> {
    // In WebGPU, all operations are inherently async
    return this.cudaMemcpy(dst, src, count, kind);
  }

  // Memory set operations

  async cudaMemset(
    devPtr: CudaPointer,
    value: number,
    count: number
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const pointer = this.pointers.get(devPtr.id);
    if (!pointer) return CudaError.cudaErrorInvalidValue;

    // Create host buffer with the value repeated
    const hostBuffer = new Uint8Array(count);
    hostBuffer.fill(value & 0xff);

    device.queue.writeBuffer(pointer.buffer, devPtr.offset, hostBuffer);
    return CudaError.cudaSuccess;
  }

  async cudaMemset2D(
    devPtr: CudaPointer,
    pitch: number,
    value: number,
    width: number,
    height: number
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const pointer = this.pointers.get(devPtr.id);
    if (!pointer) return CudaError.cudaErrorInvalidValue;

    // Create host buffer with proper pitch
    const hostBuffer = new Uint8Array(pitch * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        hostBuffer[y * pitch + x] = value & 0xff;
      }
    }

    device.queue.writeBuffer(pointer.buffer, devPtr.offset, hostBuffer);
    return CudaError.cudaSuccess;
  }

  async cudaMemset3D(
    pitchedPtr: CudaPitchedPtr,
    value: number,
    extent: { width: number; height: number; depth: number }
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const pointer = this.pointers.get(pitchedPtr.ptr.id);
    if (!pointer) return CudaError.cudaErrorInvalidValue;

    const sliceSize = pitchedPtr.pitch * extent.height;
    const totalSize = sliceSize * extent.depth;
    const hostBuffer = new Uint8Array(totalSize);

    for (let z = 0; z < extent.depth; z++) {
      for (let y = 0; y < extent.height; y++) {
        for (let x = 0; x < extent.width; x++) {
          hostBuffer[z * sliceSize + y * pitchedPtr.pitch + x] = value & 0xff;
        }
      }
    }

    device.queue.writeBuffer(pointer.buffer, pitchedPtr.ptr.offset, hostBuffer);
    return CudaError.cudaSuccess;
  }

  // 2D and 3D memory copy

  async cudaMemcpy2D(
    dst: CudaPointer | ArrayBuffer,
    dpitch: number,
    src: CudaPointer | ArrayBuffer,
    spitch: number,
    width: number,
    height: number,
    kind: CudaMemcpyKind
  ): Promise<CudaError> {
    // For simplicity, implement as series of 1D copies
    for (let y = 0; y < height; y++) {
      const srcOffset = y * spitch;
      const dstOffset = y * dpitch;

      // Adjust pointers for the row
      let rowSrc: CudaPointer | ArrayBuffer;
      let rowDst: CudaPointer | ArrayBuffer;

      if (this.isDevicePointer(src)) {
        rowSrc = { ...src, offset: src.offset + srcOffset };
      } else {
        rowSrc = (src as ArrayBuffer).slice(srcOffset, srcOffset + width);
      }

      if (this.isDevicePointer(dst)) {
        rowDst = { ...dst, offset: dst.offset + dstOffset };
      } else {
        // For host destination, we'll need to handle this differently
        const result = await this.cudaMemcpy(
          new ArrayBuffer(width),
          rowSrc,
          width,
          kind
        );
        if (result !== CudaError.cudaSuccess) return result;
        continue;
      }

      const result = await this.cudaMemcpy(rowDst, rowSrc, width, kind);
      if (result !== CudaError.cudaSuccess) return result;
    }

    return CudaError.cudaSuccess;
  }

  async cudaMemcpy3D(parms: CudaMemcpy3DParms): Promise<CudaError> {
    // 3D copy implementation
    const { extent, kind } = parms;

    for (let z = 0; z < extent.depth; z++) {
      for (let y = 0; y < extent.height; y++) {
        // Calculate source and destination offsets
        const srcSlice = parms.srcPtr
          ? parms.srcPtr.pitch * parms.srcPtr.ysize * (parms.srcPos.z + z) +
            parms.srcPtr.pitch * (parms.srcPos.y + y) +
            parms.srcPos.x
          : 0;
        const dstSlice = parms.dstPtr
          ? parms.dstPtr.pitch * parms.dstPtr.ysize * (parms.dstPos.z + z) +
            parms.dstPtr.pitch * (parms.dstPos.y + y) +
            parms.dstPos.x
          : 0;

        if (parms.srcPtr && parms.dstPtr) {
          const srcPtr = { ...parms.srcPtr.ptr, offset: srcSlice };
          const dstPtr = { ...parms.dstPtr.ptr, offset: dstSlice };
          const result = await this.cudaMemcpy(
            dstPtr,
            srcPtr,
            extent.width,
            kind
          );
          if (result !== CudaError.cudaSuccess) return result;
        }
      }
    }

    return CudaError.cudaSuccess;
  }

  // Pointer attributes

  cudaPointerGetAttributes(ptr: CudaPointer): {
    attributes: CudaPointerAttributes | null;
    error: CudaError;
  } {
    const pointer = this.pointers.get(ptr.id);
    if (!pointer) {
      return { attributes: null, error: CudaError.cudaErrorInvalidValue };
    }

    let type: CudaMemoryType;
    switch (pointer.type) {
      case 'device':
        type = CudaMemoryType.cudaMemoryTypeDevice;
        break;
      case 'host':
      case 'pinned':
        type = CudaMemoryType.cudaMemoryTypeHost;
        break;
      case 'managed':
        type = CudaMemoryType.cudaMemoryTypeManaged;
        break;
      default:
        type = CudaMemoryType.cudaMemoryTypeUnregistered;
    }

    const attributes: CudaPointerAttributes = {
      type,
      device: 0,
      devicePointer: pointer,
      hostPointer: this.hostBuffers.get(ptr.id) || null,
    };

    return { attributes, error: CudaError.cudaSuccess };
  }

  // Memory info

  cudaMemGetInfo(): { free: number; total: number; error: CudaError } {
    const device = cudaDevice.getDevice();
    if (!device) {
      return { free: 0, total: 0, error: CudaError.cudaErrorInitializationError };
    }

    const total = device.limits.maxBufferSize;
    const free = total - this.totalAllocatedMemory;

    return { free, total, error: CudaError.cudaSuccess };
  }

  // Get buffer for external use

  getBuffer(ptr: CudaPointer): GPUBuffer | null {
    const pointer = this.pointers.get(ptr.id);
    return pointer?.buffer || null;
  }

  // Prefetch for managed memory

  async cudaMemPrefetchAsync(
    devPtr: CudaPointer,
    count: number,
    dstDevice: number
  ): Promise<CudaError> {
    // In WebGPU, managed memory is automatically synchronized
    // This is essentially a no-op but we ensure the host buffer is synced
    const pointer = this.pointers.get(devPtr.id);
    if (!pointer || pointer.type !== 'managed') {
      return CudaError.cudaErrorInvalidValue;
    }

    const hostBuffer = this.hostBuffers.get(devPtr.id);
    if (hostBuffer) {
      const device = this.getDevice();
      if (device) {
        device.queue.writeBuffer(
          pointer.buffer,
          devPtr.offset,
          new Uint8Array(hostBuffer, 0, count)
        );
      }
    }

    return CudaError.cudaSuccess;
  }

  // Advise for managed memory

  cudaMemAdvise(
    devPtr: CudaPointer,
    count: number,
    advice: number
  ): CudaError {
    // Memory advice is used for optimization hints
    // In WebGPU, we don't have direct control over memory placement
    return CudaError.cudaSuccess;
  }

  // Clear all allocations (for cleanup)

  clearAll(): void {
    for (const [, pointer] of this.pointers) {
      pointer.buffer.destroy();
    }
    for (const [, pool] of this.bufferPool) {
      for (const buffer of pool) {
        buffer.destroy();
      }
    }
    this.pointers.clear();
    this.hostBuffers.clear();
    this.arrays.clear();
    this.bufferPool.clear();
    this.totalAllocatedMemory = 0;
  }
}

// Stream placeholder for async operations
export interface CudaStream {
  id: number;
}

// Export singleton
export const cudaMemory = CudaMemoryManager.getInstance();

// Export CUDA-like API functions
export function cudaMalloc(size: number): { ptr: CudaPointer | null; error: CudaError } {
  return cudaMemory.cudaMalloc(size);
}

export function cudaMallocPitch(
  width: number,
  height: number,
  elementSize: number
): { ptr: CudaPitchedPtr | null; error: CudaError } {
  return cudaMemory.cudaMallocPitch(width, height, elementSize);
}

export function cudaMalloc3D(
  extent: { width: number; height: number; depth: number },
  elementSize: number
): { ptr: CudaPitchedPtr | null; error: CudaError } {
  return cudaMemory.cudaMalloc3D(extent, elementSize);
}

export function cudaMallocHost(size: number): { ptr: CudaPointer | null; error: CudaError } {
  return cudaMemory.cudaMallocHost(size);
}

export function cudaHostAlloc(
  size: number,
  flags?: number
): { ptr: CudaPointer | null; error: CudaError } {
  return cudaMemory.cudaHostAlloc(size, flags);
}

export function cudaMallocManaged(size: number): { ptr: CudaPointer | null; error: CudaError } {
  return cudaMemory.cudaMallocManaged(size);
}

export function cudaFree(ptr: CudaPointer | null): CudaError {
  return cudaMemory.cudaFree(ptr);
}

export function cudaFreeHost(ptr: CudaPointer | null): CudaError {
  return cudaMemory.cudaFreeHost(ptr);
}

export function cudaMemcpy(
  dst: CudaPointer | ArrayBuffer,
  src: CudaPointer | ArrayBuffer,
  count: number,
  kind: CudaMemcpyKind
): Promise<CudaError> {
  return cudaMemory.cudaMemcpy(dst, src, count, kind);
}

export function cudaMemcpyAsync(
  dst: CudaPointer | ArrayBuffer,
  src: CudaPointer | ArrayBuffer,
  count: number,
  kind: CudaMemcpyKind,
  stream?: CudaStream
): Promise<CudaError> {
  return cudaMemory.cudaMemcpyAsync(dst, src, count, kind, stream);
}

export function cudaMemset(
  devPtr: CudaPointer,
  value: number,
  count: number
): Promise<CudaError> {
  return cudaMemory.cudaMemset(devPtr, value, count);
}

export function cudaMemset2D(
  devPtr: CudaPointer,
  pitch: number,
  value: number,
  width: number,
  height: number
): Promise<CudaError> {
  return cudaMemory.cudaMemset2D(devPtr, pitch, value, width, height);
}

export function cudaMemset3D(
  pitchedPtr: CudaPitchedPtr,
  value: number,
  extent: { width: number; height: number; depth: number }
): Promise<CudaError> {
  return cudaMemory.cudaMemset3D(pitchedPtr, value, extent);
}

export function cudaMemcpy2D(
  dst: CudaPointer | ArrayBuffer,
  dpitch: number,
  src: CudaPointer | ArrayBuffer,
  spitch: number,
  width: number,
  height: number,
  kind: CudaMemcpyKind
): Promise<CudaError> {
  return cudaMemory.cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

export function cudaMemcpy3D(parms: CudaMemcpy3DParms): Promise<CudaError> {
  return cudaMemory.cudaMemcpy3D(parms);
}

export function cudaPointerGetAttributes(ptr: CudaPointer): {
  attributes: CudaPointerAttributes | null;
  error: CudaError;
} {
  return cudaMemory.cudaPointerGetAttributes(ptr);
}

export function cudaMemGetInfo(): { free: number; total: number; error: CudaError } {
  return cudaMemory.cudaMemGetInfo();
}

export function cudaMemPrefetchAsync(
  devPtr: CudaPointer,
  count: number,
  dstDevice: number
): Promise<CudaError> {
  return cudaMemory.cudaMemPrefetchAsync(devPtr, count, dstDevice);
}

export function cudaMemAdvise(
  devPtr: CudaPointer,
  count: number,
  advice: number
): CudaError {
  return cudaMemory.cudaMemAdvise(devPtr, count, advice);
}

export function getBuffer(ptr: CudaPointer): GPUBuffer | null {
  return cudaMemory.getBuffer(ptr);
}
