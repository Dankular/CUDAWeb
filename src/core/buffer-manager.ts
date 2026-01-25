/**
 * GPU Buffer Manager
 * Handles efficient allocation, pooling, and management of GPU buffers
 * Supports chunked uploads for large model weights (2GB+)
 */

import { GPUContext } from './gpu-context';

export interface BufferAllocation {
  buffer: GPUBuffer;
  offset: number;
  size: number;
  label?: string;
}

export interface ChunkedUploadOptions {
  chunkSize?: number;
  onProgress?: (loaded: number, total: number) => void;
}

export class BufferManager {
  private gpu: GPUContext;
  private bufferPool: Map<string, GPUBuffer[]> = new Map();
  private allocations: Map<GPUBuffer, BufferAllocation> = new Map();
  private stagingBuffers: GPUBuffer[] = [];

  // Default chunk size for uploads (64MB)
  private readonly DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024;

  // Maximum staging buffer pool size
  private readonly MAX_STAGING_POOL = 4;

  constructor(gpu: GPUContext) {
    this.gpu = gpu;
  }

  /**
   * Create a buffer with the specified usage and size
   */
  createBuffer(
    size: number,
    usage: GPUBufferUsageFlags,
    label?: string,
    mappedAtCreation = false
  ): GPUBuffer {
    const buffer = this.gpu.createBuffer({
      size: this.alignSize(size),
      usage,
      label,
      mappedAtCreation,
    });

    this.allocations.set(buffer, {
      buffer,
      offset: 0,
      size,
      label,
    });

    return buffer;
  }

  /**
   * Create a storage buffer (readable/writable in compute shaders)
   */
  createStorageBuffer(size: number, label?: string): GPUBuffer {
    return this.createBuffer(
      size,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label
    );
  }

  /**
   * Create a uniform buffer
   */
  createUniformBuffer(size: number, label?: string): GPUBuffer {
    return this.createBuffer(
      size,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label
    );
  }

  /**
   * Create a vertex buffer
   */
  createVertexBuffer(size: number, label?: string): GPUBuffer {
    return this.createBuffer(
      size,
      GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      label
    );
  }

  /**
   * Create an index buffer
   */
  createIndexBuffer(size: number, label?: string): GPUBuffer {
    return this.createBuffer(
      size,
      GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      label
    );
  }

  /**
   * Create a read-back buffer for GPU -> CPU data transfer
   */
  createReadBuffer(size: number, label?: string): GPUBuffer {
    return this.createBuffer(
      size,
      GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label
    );
  }

  /**
   * Get or create a staging buffer from the pool
   */
  private getStagingBuffer(size: number): GPUBuffer {
    const alignedSize = this.alignSize(size);

    // Try to find a suitable buffer in the pool
    if (this.stagingBuffers.length > 0) {
      const idx = this.stagingBuffers.findIndex(
        (b) => b.size >= alignedSize && b.mapState === 'unmapped'
      );
      if (idx >= 0) {
        return this.stagingBuffers.splice(idx, 1)[0];
      }
    }

    // Create a new staging buffer
    return this.createBuffer(
      alignedSize,
      GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
      'staging-buffer',
      true
    );
  }

  /**
   * Return a staging buffer to the pool
   */
  private returnStagingBuffer(buffer: GPUBuffer): void {
    if (this.stagingBuffers.length < this.MAX_STAGING_POOL) {
      this.stagingBuffers.push(buffer);
    } else {
      buffer.destroy();
    }
  }

  /**
   * Upload data to a buffer using the queue
   */
  uploadToBuffer(buffer: GPUBuffer, data: ArrayBuffer | ArrayBufferView, offset = 0): void {
    this.gpu.writeBuffer(buffer, offset, data);
  }

  /**
   * Upload large data in chunks with progress reporting
   * Essential for 2GB+ model weights
   */
  async uploadLargeBuffer(
    buffer: GPUBuffer,
    data: ArrayBuffer,
    options: ChunkedUploadOptions = {}
  ): Promise<void> {
    const chunkSize = options.chunkSize ?? this.DEFAULT_CHUNK_SIZE;
    const totalSize = data.byteLength;
    let uploaded = 0;

    const dataView = new Uint8Array(data);

    while (uploaded < totalSize) {
      const remaining = totalSize - uploaded;
      const currentChunkSize = Math.min(chunkSize, remaining);

      // Upload chunk
      const chunk = dataView.subarray(uploaded, uploaded + currentChunkSize);
      this.gpu.writeBuffer(buffer, uploaded, chunk);

      uploaded += currentChunkSize;

      // Report progress
      options.onProgress?.(uploaded, totalSize);

      // Yield to allow other operations
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  /**
   * Stream data from a URL directly to a GPU buffer
   * Useful for loading large model weights without full memory allocation
   */
  async streamToBuffer(
    url: string,
    buffer: GPUBuffer,
    options: ChunkedUploadOptions = {}
  ): Promise<void> {
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Failed to fetch: ${response.status} ${response.statusText}`);
    }

    const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
    const reader = response.body?.getReader();

    if (!reader) {
      throw new Error('Failed to get response reader');
    }

    const chunkSize = options.chunkSize ?? this.DEFAULT_CHUNK_SIZE;
    let uploadBuffer = new Uint8Array(chunkSize);
    let bufferOffset = 0;
    let totalUploaded = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          // Upload any remaining data
          if (bufferOffset > 0) {
            this.gpu.writeBuffer(buffer, totalUploaded, uploadBuffer.subarray(0, bufferOffset));
            totalUploaded += bufferOffset;
          }
          break;
        }

        let valueOffset = 0;
        while (valueOffset < value.length) {
          const remaining = chunkSize - bufferOffset;
          const toCopy = Math.min(remaining, value.length - valueOffset);

          uploadBuffer.set(value.subarray(valueOffset, valueOffset + toCopy), bufferOffset);
          bufferOffset += toCopy;
          valueOffset += toCopy;

          // Upload when chunk is full
          if (bufferOffset >= chunkSize) {
            this.gpu.writeBuffer(buffer, totalUploaded, uploadBuffer);
            totalUploaded += chunkSize;
            bufferOffset = 0;

            options.onProgress?.(totalUploaded, contentLength);
          }
        }
      }

      options.onProgress?.(totalUploaded, contentLength);
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Read data back from GPU buffer
   */
  async readBuffer(buffer: GPUBuffer, offset = 0, size?: number): Promise<ArrayBuffer> {
    const readSize = size ?? buffer.size - offset;
    const readBuffer = this.createReadBuffer(readSize);

    const encoder = this.gpu.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, readBuffer, 0, readSize);
    this.gpu.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const data = readBuffer.getMappedRange().slice(0);
    readBuffer.unmap();
    readBuffer.destroy();

    return data;
  }

  /**
   * Copy data between buffers
   */
  copyBuffer(
    source: GPUBuffer,
    destination: GPUBuffer,
    size: number,
    sourceOffset = 0,
    destOffset = 0
  ): GPUCommandBuffer {
    const encoder = this.gpu.createCommandEncoder();
    encoder.copyBufferToBuffer(source, sourceOffset, destination, destOffset, size);
    return encoder.finish();
  }

  /**
   * Clear a buffer to zero
   */
  clearBuffer(buffer: GPUBuffer, offset = 0, size?: number): void {
    const clearSize = size ?? buffer.size - offset;
    const zeros = new Uint8Array(Math.min(clearSize, this.DEFAULT_CHUNK_SIZE));

    let cleared = 0;
    while (cleared < clearSize) {
      const remaining = clearSize - cleared;
      const toClear = Math.min(zeros.length, remaining);
      this.gpu.writeBuffer(buffer, offset + cleared, zeros.subarray(0, toClear));
      cleared += toClear;
    }
  }

  /**
   * Align size to 4-byte boundary (required by WebGPU)
   */
  private alignSize(size: number, alignment = 4): number {
    return Math.ceil(size / alignment) * alignment;
  }

  /**
   * Get buffer from pool or create new one
   */
  getPooledBuffer(key: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const pool = this.bufferPool.get(key) ?? [];

    // Find a suitable buffer in the pool
    const idx = pool.findIndex((b) => b.size >= size);
    if (idx >= 0) {
      return pool.splice(idx, 1)[0];
    }

    // Create new buffer
    return this.createBuffer(size, usage, key);
  }

  /**
   * Return buffer to pool
   */
  returnToPool(key: string, buffer: GPUBuffer): void {
    const pool = this.bufferPool.get(key) ?? [];
    pool.push(buffer);
    this.bufferPool.set(key, pool);
  }

  /**
   * Destroy a buffer and remove from tracking
   */
  destroyBuffer(buffer: GPUBuffer): void {
    buffer.destroy();
    this.allocations.delete(buffer);
  }

  /**
   * Clear all pooled buffers
   */
  clearPools(): void {
    for (const pool of this.bufferPool.values()) {
      for (const buffer of pool) {
        buffer.destroy();
      }
    }
    this.bufferPool.clear();

    for (const buffer of this.stagingBuffers) {
      buffer.destroy();
    }
    this.stagingBuffers = [];
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): {
    totalAllocated: number;
    bufferCount: number;
    pooledBuffers: number;
  } {
    let totalAllocated = 0;
    for (const alloc of this.allocations.values()) {
      totalAllocated += alloc.size;
    }

    let pooledBuffers = 0;
    for (const pool of this.bufferPool.values()) {
      pooledBuffers += pool.length;
    }

    return {
      totalAllocated,
      bufferCount: this.allocations.size,
      pooledBuffers,
    };
  }
}
