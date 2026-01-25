/**
 * Weight Loader for Large Model Files
 * Handles streaming, chunked loading, and memory-efficient weight management
 * Supports 2GB+ model files through Range requests and chunked GPU uploads
 */

import { GPUContext, BufferManager } from '../core';

export interface WeightMetadata {
  name: string;
  dtype: 'float32' | 'float16' | 'int32' | 'uint32' | 'int8' | 'uint8';
  shape: number[];
  offset: number;
  size: number;
}

export interface ModelManifest {
  version: string;
  totalSize: number;
  weights: WeightMetadata[];
  shDegree?: number;
  numGaussians?: number;
}

export interface LoadProgress {
  loaded: number;
  total: number;
  currentWeight?: string;
  phase: 'metadata' | 'weights' | 'complete';
}

export type ProgressCallback = (progress: LoadProgress) => void;

export interface WeightBuffer {
  name: string;
  buffer: GPUBuffer;
  metadata: WeightMetadata;
}

const DTYPE_SIZES: Record<string, number> = {
  float32: 4,
  float16: 2,
  int32: 4,
  uint32: 4,
  int8: 1,
  uint8: 1,
};

export class WeightLoader {
  private gpu: GPUContext;
  private bufferManager: BufferManager;
  private loadedWeights: Map<string, WeightBuffer> = new Map();
  private abortController: AbortController | null = null;

  // Chunk size for streaming (64MB default)
  private readonly CHUNK_SIZE = 64 * 1024 * 1024;

  constructor(gpu: GPUContext, bufferManager: BufferManager) {
    this.gpu = gpu;
    this.bufferManager = bufferManager;
  }

  /**
   * Load model weights from URL with streaming
   */
  async loadFromUrl(
    url: string,
    onProgress?: ProgressCallback
  ): Promise<Map<string, WeightBuffer>> {
    this.abortController = new AbortController();

    try {
      // First, load the manifest
      onProgress?.({ loaded: 0, total: 0, phase: 'metadata' });

      const manifestUrl = url.endsWith('.bin')
        ? url.replace('.bin', '.json')
        : `${url}/manifest.json`;

      const manifest = await this.loadManifest(manifestUrl);

      onProgress?.({ loaded: 0, total: manifest.totalSize, phase: 'weights' });

      // Load weights based on manifest
      const weightsUrl = url.endsWith('.bin') ? url : `${url}/weights.bin`;

      await this.loadWeightsStreaming(weightsUrl, manifest, onProgress);

      onProgress?.({ loaded: manifest.totalSize, total: manifest.totalSize, phase: 'complete' });

      return this.loadedWeights;
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        console.log('Weight loading aborted');
      }
      throw error;
    }
  }

  /**
   * Load manifest JSON file
   */
  private async loadManifest(url: string): Promise<ModelManifest> {
    const response = await fetch(url, {
      signal: this.abortController?.signal,
    });

    if (!response.ok) {
      throw new Error(`Failed to load manifest: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Stream weights from URL and upload to GPU
   */
  private async loadWeightsStreaming(
    url: string,
    manifest: ModelManifest,
    onProgress?: ProgressCallback
  ): Promise<void> {
    // Check if server supports range requests
    const headResponse = await fetch(url, {
      method: 'HEAD',
      signal: this.abortController?.signal,
    });

    const acceptRanges = headResponse.headers.get('Accept-Ranges');
    const supportsRange = acceptRanges === 'bytes';

    if (supportsRange) {
      // Load weights using range requests for better memory efficiency
      await this.loadWeightsWithRangeRequests(url, manifest, onProgress);
    } else {
      // Fallback to streaming the entire file
      await this.loadWeightsWithStream(url, manifest, onProgress);
    }
  }

  /**
   * Load weights using HTTP Range requests
   * This allows loading specific weight tensors without downloading the entire file
   */
  private async loadWeightsWithRangeRequests(
    url: string,
    manifest: ModelManifest,
    onProgress?: ProgressCallback
  ): Promise<void> {
    let totalLoaded = 0;

    for (const weight of manifest.weights) {
      onProgress?.({
        loaded: totalLoaded,
        total: manifest.totalSize,
        currentWeight: weight.name,
        phase: 'weights',
      });

      // Create GPU buffer for this weight
      const gpuBuffer = this.bufferManager.createStorageBuffer(
        weight.size,
        `weight-${weight.name}`
      );

      // Load in chunks
      let weightLoaded = 0;
      while (weightLoaded < weight.size) {
        const chunkStart = weight.offset + weightLoaded;
        const chunkEnd = Math.min(
          weight.offset + weight.size - 1,
          chunkStart + this.CHUNK_SIZE - 1
        );
        const chunkSize = chunkEnd - chunkStart + 1;

        const response = await fetch(url, {
          headers: {
            Range: `bytes=${chunkStart}-${chunkEnd}`,
          },
          signal: this.abortController?.signal,
        });

        if (!response.ok && response.status !== 206) {
          throw new Error(`Failed to load weight chunk: ${response.status}`);
        }

        const chunk = await response.arrayBuffer();

        // Upload to GPU
        this.gpu.writeBuffer(gpuBuffer, weightLoaded, chunk);

        weightLoaded += chunkSize;
        totalLoaded += chunkSize;

        onProgress?.({
          loaded: totalLoaded,
          total: manifest.totalSize,
          currentWeight: weight.name,
          phase: 'weights',
        });

        // Yield to prevent blocking
        await this.yieldToMain();
      }

      // Store loaded weight
      this.loadedWeights.set(weight.name, {
        name: weight.name,
        buffer: gpuBuffer,
        metadata: weight,
      });
    }
  }

  /**
   * Load weights by streaming the entire file
   * Used when Range requests are not supported
   */
  private async loadWeightsWithStream(
    url: string,
    manifest: ModelManifest,
    onProgress?: ProgressCallback
  ): Promise<void> {
    // Pre-create all GPU buffers
    const buffers = new Map<string, GPUBuffer>();
    for (const weight of manifest.weights) {
      buffers.set(
        weight.name,
        this.bufferManager.createStorageBuffer(weight.size, `weight-${weight.name}`)
      );
    }

    const response = await fetch(url, {
      signal: this.abortController?.signal,
    });

    if (!response.ok) {
      throw new Error(`Failed to load weights: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Failed to get response reader');
    }

    let totalLoaded = 0;
    let currentOffset = 0;
    let pendingData = new Uint8Array(0);

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        // Append to pending data
        const newPending = new Uint8Array(pendingData.length + value.length);
        newPending.set(pendingData);
        newPending.set(value, pendingData.length);
        pendingData = newPending;
        totalLoaded += value.length;

        // Process complete weights
        while (pendingData.length > 0) {
          // Find which weight this data belongs to
          const weight = manifest.weights.find(
            (w) =>
              w.offset <= currentOffset &&
              w.offset + w.size > currentOffset
          );

          if (!weight) {
            // Data before first weight or between weights
            const nextWeight = manifest.weights.find((w) => w.offset >= currentOffset);
            if (nextWeight) {
              const skip = nextWeight.offset - currentOffset;
              if (pendingData.length >= skip) {
                pendingData = pendingData.slice(skip);
                currentOffset = nextWeight.offset;
                continue;
              }
            }
            break;
          }

          const weightBuffer = buffers.get(weight.name)!;
          const offsetInWeight = currentOffset - weight.offset;
          const remainingInWeight = weight.size - offsetInWeight;
          const toUpload = Math.min(remainingInWeight, pendingData.length);

          // Upload chunk to GPU
          this.gpu.writeBuffer(weightBuffer, offsetInWeight, pendingData.slice(0, toUpload));

          pendingData = pendingData.slice(toUpload);
          currentOffset += toUpload;

          // Check if weight is complete
          if (offsetInWeight + toUpload >= weight.size) {
            this.loadedWeights.set(weight.name, {
              name: weight.name,
              buffer: weightBuffer,
              metadata: weight,
            });
          }
        }

        onProgress?.({
          loaded: totalLoaded,
          total: manifest.totalSize,
          phase: 'weights',
        });

        await this.yieldToMain();
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Load Gaussian splatting specific weights
   * Optimized structure for the 3DGS pipeline
   */
  async loadGaussianModel(
    url: string,
    onProgress?: ProgressCallback
  ): Promise<GaussianModelWeights> {
    const weights = await this.loadFromUrl(url, onProgress);

    // Extract Gaussian-specific weights
    const result: GaussianModelWeights = {
      positions: this.getRequiredWeight(weights, 'positions'),
      scales: this.getRequiredWeight(weights, 'scales'),
      rotations: this.getRequiredWeight(weights, 'rotations'),
      opacities: this.getRequiredWeight(weights, 'opacities'),
      shCoefficients: this.getRequiredWeight(weights, 'sh_coefficients'),
      numGaussians: 0,
      shDegree: 0,
    };

    // Calculate number of Gaussians from positions shape
    const posMetadata = result.positions.metadata;
    result.numGaussians = posMetadata.shape[0];
    result.shDegree = Math.floor(Math.sqrt(result.shCoefficients.metadata.shape[1])) - 1;

    return result;
  }

  private getRequiredWeight(
    weights: Map<string, WeightBuffer>,
    name: string
  ): WeightBuffer {
    const weight = weights.get(name);
    if (!weight) {
      throw new Error(`Required weight "${name}" not found in model`);
    }
    return weight;
  }

  /**
   * Load weights from IndexedDB cache
   */
  async loadFromCache(cacheKey: string): Promise<Map<string, WeightBuffer> | null> {
    try {
      const db = await this.openDatabase();
      const tx = db.transaction('weights', 'readonly');
      const store = tx.objectStore('weights');

      const cached = await this.promisifyRequest<CachedModel>(store.get(cacheKey));

      if (!cached) {
        return null;
      }

      // Restore GPU buffers from cached data
      for (const weight of cached.weights) {
        const gpuBuffer = this.bufferManager.createStorageBuffer(
          weight.data.byteLength,
          `weight-${weight.name}`
        );
        this.gpu.writeBuffer(gpuBuffer, 0, weight.data);

        this.loadedWeights.set(weight.name, {
          name: weight.name,
          buffer: gpuBuffer,
          metadata: weight.metadata,
        });
      }

      return this.loadedWeights;
    } catch {
      console.warn('Failed to load from cache');
      return null;
    }
  }

  /**
   * Save loaded weights to IndexedDB cache
   */
  async saveToCache(cacheKey: string): Promise<void> {
    try {
      const db = await this.openDatabase();
      const tx = db.transaction('weights', 'readwrite');
      const store = tx.objectStore('weights');

      // Read GPU buffers back to CPU
      const weights: CachedWeight[] = [];
      for (const [name, weight] of this.loadedWeights) {
        const data = await this.bufferManager.readBuffer(weight.buffer);
        weights.push({
          name,
          metadata: weight.metadata,
          data: new Uint8Array(data),
        });
      }

      await this.promisifyRequest(
        store.put({
          key: cacheKey,
          timestamp: Date.now(),
          weights,
        } as CachedModel)
      );
    } catch (error) {
      console.warn('Failed to save to cache:', error);
    }
  }

  private async openDatabase(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('gaussian-splatting-weights', 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains('weights')) {
          db.createObjectStore('weights', { keyPath: 'key' });
        }
      };
    });
  }

  private promisifyRequest<T>(request: IDBRequest<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  private async yieldToMain(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
  }

  /**
   * Abort any in-progress loading
   */
  abort(): void {
    this.abortController?.abort();
  }

  /**
   * Get a loaded weight buffer by name
   */
  getWeight(name: string): WeightBuffer | undefined {
    return this.loadedWeights.get(name);
  }

  /**
   * Release all loaded weights
   */
  dispose(): void {
    for (const weight of this.loadedWeights.values()) {
      this.bufferManager.destroyBuffer(weight.buffer);
    }
    this.loadedWeights.clear();
  }
}

export interface GaussianModelWeights {
  positions: WeightBuffer;
  scales: WeightBuffer;
  rotations: WeightBuffer;
  opacities: WeightBuffer;
  shCoefficients: WeightBuffer;
  numGaussians: number;
  shDegree: number;
}

interface CachedWeight {
  name: string;
  metadata: WeightMetadata;
  data: Uint8Array;
}

interface CachedModel {
  key: string;
  timestamp: number;
  weights: CachedWeight[];
}
