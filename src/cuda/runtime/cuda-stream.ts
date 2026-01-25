/**
 * CUDA Stream and Event Management API for WebGPU
 *
 * Provides CUDA-compatible stream and event synchronization
 */

import { CudaError, cudaDevice } from './cuda-device';

export interface CudaStream {
  id: number;
  queue: GPUQueue;
  commandBuffers: GPUCommandBuffer[];
  pendingOperations: Promise<void>[];
  priority: number;
  flags: number;
  isNull: boolean;
}

export interface CudaEvent {
  id: number;
  timestamp: number;
  recorded: boolean;
  stream: CudaStream | null;
  querySet: GPUQuerySet | null;
  queryBuffer: GPUBuffer | null;
}

export enum CudaStreamFlags {
  cudaStreamDefault = 0,
  cudaStreamNonBlocking = 1,
}

export enum CudaEventFlags {
  cudaEventDefault = 0,
  cudaEventBlockingSync = 1,
  cudaEventDisableTiming = 2,
  cudaEventInterprocess = 4,
}

export enum CudaStreamCaptureStatus {
  cudaStreamCaptureStatusNone = 0,
  cudaStreamCaptureStatusActive = 1,
  cudaStreamCaptureStatusInvalidated = 2,
}

class CudaStreamManager {
  private static instance: CudaStreamManager | null = null;
  private streams: Map<number, CudaStream> = new Map();
  private events: Map<number, CudaEvent> = new Map();
  private nextStreamId: number = 1;
  private nextEventId: number = 1;
  private nullStream: CudaStream | null = null;
  private captureStatus: Map<number, CudaStreamCaptureStatus> = new Map();
  private capturedGraphs: Map<number, GPUCommandBuffer[]> = new Map();

  private constructor() {}

  static getInstance(): CudaStreamManager {
    if (!CudaStreamManager.instance) {
      CudaStreamManager.instance = new CudaStreamManager();
    }
    return CudaStreamManager.instance;
  }

  private getDevice(): GPUDevice | null {
    return cudaDevice.getDevice();
  }

  // Stream management

  cudaStreamCreate(): { stream: CudaStream | null; error: CudaError } {
    return this.cudaStreamCreateWithFlags(CudaStreamFlags.cudaStreamDefault);
  }

  cudaStreamCreateWithFlags(flags: number): {
    stream: CudaStream | null;
    error: CudaError;
  } {
    const device = this.getDevice();
    if (!device) {
      return { stream: null, error: CudaError.cudaErrorInitializationError };
    }

    const stream: CudaStream = {
      id: this.nextStreamId++,
      queue: device.queue,
      commandBuffers: [],
      pendingOperations: [],
      priority: 0,
      flags,
      isNull: false,
    };

    this.streams.set(stream.id, stream);
    this.captureStatus.set(stream.id, CudaStreamCaptureStatus.cudaStreamCaptureStatusNone);

    return { stream, error: CudaError.cudaSuccess };
  }

  cudaStreamCreateWithPriority(
    flags: number,
    priority: number
  ): { stream: CudaStream | null; error: CudaError } {
    const { stream, error } = this.cudaStreamCreateWithFlags(flags);
    if (stream) {
      stream.priority = priority;
    }
    return { stream, error };
  }

  cudaStreamDestroy(stream: CudaStream): CudaError {
    if (!stream || stream.isNull) {
      return CudaError.cudaSuccess;
    }

    const s = this.streams.get(stream.id);
    if (!s) {
      return CudaError.cudaErrorInvalidValue;
    }

    // Wait for all pending operations
    Promise.all(s.pendingOperations).catch(() => {});

    this.streams.delete(stream.id);
    this.captureStatus.delete(stream.id);
    this.capturedGraphs.delete(stream.id);

    return CudaError.cudaSuccess;
  }

  async cudaStreamSynchronize(stream: CudaStream | null): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) {
      return CudaError.cudaErrorInitializationError;
    }

    if (!stream || stream.isNull) {
      // Synchronize the default stream (all operations)
      await device.queue.onSubmittedWorkDone();
      return CudaError.cudaSuccess;
    }

    const s = this.streams.get(stream.id);
    if (!s) {
      return CudaError.cudaErrorInvalidValue;
    }

    // Wait for all pending operations on this stream
    await Promise.all(s.pendingOperations);
    s.pendingOperations = [];

    return CudaError.cudaSuccess;
  }

  cudaStreamQuery(stream: CudaStream | null): CudaError {
    if (!stream || stream.isNull) {
      return CudaError.cudaSuccess;
    }

    const s = this.streams.get(stream.id);
    if (!s) {
      return CudaError.cudaErrorInvalidValue;
    }

    // Check if all operations are complete
    if (s.pendingOperations.length > 0) {
      return CudaError.cudaErrorNotReady;
    }

    return CudaError.cudaSuccess;
  }

  cudaStreamWaitEvent(
    stream: CudaStream | null,
    event: CudaEvent,
    flags: number = 0
  ): CudaError {
    const e = this.events.get(event.id);
    if (!e || !e.recorded) {
      return CudaError.cudaErrorInvalidValue;
    }

    // In WebGPU, we handle this by adding a dependency
    // The stream will wait until the event's stream has completed
    if (e.stream && !stream?.isNull) {
      const targetStream = stream ? this.streams.get(stream.id) : null;
      if (targetStream) {
        const eventStream = this.streams.get(e.stream.id);
        if (eventStream) {
          targetStream.pendingOperations.push(
            Promise.all(eventStream.pendingOperations).then(() => {})
          );
        }
      }
    }

    return CudaError.cudaSuccess;
  }

  cudaStreamGetPriority(stream: CudaStream): {
    priority: number;
    error: CudaError;
  } {
    if (stream.isNull) {
      return { priority: 0, error: CudaError.cudaSuccess };
    }

    const s = this.streams.get(stream.id);
    if (!s) {
      return { priority: 0, error: CudaError.cudaErrorInvalidValue };
    }

    return { priority: s.priority, error: CudaError.cudaSuccess };
  }

  cudaStreamGetFlags(stream: CudaStream): { flags: number; error: CudaError } {
    if (stream.isNull) {
      return { flags: 0, error: CudaError.cudaSuccess };
    }

    const s = this.streams.get(stream.id);
    if (!s) {
      return { flags: 0, error: CudaError.cudaErrorInvalidValue };
    }

    return { flags: s.flags, error: CudaError.cudaSuccess };
  }

  getDefaultStream(): CudaStream {
    if (!this.nullStream) {
      const device = this.getDevice();
      this.nullStream = {
        id: 0,
        queue: device!.queue,
        commandBuffers: [],
        pendingOperations: [],
        priority: 0,
        flags: CudaStreamFlags.cudaStreamDefault,
        isNull: true,
      };
    }
    return this.nullStream;
  }

  // Event management

  cudaEventCreate(): { event: CudaEvent | null; error: CudaError } {
    return this.cudaEventCreateWithFlags(CudaEventFlags.cudaEventDefault);
  }

  cudaEventCreateWithFlags(flags: number): {
    event: CudaEvent | null;
    error: CudaError;
  } {
    const device = this.getDevice();
    if (!device) {
      return { event: null, error: CudaError.cudaErrorInitializationError };
    }

    let querySet: GPUQuerySet | null = null;
    let queryBuffer: GPUBuffer | null = null;

    // Create timestamp query set if timing is enabled
    if (!(flags & CudaEventFlags.cudaEventDisableTiming)) {
      if (cudaDevice.hasFeature('timestamp-query')) {
        querySet = device.createQuerySet({
          type: 'timestamp',
          count: 2,
        });
        queryBuffer = device.createBuffer({
          size: 16, // 2 * 8 bytes for timestamps
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
      }
    }

    const event: CudaEvent = {
      id: this.nextEventId++,
      timestamp: 0,
      recorded: false,
      stream: null,
      querySet,
      queryBuffer,
    };

    this.events.set(event.id, event);

    return { event, error: CudaError.cudaSuccess };
  }

  cudaEventDestroy(event: CudaEvent): CudaError {
    const e = this.events.get(event.id);
    if (!e) {
      return CudaError.cudaErrorInvalidValue;
    }

    if (e.querySet) {
      e.querySet.destroy();
    }
    if (e.queryBuffer) {
      e.queryBuffer.destroy();
    }

    this.events.delete(event.id);
    return CudaError.cudaSuccess;
  }

  cudaEventRecord(event: CudaEvent, stream: CudaStream | null = null): CudaError {
    const device = this.getDevice();
    if (!device) {
      return CudaError.cudaErrorInitializationError;
    }

    const e = this.events.get(event.id);
    if (!e) {
      return CudaError.cudaErrorInvalidValue;
    }

    e.recorded = true;
    e.stream = stream;
    e.timestamp = performance.now();

    // If we have timestamp queries, record them
    if (e.querySet && e.queryBuffer) {
      const commandEncoder = device.createCommandEncoder();
      commandEncoder.writeTimestamp(e.querySet, 0);
      commandEncoder.resolveQuerySet(e.querySet, 0, 1, e.queryBuffer, 0);
      device.queue.submit([commandEncoder.finish()]);
    }

    return CudaError.cudaSuccess;
  }

  async cudaEventSynchronize(event: CudaEvent): Promise<CudaError> {
    const e = this.events.get(event.id);
    if (!e || !e.recorded) {
      return CudaError.cudaErrorInvalidValue;
    }

    // Wait for the stream associated with the event
    if (e.stream) {
      await this.cudaStreamSynchronize(e.stream);
    } else {
      // Wait for default stream
      const device = this.getDevice();
      if (device) {
        await device.queue.onSubmittedWorkDone();
      }
    }

    return CudaError.cudaSuccess;
  }

  cudaEventQuery(event: CudaEvent): CudaError {
    const e = this.events.get(event.id);
    if (!e) {
      return CudaError.cudaErrorInvalidValue;
    }

    if (!e.recorded) {
      return CudaError.cudaErrorNotReady;
    }

    // Check if the stream has completed
    if (e.stream) {
      const s = this.streams.get(e.stream.id);
      if (s && s.pendingOperations.length > 0) {
        return CudaError.cudaErrorNotReady;
      }
    }

    return CudaError.cudaSuccess;
  }

  async cudaEventElapsedTime(
    start: CudaEvent,
    end: CudaEvent
  ): Promise<{ time: number; error: CudaError }> {
    const startEvent = this.events.get(start.id);
    const endEvent = this.events.get(end.id);

    if (!startEvent || !endEvent) {
      return { time: 0, error: CudaError.cudaErrorInvalidValue };
    }

    if (!startEvent.recorded || !endEvent.recorded) {
      return { time: 0, error: CudaError.cudaErrorNotReady };
    }

    // Use high-resolution timestamps if available
    if (startEvent.queryBuffer && endEvent.queryBuffer) {
      // Read timestamp values from GPU
      // This would require reading back the query buffers
      // For simplicity, use performance.now() fallback
    }

    // Use performance.now() timestamps
    const elapsed = endEvent.timestamp - startEvent.timestamp;
    return { time: elapsed, error: CudaError.cudaSuccess };
  }

  // Stream capture (for CUDA Graphs)

  cudaStreamBeginCapture(
    stream: CudaStream,
    mode: number = 0
  ): CudaError {
    const s = this.streams.get(stream.id);
    if (!s) {
      return CudaError.cudaErrorInvalidValue;
    }

    this.captureStatus.set(stream.id, CudaStreamCaptureStatus.cudaStreamCaptureStatusActive);
    this.capturedGraphs.set(stream.id, []);

    return CudaError.cudaSuccess;
  }

  cudaStreamEndCapture(stream: CudaStream): {
    graph: GPUCommandBuffer[];
    error: CudaError;
  } {
    const s = this.streams.get(stream.id);
    if (!s) {
      return { graph: [], error: CudaError.cudaErrorInvalidValue };
    }

    const status = this.captureStatus.get(stream.id);
    if (status !== CudaStreamCaptureStatus.cudaStreamCaptureStatusActive) {
      return { graph: [], error: CudaError.cudaErrorInvalidValue };
    }

    const graph = this.capturedGraphs.get(stream.id) || [];
    this.captureStatus.set(stream.id, CudaStreamCaptureStatus.cudaStreamCaptureStatusNone);
    this.capturedGraphs.delete(stream.id);

    return { graph, error: CudaError.cudaSuccess };
  }

  cudaStreamIsCapturing(stream: CudaStream): {
    status: CudaStreamCaptureStatus;
    error: CudaError;
  } {
    const status =
      this.captureStatus.get(stream.id) ||
      CudaStreamCaptureStatus.cudaStreamCaptureStatusNone;
    return { status, error: CudaError.cudaSuccess };
  }

  // Add command buffer to stream (for graph capture)

  addCommandBuffer(stream: CudaStream, commandBuffer: GPUCommandBuffer): void {
    const status = this.captureStatus.get(stream.id);
    if (status === CudaStreamCaptureStatus.cudaStreamCaptureStatusActive) {
      const captured = this.capturedGraphs.get(stream.id) || [];
      captured.push(commandBuffer);
      this.capturedGraphs.set(stream.id, captured);
    }
  }

  // Add pending operation to stream

  addPendingOperation(stream: CudaStream | null, operation: Promise<void>): void {
    if (!stream || stream.isNull) {
      return;
    }

    const s = this.streams.get(stream.id);
    if (s) {
      s.pendingOperations.push(operation);
    }
  }

  // Stream callback

  cudaStreamAddCallback(
    stream: CudaStream | null,
    callback: () => void,
    userData: unknown = null
  ): CudaError {
    const operation = (async () => {
      if (stream && !stream.isNull) {
        const s = this.streams.get(stream.id);
        if (s) {
          await Promise.all(s.pendingOperations);
        }
      }
      callback();
    })();

    if (stream && !stream.isNull) {
      const s = this.streams.get(stream.id);
      if (s) {
        s.pendingOperations.push(operation);
      }
    }

    return CudaError.cudaSuccess;
  }

  // Clear all

  clearAll(): void {
    for (const [, event] of this.events) {
      if (event.querySet) event.querySet.destroy();
      if (event.queryBuffer) event.queryBuffer.destroy();
    }
    this.streams.clear();
    this.events.clear();
    this.captureStatus.clear();
    this.capturedGraphs.clear();
    this.nullStream = null;
    this.nextStreamId = 1;
    this.nextEventId = 1;
  }
}

// Export singleton
export const cudaStreamManager = CudaStreamManager.getInstance();

// Export CUDA-like API functions

export function cudaStreamCreate(): { stream: CudaStream | null; error: CudaError } {
  return cudaStreamManager.cudaStreamCreate();
}

export function cudaStreamCreateWithFlags(flags: number): {
  stream: CudaStream | null;
  error: CudaError;
} {
  return cudaStreamManager.cudaStreamCreateWithFlags(flags);
}

export function cudaStreamCreateWithPriority(
  flags: number,
  priority: number
): { stream: CudaStream | null; error: CudaError } {
  return cudaStreamManager.cudaStreamCreateWithPriority(flags, priority);
}

export function cudaStreamDestroy(stream: CudaStream): CudaError {
  return cudaStreamManager.cudaStreamDestroy(stream);
}

export function cudaStreamSynchronize(stream: CudaStream | null): Promise<CudaError> {
  return cudaStreamManager.cudaStreamSynchronize(stream);
}

export function cudaStreamQuery(stream: CudaStream | null): CudaError {
  return cudaStreamManager.cudaStreamQuery(stream);
}

export function cudaStreamWaitEvent(
  stream: CudaStream | null,
  event: CudaEvent,
  flags?: number
): CudaError {
  return cudaStreamManager.cudaStreamWaitEvent(stream, event, flags);
}

export function cudaStreamGetPriority(stream: CudaStream): {
  priority: number;
  error: CudaError;
} {
  return cudaStreamManager.cudaStreamGetPriority(stream);
}

export function cudaStreamGetFlags(stream: CudaStream): {
  flags: number;
  error: CudaError;
} {
  return cudaStreamManager.cudaStreamGetFlags(stream);
}

export function cudaStreamBeginCapture(
  stream: CudaStream,
  mode?: number
): CudaError {
  return cudaStreamManager.cudaStreamBeginCapture(stream, mode);
}

export function cudaStreamEndCapture(stream: CudaStream): {
  graph: GPUCommandBuffer[];
  error: CudaError;
} {
  return cudaStreamManager.cudaStreamEndCapture(stream);
}

export function cudaStreamIsCapturing(stream: CudaStream): {
  status: CudaStreamCaptureStatus;
  error: CudaError;
} {
  return cudaStreamManager.cudaStreamIsCapturing(stream);
}

export function cudaStreamAddCallback(
  stream: CudaStream | null,
  callback: () => void,
  userData?: unknown
): CudaError {
  return cudaStreamManager.cudaStreamAddCallback(stream, callback, userData);
}

export function cudaEventCreate(): { event: CudaEvent | null; error: CudaError } {
  return cudaStreamManager.cudaEventCreate();
}

export function cudaEventCreateWithFlags(flags: number): {
  event: CudaEvent | null;
  error: CudaError;
} {
  return cudaStreamManager.cudaEventCreateWithFlags(flags);
}

export function cudaEventDestroy(event: CudaEvent): CudaError {
  return cudaStreamManager.cudaEventDestroy(event);
}

export function cudaEventRecord(
  event: CudaEvent,
  stream?: CudaStream | null
): CudaError {
  return cudaStreamManager.cudaEventRecord(event, stream);
}

export function cudaEventSynchronize(event: CudaEvent): Promise<CudaError> {
  return cudaStreamManager.cudaEventSynchronize(event);
}

export function cudaEventQuery(event: CudaEvent): CudaError {
  return cudaStreamManager.cudaEventQuery(event);
}

export function cudaEventElapsedTime(
  start: CudaEvent,
  end: CudaEvent
): Promise<{ time: number; error: CudaError }> {
  return cudaStreamManager.cudaEventElapsedTime(start, end);
}

export function getDefaultStream(): CudaStream {
  return cudaStreamManager.getDefaultStream();
}
