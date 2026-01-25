/**
 * CUDA Runtime API for WebGPU
 *
 * Complete CUDA Runtime API implementation mapped to WebGPU
 */

// Device Management
export * from './cuda-device';

// Memory Management
export * from './cuda-memory';

// Stream and Event Management
export * from './cuda-stream';

// Kernel Launch
export * from './cuda-kernel';

// Re-export common types and functions for convenience
export {
  CudaError,
  CudaMemcpyKind,
  CudaComputeMode,
  CudaDeviceAttr,
  cudaInit,
  cudaGetDeviceCount,
  cudaGetDevice,
  cudaSetDevice,
  cudaGetDeviceProperties,
  cudaDeviceSynchronize,
  cudaDeviceReset,
  cudaGetLastError,
  cudaPeekAtLastError,
  cudaGetErrorString,
  cudaGetErrorName,
} from './cuda-device';

export {
  cudaMalloc,
  cudaMallocPitch,
  cudaMalloc3D,
  cudaMallocHost,
  cudaMallocManaged,
  cudaFree,
  cudaFreeHost,
  cudaMemcpy,
  cudaMemcpyAsync,
  cudaMemset,
  cudaMemset2D,
  cudaMemset3D,
  cudaMemcpy2D,
  cudaMemcpy3D,
  cudaMemGetInfo,
  cudaPointerGetAttributes,
} from './cuda-memory';

export {
  cudaStreamCreate,
  cudaStreamCreateWithFlags,
  cudaStreamCreateWithPriority,
  cudaStreamDestroy,
  cudaStreamSynchronize,
  cudaStreamQuery,
  cudaStreamWaitEvent,
  cudaStreamGetPriority,
  cudaStreamGetFlags,
  cudaStreamBeginCapture,
  cudaStreamEndCapture,
  cudaStreamIsCapturing,
  cudaStreamAddCallback,
  cudaEventCreate,
  cudaEventCreateWithFlags,
  cudaEventDestroy,
  cudaEventRecord,
  cudaEventSynchronize,
  cudaEventQuery,
  cudaEventElapsedTime,
  getDefaultStream,
  CudaStreamFlags,
  CudaEventFlags,
  CudaStreamCaptureStatus,
} from './cuda-stream';

export {
  dim3,
  cudaCreateKernel,
  cudaLaunchKernel,
  cudaLaunch,
  cudaGetKernel,
  cudaOccupancyMaxActiveBlocksPerMultiprocessor,
  cudaOccupancyMaxPotentialBlockSize,
  cudaLaunchCooperativeKernel,
  cudaFuncGetAttributes,
  cudaDestroyKernel,
} from './cuda-kernel';
