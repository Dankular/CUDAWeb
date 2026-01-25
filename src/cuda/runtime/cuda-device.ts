/**
 * CUDA Device Management API for WebGPU
 *
 * Provides CUDA-compatible device management functions mapped to WebGPU
 */

export interface CudaDeviceProperties {
  name: string;
  totalGlobalMem: number;
  sharedMemPerBlock: number;
  regsPerBlock: number;
  warpSize: number;
  maxThreadsPerBlock: number;
  maxThreadsDim: [number, number, number];
  maxGridSize: [number, number, number];
  clockRate: number;
  totalConstMem: number;
  major: number;
  minor: number;
  multiProcessorCount: number;
  l2CacheSize: number;
  maxThreadsPerMultiProcessor: number;
  computeMode: number;
  concurrentKernels: number;
  asyncEngineCount: number;
  unifiedAddressing: boolean;
  memoryClockRate: number;
  memoryBusWidth: number;
  maxSharedMemoryPerMultiprocessor: number;
}

export interface CudaDeviceProp extends CudaDeviceProperties {
  uuid: string;
  luid: string;
  pciDeviceID: number;
  pciBusID: number;
  pciDomainID: number;
}

export enum CudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorLaunchFailure = 4,
  cudaErrorPriorLaunchFailure = 5,
  cudaErrorLaunchTimeout = 6,
  cudaErrorLaunchOutOfResources = 7,
  cudaErrorInvalidDeviceFunction = 8,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidMemcpyDirection = 11,
  cudaErrorUnknown = 30,
  cudaErrorNotReady = 34,
  cudaErrorNoDevice = 38,
  cudaErrorECCUncorrectable = 39,
  cudaErrorDevicesUnavailable = 46,
  cudaErrorPeerAccessAlreadyEnabled = 50,
  cudaErrorPeerAccessNotEnabled = 51,
  cudaErrorDeviceAlreadyInUse = 54,
  cudaErrorAssert = 59,
  cudaErrorHostMemoryAlreadyRegistered = 63,
  cudaErrorHostMemoryNotRegistered = 64,
  cudaErrorNotSupported = 71,
  cudaErrorSystemNotReady = 76,
}

export enum CudaComputeMode {
  cudaComputeModeDefault = 0,
  cudaComputeModeExclusive = 1,
  cudaComputeModeProhibited = 2,
  cudaComputeModeExclusiveProcess = 3,
}

export enum CudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
}

export enum CudaDeviceAttr {
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxBlockDimX = 2,
  cudaDevAttrMaxBlockDimY = 3,
  cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5,
  cudaDevAttrMaxGridDimY = 6,
  cudaDevAttrMaxGridDimZ = 7,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrTotalConstantMemory = 9,
  cudaDevAttrWarpSize = 10,
  cudaDevAttrMaxRegistersPerBlock = 12,
  cudaDevAttrClockRate = 13,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
  cudaDevAttrMaxWarpsPerMultiprocessor = 82,
}

const ERROR_STRINGS: Record<CudaError, string> = {
  [CudaError.cudaSuccess]: 'no error',
  [CudaError.cudaErrorInvalidValue]: 'invalid argument',
  [CudaError.cudaErrorMemoryAllocation]: 'out of memory',
  [CudaError.cudaErrorInitializationError]: 'initialization error',
  [CudaError.cudaErrorLaunchFailure]: 'unspecified launch failure',
  [CudaError.cudaErrorPriorLaunchFailure]: 'unspecified launch failure',
  [CudaError.cudaErrorLaunchTimeout]: 'the launch timed out and was terminated',
  [CudaError.cudaErrorLaunchOutOfResources]: 'too many resources requested for launch',
  [CudaError.cudaErrorInvalidDeviceFunction]: 'invalid device function',
  [CudaError.cudaErrorInvalidConfiguration]: 'invalid configuration argument',
  [CudaError.cudaErrorInvalidDevice]: 'invalid device ordinal',
  [CudaError.cudaErrorInvalidMemcpyDirection]: 'invalid memcpy direction',
  [CudaError.cudaErrorUnknown]: 'unknown error',
  [CudaError.cudaErrorNotReady]: 'device not ready',
  [CudaError.cudaErrorNoDevice]: 'no CUDA-capable device is detected',
  [CudaError.cudaErrorECCUncorrectable]: 'uncorrectable ECC error',
  [CudaError.cudaErrorDevicesUnavailable]: 'all CUDA devices are busy or unavailable',
  [CudaError.cudaErrorPeerAccessAlreadyEnabled]: 'peer access is already enabled',
  [CudaError.cudaErrorPeerAccessNotEnabled]: 'peer access has not been enabled',
  [CudaError.cudaErrorDeviceAlreadyInUse]: 'device already in use',
  [CudaError.cudaErrorAssert]: 'device-side assert triggered',
  [CudaError.cudaErrorHostMemoryAlreadyRegistered]: 'host memory already registered',
  [CudaError.cudaErrorHostMemoryNotRegistered]: 'host memory not registered',
  [CudaError.cudaErrorNotSupported]: 'operation not supported',
  [CudaError.cudaErrorSystemNotReady]: 'system not ready',
};

export class CudaDevice {
  private static instance: CudaDevice | null = null;
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private currentDevice: number = 0;
  private lastError: CudaError = CudaError.cudaSuccess;
  private devices: GPUAdapter[] = [];
  private deviceProperties: Map<number, CudaDeviceProp> = new Map();
  private initialized: boolean = false;

  private constructor() {}

  static getInstance(): CudaDevice {
    if (!CudaDevice.instance) {
      CudaDevice.instance = new CudaDevice();
    }
    return CudaDevice.instance;
  }

  async initialize(): Promise<CudaError> {
    if (this.initialized) {
      return CudaError.cudaSuccess;
    }

    try {
      if (!navigator.gpu) {
        this.lastError = CudaError.cudaErrorNoDevice;
        return this.lastError;
      }

      // Get all available adapters
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!adapter) {
        this.lastError = CudaError.cudaErrorNoDevice;
        return this.lastError;
      }

      this.adapter = adapter;
      this.devices = [adapter];

      // Request device with maximum features
      this.device = await adapter.requestDevice({
        requiredFeatures: this.getRequiredFeatures(adapter),
        requiredLimits: this.getRequiredLimits(adapter),
      });

      // Build device properties
      this.buildDeviceProperties(0, adapter);
      this.initialized = true;
      this.lastError = CudaError.cudaSuccess;
      return this.lastError;
    } catch (error) {
      console.error('CUDA initialization error:', error);
      this.lastError = CudaError.cudaErrorInitializationError;
      return this.lastError;
    }
  }

  private getRequiredFeatures(adapter: GPUAdapter): GPUFeatureName[] {
    const features: GPUFeatureName[] = [];
    const supportedFeatures = adapter.features;

    // Request all available features
    const desiredFeatures: GPUFeatureName[] = [
      'shader-f16',
      'timestamp-query',
      'float32-filterable',
    ];

    for (const feature of desiredFeatures) {
      if (supportedFeatures.has(feature)) {
        features.push(feature);
      }
    }

    return features;
  }

  private getRequiredLimits(adapter: GPUAdapter): Record<string, number> {
    const limits: Record<string, number> = {};
    const adapterLimits = adapter.limits;

    // Request maximum limits
    limits.maxStorageBufferBindingSize = adapterLimits.maxStorageBufferBindingSize;
    limits.maxBufferSize = adapterLimits.maxBufferSize;
    limits.maxComputeWorkgroupStorageSize = adapterLimits.maxComputeWorkgroupStorageSize;
    limits.maxComputeInvocationsPerWorkgroup = adapterLimits.maxComputeInvocationsPerWorkgroup;
    limits.maxComputeWorkgroupSizeX = adapterLimits.maxComputeWorkgroupSizeX;
    limits.maxComputeWorkgroupSizeY = adapterLimits.maxComputeWorkgroupSizeY;
    limits.maxComputeWorkgroupSizeZ = adapterLimits.maxComputeWorkgroupSizeZ;
    limits.maxComputeWorkgroupsPerDimension = adapterLimits.maxComputeWorkgroupsPerDimension;
    limits.maxBindGroups = adapterLimits.maxBindGroups;
    limits.maxBindingsPerBindGroup = adapterLimits.maxBindingsPerBindGroup;
    limits.maxStorageBuffersPerShaderStage = adapterLimits.maxStorageBuffersPerShaderStage;
    limits.maxUniformBufferBindingSize = adapterLimits.maxUniformBufferBindingSize;

    return limits;
  }

  private buildDeviceProperties(deviceId: number, adapter: GPUAdapter): void {
    const info = adapter.info;
    const limits = adapter.limits;

    const props: CudaDeviceProp = {
      name: info.device || 'WebGPU Device',
      totalGlobalMem: limits.maxBufferSize,
      sharedMemPerBlock: limits.maxComputeWorkgroupStorageSize,
      regsPerBlock: 65536, // Estimated
      warpSize: 32, // WebGPU subgroup size (typically 32)
      maxThreadsPerBlock: limits.maxComputeInvocationsPerWorkgroup,
      maxThreadsDim: [
        limits.maxComputeWorkgroupSizeX,
        limits.maxComputeWorkgroupSizeY,
        limits.maxComputeWorkgroupSizeZ,
      ],
      maxGridSize: [
        limits.maxComputeWorkgroupsPerDimension,
        limits.maxComputeWorkgroupsPerDimension,
        limits.maxComputeWorkgroupsPerDimension,
      ],
      clockRate: 1000000, // Unknown, default to 1GHz
      totalConstMem: limits.maxUniformBufferBindingSize,
      major: 8, // WebGPU equivalent to CUDA 8.x capabilities
      minor: 0,
      multiProcessorCount: 1, // Unknown
      l2CacheSize: 0, // Unknown
      maxThreadsPerMultiProcessor: limits.maxComputeInvocationsPerWorkgroup,
      computeMode: CudaComputeMode.cudaComputeModeDefault,
      concurrentKernels: 1,
      asyncEngineCount: 1,
      unifiedAddressing: true,
      memoryClockRate: 0,
      memoryBusWidth: 0,
      maxSharedMemoryPerMultiprocessor: limits.maxComputeWorkgroupStorageSize,
      uuid: crypto.randomUUID(),
      luid: '',
      pciDeviceID: 0,
      pciBusID: 0,
      pciDomainID: 0,
    };

    // Adjust properties based on vendor
    if (info.vendor?.toLowerCase().includes('nvidia')) {
      props.multiProcessorCount = 68; // Typical for RTX GPUs
      props.clockRate = 1800000; // 1.8GHz typical
    } else if (info.vendor?.toLowerCase().includes('amd')) {
      props.multiProcessorCount = 60;
      props.clockRate = 2100000;
    } else if (info.vendor?.toLowerCase().includes('intel')) {
      props.multiProcessorCount = 32;
      props.clockRate = 1650000;
    } else if (info.vendor?.toLowerCase().includes('apple')) {
      props.multiProcessorCount = 10;
      props.clockRate = 1400000;
    }

    this.deviceProperties.set(deviceId, props);
  }

  // CUDA Runtime API Methods

  cudaGetDeviceCount(): { count: number; error: CudaError } {
    if (!this.initialized) {
      return { count: 0, error: CudaError.cudaErrorInitializationError };
    }
    return { count: this.devices.length, error: CudaError.cudaSuccess };
  }

  cudaGetDevice(): { device: number; error: CudaError } {
    if (!this.initialized) {
      return { device: -1, error: CudaError.cudaErrorInitializationError };
    }
    return { device: this.currentDevice, error: CudaError.cudaSuccess };
  }

  cudaSetDevice(device: number): CudaError {
    if (!this.initialized) {
      this.lastError = CudaError.cudaErrorInitializationError;
      return this.lastError;
    }
    if (device < 0 || device >= this.devices.length) {
      this.lastError = CudaError.cudaErrorInvalidDevice;
      return this.lastError;
    }
    this.currentDevice = device;
    return CudaError.cudaSuccess;
  }

  cudaGetDeviceProperties(device: number): { props: CudaDeviceProp | null; error: CudaError } {
    if (!this.initialized) {
      return { props: null, error: CudaError.cudaErrorInitializationError };
    }
    if (device < 0 || device >= this.devices.length) {
      return { props: null, error: CudaError.cudaErrorInvalidDevice };
    }
    const props = this.deviceProperties.get(device);
    if (!props) {
      return { props: null, error: CudaError.cudaErrorInvalidDevice };
    }
    return { props, error: CudaError.cudaSuccess };
  }

  cudaDeviceGetAttribute(attr: CudaDeviceAttr, device: number): { value: number; error: CudaError } {
    const { props, error } = this.cudaGetDeviceProperties(device);
    if (error !== CudaError.cudaSuccess || !props) {
      return { value: 0, error };
    }

    let value = 0;
    switch (attr) {
      case CudaDeviceAttr.cudaDevAttrMaxThreadsPerBlock:
        value = props.maxThreadsPerBlock;
        break;
      case CudaDeviceAttr.cudaDevAttrMaxBlockDimX:
        value = props.maxThreadsDim[0];
        break;
      case CudaDeviceAttr.cudaDevAttrMaxBlockDimY:
        value = props.maxThreadsDim[1];
        break;
      case CudaDeviceAttr.cudaDevAttrMaxBlockDimZ:
        value = props.maxThreadsDim[2];
        break;
      case CudaDeviceAttr.cudaDevAttrMaxGridDimX:
        value = props.maxGridSize[0];
        break;
      case CudaDeviceAttr.cudaDevAttrMaxGridDimY:
        value = props.maxGridSize[1];
        break;
      case CudaDeviceAttr.cudaDevAttrMaxGridDimZ:
        value = props.maxGridSize[2];
        break;
      case CudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlock:
        value = props.sharedMemPerBlock;
        break;
      case CudaDeviceAttr.cudaDevAttrTotalConstantMemory:
        value = props.totalConstMem;
        break;
      case CudaDeviceAttr.cudaDevAttrWarpSize:
        value = props.warpSize;
        break;
      case CudaDeviceAttr.cudaDevAttrMaxRegistersPerBlock:
        value = props.regsPerBlock;
        break;
      case CudaDeviceAttr.cudaDevAttrClockRate:
        value = props.clockRate;
        break;
      case CudaDeviceAttr.cudaDevAttrMultiProcessorCount:
        value = props.multiProcessorCount;
        break;
      case CudaDeviceAttr.cudaDevAttrComputeCapabilityMajor:
        value = props.major;
        break;
      case CudaDeviceAttr.cudaDevAttrComputeCapabilityMinor:
        value = props.minor;
        break;
      case CudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor:
        value = props.maxSharedMemoryPerMultiprocessor;
        break;
      case CudaDeviceAttr.cudaDevAttrMaxWarpsPerMultiprocessor:
        value = Math.floor(props.maxThreadsPerMultiProcessor / props.warpSize);
        break;
      default:
        return { value: 0, error: CudaError.cudaErrorInvalidValue };
    }

    return { value, error: CudaError.cudaSuccess };
  }

  cudaDeviceSynchronize(): CudaError {
    // In WebGPU, synchronization happens at command buffer submission
    // This is a no-op as we handle synchronization differently
    return CudaError.cudaSuccess;
  }

  cudaDeviceReset(): CudaError {
    if (!this.initialized) {
      return CudaError.cudaErrorInitializationError;
    }
    // Reset is handled by destroying and recreating the device
    this.device?.destroy();
    this.initialized = false;
    return CudaError.cudaSuccess;
  }

  cudaGetLastError(): CudaError {
    const error = this.lastError;
    this.lastError = CudaError.cudaSuccess;
    return error;
  }

  cudaPeekAtLastError(): CudaError {
    return this.lastError;
  }

  cudaGetErrorString(error: CudaError): string {
    return ERROR_STRINGS[error] || 'unknown error';
  }

  cudaGetErrorName(error: CudaError): string {
    return CudaError[error] || 'cudaErrorUnknown';
  }

  // Internal methods for other CUDA modules
  getDevice(): GPUDevice | null {
    return this.device;
  }

  getAdapter(): GPUAdapter | null {
    return this.adapter;
  }

  setLastError(error: CudaError): void {
    this.lastError = error;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  hasFeature(feature: GPUFeatureName): boolean {
    return this.device?.features.has(feature) ?? false;
  }

  getLimit(limit: keyof GPUSupportedLimits): number {
    return this.device?.limits[limit] ?? 0;
  }
}

// Export singleton functions for CUDA-like API
export const cudaDevice = CudaDevice.getInstance();

export async function cudaInit(): Promise<CudaError> {
  return cudaDevice.initialize();
}

export function cudaGetDeviceCount(): { count: number; error: CudaError } {
  return cudaDevice.cudaGetDeviceCount();
}

export function cudaGetDevice(): { device: number; error: CudaError } {
  return cudaDevice.cudaGetDevice();
}

export function cudaSetDevice(device: number): CudaError {
  return cudaDevice.cudaSetDevice(device);
}

export function cudaGetDeviceProperties(device: number): { props: CudaDeviceProp | null; error: CudaError } {
  return cudaDevice.cudaGetDeviceProperties(device);
}

export function cudaDeviceGetAttribute(attr: CudaDeviceAttr, device: number): { value: number; error: CudaError } {
  return cudaDevice.cudaDeviceGetAttribute(attr, device);
}

export function cudaDeviceSynchronize(): CudaError {
  return cudaDevice.cudaDeviceSynchronize();
}

export function cudaDeviceReset(): CudaError {
  return cudaDevice.cudaDeviceReset();
}

export function cudaGetLastError(): CudaError {
  return cudaDevice.cudaGetLastError();
}

export function cudaPeekAtLastError(): CudaError {
  return cudaDevice.cudaPeekAtLastError();
}

export function cudaGetErrorString(error: CudaError): string {
  return cudaDevice.cudaGetErrorString(error);
}

export function cudaGetErrorName(error: CudaError): string {
  return cudaDevice.cudaGetErrorName(error);
}
