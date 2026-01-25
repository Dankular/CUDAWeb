/**
 * CUDA Texture and Surface Operations for WebGPU
 *
 * Provides CUDA-compatible texture binding and sampling
 */

import { CudaError, cudaDevice } from '../runtime/cuda-device';

export enum CudaTextureAddressMode {
  cudaAddressModeWrap = 0,
  cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2,
  cudaAddressModeBorder = 3,
}

export enum CudaTextureFilterMode {
  cudaFilterModePoint = 0,
  cudaFilterModeLinear = 1,
}

export enum CudaTextureReadMode {
  cudaReadModeElementType = 0,
  cudaReadModeNormalizedFloat = 1,
}

export enum CudaChannelFormatKind {
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3,
}

export interface CudaChannelFormatDesc {
  x: number;
  y: number;
  z: number;
  w: number;
  f: CudaChannelFormatKind;
}

export interface CudaTextureDesc {
  addressMode: [CudaTextureAddressMode, CudaTextureAddressMode, CudaTextureAddressMode];
  filterMode: CudaTextureFilterMode;
  readMode: CudaTextureReadMode;
  sRGB: boolean;
  borderColor: [number, number, number, number];
  normalizedCoords: boolean;
  maxAnisotropy: number;
  mipmapFilterMode: CudaTextureFilterMode;
  mipmapLevelBias: number;
  minMipmapLevelClamp: number;
  maxMipmapLevelClamp: number;
}

export interface CudaResourceDesc {
  resType: 'array' | 'mipmap' | 'linear' | 'pitch2D';
  array?: GPUTexture;
  format?: GPUTextureFormat;
  width?: number;
  height?: number;
  depth?: number;
  pitchInBytes?: number;
  sizeInBytes?: number;
}

export interface CudaTextureObject {
  id: number;
  texture: GPUTexture;
  sampler: GPUSampler;
  view: GPUTextureView;
  desc: CudaTextureDesc;
  channelDesc: CudaChannelFormatDesc;
}

export interface CudaSurfaceObject {
  id: number;
  texture: GPUTexture;
  view: GPUTextureView;
  format: GPUTextureFormat;
}

export interface CudaArray {
  id: number;
  texture: GPUTexture;
  format: GPUTextureFormat;
  width: number;
  height: number;
  depth: number;
  layered: boolean;
}

class CudaTextureManager {
  private static instance: CudaTextureManager | null = null;
  private textures: Map<number, CudaTextureObject> = new Map();
  private surfaces: Map<number, CudaSurfaceObject> = new Map();
  private arrays: Map<number, CudaArray> = new Map();
  private nextId: number = 1;

  private constructor() {}

  static getInstance(): CudaTextureManager {
    if (!CudaTextureManager.instance) {
      CudaTextureManager.instance = new CudaTextureManager();
    }
    return CudaTextureManager.instance;
  }

  private getDevice(): GPUDevice | null {
    return cudaDevice.getDevice();
  }

  // Create default texture descriptor
  getDefaultTextureDesc(): CudaTextureDesc {
    return {
      addressMode: [
        CudaTextureAddressMode.cudaAddressModeClamp,
        CudaTextureAddressMode.cudaAddressModeClamp,
        CudaTextureAddressMode.cudaAddressModeClamp,
      ],
      filterMode: CudaTextureFilterMode.cudaFilterModeLinear,
      readMode: CudaTextureReadMode.cudaReadModeNormalizedFloat,
      sRGB: false,
      borderColor: [0, 0, 0, 0],
      normalizedCoords: true,
      maxAnisotropy: 1,
      mipmapFilterMode: CudaTextureFilterMode.cudaFilterModeLinear,
      mipmapLevelBias: 0,
      minMipmapLevelClamp: 0,
      maxMipmapLevelClamp: 1000,
    };
  }

  // Map CUDA address mode to WebGPU
  private mapAddressMode(mode: CudaTextureAddressMode): GPUAddressMode {
    switch (mode) {
      case CudaTextureAddressMode.cudaAddressModeWrap:
        return 'repeat';
      case CudaTextureAddressMode.cudaAddressModeClamp:
        return 'clamp-to-edge';
      case CudaTextureAddressMode.cudaAddressModeMirror:
        return 'mirror-repeat';
      case CudaTextureAddressMode.cudaAddressModeBorder:
        return 'clamp-to-edge'; // WebGPU doesn't have border
      default:
        return 'clamp-to-edge';
    }
  }

  // Map CUDA filter mode to WebGPU
  private mapFilterMode(mode: CudaTextureFilterMode): GPUFilterMode {
    return mode === CudaTextureFilterMode.cudaFilterModeLinear
      ? 'linear'
      : 'nearest';
  }

  // Create CUDA array (texture backing)
  cudaMallocArray(
    width: number,
    height: number,
    channelDesc: CudaChannelFormatDesc,
    flags: number = 0
  ): { array: CudaArray | null; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { array: null, error: CudaError.cudaErrorInitializationError };
    }

    try {
      const format = this.channelDescToFormat(channelDesc);

      const texture = device.createTexture({
        size: { width, height, depthOrArrayLayers: 1 },
        format,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.STORAGE_BINDING,
        dimension: '2d',
      });

      const array: CudaArray = {
        id: this.nextId++,
        texture,
        format,
        width,
        height,
        depth: 1,
        layered: false,
      };

      this.arrays.set(array.id, array);
      return { array, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('cudaMallocArray error:', error);
      return { array: null, error: CudaError.cudaErrorMemoryAllocation };
    }
  }

  cudaMalloc3DArray(
    width: number,
    height: number,
    depth: number,
    channelDesc: CudaChannelFormatDesc,
    flags: number = 0
  ): { array: CudaArray | null; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { array: null, error: CudaError.cudaErrorInitializationError };
    }

    try {
      const format = this.channelDescToFormat(channelDesc);

      const texture = device.createTexture({
        size: { width, height, depthOrArrayLayers: depth },
        format,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.STORAGE_BINDING,
        dimension: '3d',
      });

      const array: CudaArray = {
        id: this.nextId++,
        texture,
        format,
        width,
        height,
        depth,
        layered: false,
      };

      this.arrays.set(array.id, array);
      return { array, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('cudaMalloc3DArray error:', error);
      return { array: null, error: CudaError.cudaErrorMemoryAllocation };
    }
  }

  cudaFreeArray(array: CudaArray): CudaError {
    const a = this.arrays.get(array.id);
    if (!a) {
      return CudaError.cudaErrorInvalidValue;
    }
    a.texture.destroy();
    this.arrays.delete(array.id);
    return CudaError.cudaSuccess;
  }

  // Create texture object
  cudaCreateTextureObject(
    resDesc: CudaResourceDesc,
    texDesc: CudaTextureDesc,
    channelDesc?: CudaChannelFormatDesc
  ): { texObj: CudaTextureObject | null; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { texObj: null, error: CudaError.cudaErrorInitializationError };
    }

    try {
      let texture: GPUTexture;
      let format: GPUTextureFormat;

      if (resDesc.array) {
        texture = resDesc.array;
        format = resDesc.format || 'rgba8unorm';
      } else {
        return { texObj: null, error: CudaError.cudaErrorInvalidValue };
      }

      // Create sampler
      const sampler = device.createSampler({
        addressModeU: this.mapAddressMode(texDesc.addressMode[0]),
        addressModeV: this.mapAddressMode(texDesc.addressMode[1]),
        addressModeW: this.mapAddressMode(texDesc.addressMode[2]),
        magFilter: this.mapFilterMode(texDesc.filterMode),
        minFilter: this.mapFilterMode(texDesc.filterMode),
        mipmapFilter: this.mapFilterMode(texDesc.mipmapFilterMode),
        maxAnisotropy: texDesc.maxAnisotropy,
      });

      // Create view
      const view = texture.createView();

      const texObj: CudaTextureObject = {
        id: this.nextId++,
        texture,
        sampler,
        view,
        desc: texDesc,
        channelDesc: channelDesc || this.formatToChannelDesc(format),
      };

      this.textures.set(texObj.id, texObj);
      return { texObj, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('cudaCreateTextureObject error:', error);
      return { texObj: null, error: CudaError.cudaErrorUnknown };
    }
  }

  cudaDestroyTextureObject(texObj: CudaTextureObject): CudaError {
    const t = this.textures.get(texObj.id);
    if (!t) {
      return CudaError.cudaErrorInvalidValue;
    }
    this.textures.delete(texObj.id);
    return CudaError.cudaSuccess;
  }

  // Create surface object
  cudaCreateSurfaceObject(
    resDesc: CudaResourceDesc
  ): { surfObj: CudaSurfaceObject | null; error: CudaError } {
    const device = this.getDevice();
    if (!device) {
      return { surfObj: null, error: CudaError.cudaErrorInitializationError };
    }

    try {
      let texture: GPUTexture;
      let format: GPUTextureFormat;

      if (resDesc.array) {
        texture = resDesc.array;
        format = resDesc.format || 'rgba8unorm';
      } else {
        return { surfObj: null, error: CudaError.cudaErrorInvalidValue };
      }

      const view = texture.createView({
        format,
        dimension: '2d',
      });

      const surfObj: CudaSurfaceObject = {
        id: this.nextId++,
        texture,
        view,
        format,
      };

      this.surfaces.set(surfObj.id, surfObj);
      return { surfObj, error: CudaError.cudaSuccess };
    } catch (error) {
      console.error('cudaCreateSurfaceObject error:', error);
      return { surfObj: null, error: CudaError.cudaErrorUnknown };
    }
  }

  cudaDestroySurfaceObject(surfObj: CudaSurfaceObject): CudaError {
    const s = this.surfaces.get(surfObj.id);
    if (!s) {
      return CudaError.cudaErrorInvalidValue;
    }
    this.surfaces.delete(surfObj.id);
    return CudaError.cudaSuccess;
  }

  // Copy to array
  async cudaMemcpyToArray(
    dst: CudaArray,
    wOffset: number,
    hOffset: number,
    src: ArrayBuffer,
    count: number
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const array = this.arrays.get(dst.id);
    if (!array) return CudaError.cudaErrorInvalidValue;

    try {
      const bytesPerPixel = this.getBytesPerPixel(array.format);
      const rowPixels = array.width;
      const srcData = new Uint8Array(src, 0, count);

      device.queue.writeTexture(
        { texture: array.texture, origin: { x: wOffset, y: hOffset, z: 0 } },
        srcData,
        { bytesPerRow: rowPixels * bytesPerPixel, rowsPerImage: array.height },
        { width: rowPixels, height: Math.ceil(count / (rowPixels * bytesPerPixel)), depthOrArrayLayers: 1 }
      );

      return CudaError.cudaSuccess;
    } catch (error) {
      console.error('cudaMemcpyToArray error:', error);
      return CudaError.cudaErrorUnknown;
    }
  }

  async cudaMemcpy2DToArray(
    dst: CudaArray,
    wOffset: number,
    hOffset: number,
    src: ArrayBuffer,
    spitch: number,
    width: number,
    height: number
  ): Promise<CudaError> {
    const device = this.getDevice();
    if (!device) return CudaError.cudaErrorInitializationError;

    const array = this.arrays.get(dst.id);
    if (!array) return CudaError.cudaErrorInvalidValue;

    try {
      const srcData = new Uint8Array(src);

      device.queue.writeTexture(
        { texture: array.texture, origin: { x: wOffset, y: hOffset, z: 0 } },
        srcData,
        { bytesPerRow: spitch, rowsPerImage: height },
        { width, height, depthOrArrayLayers: 1 }
      );

      return CudaError.cudaSuccess;
    } catch (error) {
      console.error('cudaMemcpy2DToArray error:', error);
      return CudaError.cudaErrorUnknown;
    }
  }

  // Helper: Convert channel description to format
  private channelDescToFormat(desc: CudaChannelFormatDesc): GPUTextureFormat {
    const channels = (desc.x > 0 ? 1 : 0) + (desc.y > 0 ? 1 : 0) +
                     (desc.z > 0 ? 1 : 0) + (desc.w > 0 ? 1 : 0);
    const bits = desc.x;

    if (desc.f === CudaChannelFormatKind.cudaChannelFormatKindFloat) {
      if (bits === 32) {
        switch (channels) {
          case 1: return 'r32float';
          case 2: return 'rg32float';
          case 4: return 'rgba32float';
        }
      } else if (bits === 16) {
        switch (channels) {
          case 1: return 'r16float';
          case 2: return 'rg16float';
          case 4: return 'rgba16float';
        }
      }
    } else if (desc.f === CudaChannelFormatKind.cudaChannelFormatKindUnsigned) {
      if (bits === 8) {
        switch (channels) {
          case 1: return 'r8unorm';
          case 2: return 'rg8unorm';
          case 4: return 'rgba8unorm';
        }
      } else if (bits === 16) {
        switch (channels) {
          case 1: return 'r16uint';
          case 2: return 'rg16uint';
          case 4: return 'rgba16uint';
        }
      } else if (bits === 32) {
        switch (channels) {
          case 1: return 'r32uint';
          case 2: return 'rg32uint';
          case 4: return 'rgba32uint';
        }
      }
    } else if (desc.f === CudaChannelFormatKind.cudaChannelFormatKindSigned) {
      if (bits === 8) {
        switch (channels) {
          case 1: return 'r8snorm';
          case 2: return 'rg8snorm';
          case 4: return 'rgba8snorm';
        }
      } else if (bits === 16) {
        switch (channels) {
          case 1: return 'r16sint';
          case 2: return 'rg16sint';
          case 4: return 'rgba16sint';
        }
      } else if (bits === 32) {
        switch (channels) {
          case 1: return 'r32sint';
          case 2: return 'rg32sint';
          case 4: return 'rgba32sint';
        }
      }
    }

    return 'rgba8unorm';
  }

  // Helper: Convert format to channel description
  private formatToChannelDesc(format: GPUTextureFormat): CudaChannelFormatDesc {
    const desc: CudaChannelFormatDesc = {
      x: 0, y: 0, z: 0, w: 0,
      f: CudaChannelFormatKind.cudaChannelFormatKindFloat
    };

    switch (format) {
      case 'rgba8unorm':
      case 'rgba8snorm':
        desc.x = desc.y = desc.z = desc.w = 8;
        desc.f = format.includes('snorm')
          ? CudaChannelFormatKind.cudaChannelFormatKindSigned
          : CudaChannelFormatKind.cudaChannelFormatKindUnsigned;
        break;
      case 'rgba16float':
        desc.x = desc.y = desc.z = desc.w = 16;
        desc.f = CudaChannelFormatKind.cudaChannelFormatKindFloat;
        break;
      case 'rgba32float':
        desc.x = desc.y = desc.z = desc.w = 32;
        desc.f = CudaChannelFormatKind.cudaChannelFormatKindFloat;
        break;
      case 'r32float':
        desc.x = 32;
        desc.f = CudaChannelFormatKind.cudaChannelFormatKindFloat;
        break;
      case 'rg32float':
        desc.x = desc.y = 32;
        desc.f = CudaChannelFormatKind.cudaChannelFormatKindFloat;
        break;
    }

    return desc;
  }

  // Helper: Get bytes per pixel
  private getBytesPerPixel(format: GPUTextureFormat): number {
    switch (format) {
      case 'r8unorm':
      case 'r8snorm':
      case 'r8uint':
      case 'r8sint':
        return 1;
      case 'r16float':
      case 'r16uint':
      case 'r16sint':
      case 'rg8unorm':
      case 'rg8snorm':
        return 2;
      case 'r32float':
      case 'r32uint':
      case 'r32sint':
      case 'rg16float':
      case 'rg16uint':
      case 'rgba8unorm':
      case 'rgba8snorm':
      case 'rgba8uint':
      case 'rgba8sint':
        return 4;
      case 'rg32float':
      case 'rg32uint':
      case 'rgba16float':
      case 'rgba16uint':
        return 8;
      case 'rgba32float':
      case 'rgba32uint':
        return 16;
      default:
        return 4;
    }
  }

  // Get texture/surface for binding
  getTextureObject(id: number): CudaTextureObject | undefined {
    return this.textures.get(id);
  }

  getSurfaceObject(id: number): CudaSurfaceObject | undefined {
    return this.surfaces.get(id);
  }

  getArray(id: number): CudaArray | undefined {
    return this.arrays.get(id);
  }

  // Cleanup
  clearAll(): void {
    for (const [, array] of this.arrays) {
      array.texture.destroy();
    }
    this.textures.clear();
    this.surfaces.clear();
    this.arrays.clear();
  }
}

// WGSL texture sampling functions
export const CUDA_TEXTURE_WGSL = `
// Texture sampling functions for CUDA compatibility

// tex1D - 1D texture fetch
fn tex1D(t: texture_1d<f32>, s: sampler, x: f32) -> vec4<f32> {
    return textureSample(t, s, x);
}

// tex2D - 2D texture fetch
fn tex2D(t: texture_2d<f32>, s: sampler, x: f32, y: f32) -> vec4<f32> {
    return textureSample(t, s, vec2<f32>(x, y));
}

// tex3D - 3D texture fetch
fn tex3D(t: texture_3d<f32>, s: sampler, x: f32, y: f32, z: f32) -> vec4<f32> {
    return textureSample(t, s, vec3<f32>(x, y, z));
}

// texCube - Cubemap texture fetch
fn texCube(t: texture_cube<f32>, s: sampler, x: f32, y: f32, z: f32) -> vec4<f32> {
    return textureSample(t, s, vec3<f32>(x, y, z));
}

// tex2DLod - 2D texture fetch with LOD
fn tex2DLod(t: texture_2d<f32>, s: sampler, x: f32, y: f32, lod: f32) -> vec4<f32> {
    return textureSampleLevel(t, s, vec2<f32>(x, y), lod);
}

// tex2DGrad - 2D texture fetch with gradients
fn tex2DGrad(t: texture_2d<f32>, s: sampler, x: f32, y: f32, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    return textureSampleGrad(t, s, vec2<f32>(x, y), ddx, ddy);
}

// tex2DLayered - 2D array texture fetch
fn tex2DLayered(t: texture_2d_array<f32>, s: sampler, x: f32, y: f32, layer: i32) -> vec4<f32> {
    return textureSample(t, s, vec2<f32>(x, y), layer);
}

// Surface read (using storage texture)
fn surf2Dread(t: texture_storage_2d<rgba32float, read>, x: i32, y: i32) -> vec4<f32> {
    return textureLoad(t, vec2<i32>(x, y));
}

// Surface write (using storage texture)
fn surf2Dwrite(t: texture_storage_2d<rgba32float, write>, data: vec4<f32>, x: i32, y: i32) {
    textureStore(t, vec2<i32>(x, y), data);
}
`;

// Export singleton
export const cudaTextureManager = CudaTextureManager.getInstance();

// Export CUDA-like API functions
export function cudaMallocArray(
  width: number,
  height: number,
  channelDesc: CudaChannelFormatDesc,
  flags?: number
): { array: CudaArray | null; error: CudaError } {
  return cudaTextureManager.cudaMallocArray(width, height, channelDesc, flags);
}

export function cudaMalloc3DArray(
  width: number,
  height: number,
  depth: number,
  channelDesc: CudaChannelFormatDesc,
  flags?: number
): { array: CudaArray | null; error: CudaError } {
  return cudaTextureManager.cudaMalloc3DArray(width, height, depth, channelDesc, flags);
}

export function cudaFreeArray(array: CudaArray): CudaError {
  return cudaTextureManager.cudaFreeArray(array);
}

export function cudaCreateTextureObject(
  resDesc: CudaResourceDesc,
  texDesc: CudaTextureDesc,
  channelDesc?: CudaChannelFormatDesc
): { texObj: CudaTextureObject | null; error: CudaError } {
  return cudaTextureManager.cudaCreateTextureObject(resDesc, texDesc, channelDesc);
}

export function cudaDestroyTextureObject(texObj: CudaTextureObject): CudaError {
  return cudaTextureManager.cudaDestroyTextureObject(texObj);
}

export function cudaCreateSurfaceObject(
  resDesc: CudaResourceDesc
): { surfObj: CudaSurfaceObject | null; error: CudaError } {
  return cudaTextureManager.cudaCreateSurfaceObject(resDesc);
}

export function cudaDestroySurfaceObject(surfObj: CudaSurfaceObject): CudaError {
  return cudaTextureManager.cudaDestroySurfaceObject(surfObj);
}

export function cudaMemcpyToArray(
  dst: CudaArray,
  wOffset: number,
  hOffset: number,
  src: ArrayBuffer,
  count: number
): Promise<CudaError> {
  return cudaTextureManager.cudaMemcpyToArray(dst, wOffset, hOffset, src, count);
}

export function cudaMemcpy2DToArray(
  dst: CudaArray,
  wOffset: number,
  hOffset: number,
  src: ArrayBuffer,
  spitch: number,
  width: number,
  height: number
): Promise<CudaError> {
  return cudaTextureManager.cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height);
}

export function getDefaultTextureDesc(): CudaTextureDesc {
  return cudaTextureManager.getDefaultTextureDesc();
}

export function cudaCreateChannelDesc(
  x: number,
  y: number,
  z: number,
  w: number,
  f: CudaChannelFormatKind
): CudaChannelFormatDesc {
  return { x, y, z, w, f };
}

// Helper to include texture functions in shader
export function withCudaTexture(shaderCode: string): string {
  return CUDA_TEXTURE_WGSL + '\n\n' + shaderCode;
}
