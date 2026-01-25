/**
 * Splat File Loader
 * Parses .splat format (compressed Gaussian splatting format)
 * Also supports .ksplat (compressed) and other variants
 */

import { GPUContext, BufferManager } from '../core';

export interface SplatHeader {
  version: number;
  numGaussians: number;
  shDegree: number;
  flags: number;
  compression: 'none' | 'quantized' | 'compressed';
}

export interface SplatData {
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  opacities: Float32Array;
  colors: Float32Array;  // RGB colors (SH degree 0)
  shCoefficients?: Float32Array;  // Higher order SH
  count: number;
  shDegree: number;
}

export interface SplatLoadOptions {
  onProgress?: (loaded: number, total: number) => void;
  loadFullSH?: boolean;  // Load all SH coefficients vs just DC
}

// Magic bytes for different splat formats
const SPLAT_MAGIC = 0x53504C54;  // "SPLT"
const KSPLAT_MAGIC = 0x4B53504C; // "KSPL"

export class SplatLoader {
  private gpu: GPUContext;
  private bufferManager: BufferManager;

  constructor(gpu: GPUContext, bufferManager: BufferManager) {
    this.gpu = gpu;
    this.bufferManager = bufferManager;
  }

  /**
   * Load splat file from URL
   */
  async load(url: string, options: SplatLoadOptions = {}): Promise<SplatData> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load splat: ${response.status}`);
    }

    const buffer = await response.arrayBuffer();
    return this.parse(buffer, options);
  }

  /**
   * Load splat file from File object
   */
  async loadFromFile(file: File, options: SplatLoadOptions = {}): Promise<SplatData> {
    const buffer = await file.arrayBuffer();
    return this.parse(buffer, options);
  }

  /**
   * Parse splat buffer - auto-detects format
   */
  parse(buffer: ArrayBuffer, options: SplatLoadOptions = {}): SplatData {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);

    if (magic === SPLAT_MAGIC) {
      return this.parseSplatV1(buffer, options);
    } else if (magic === KSPLAT_MAGIC) {
      return this.parseKSplat(buffer, options);
    } else {
      // Try legacy format (raw packed data)
      return this.parseLegacySplat(buffer, options);
    }
  }

  /**
   * Parse SPLAT v1 format
   */
  private parseSplatV1(buffer: ArrayBuffer, options: SplatLoadOptions): SplatData {
    const view = new DataView(buffer);
    let offset = 4; // Skip magic

    const version = view.getUint32(offset, true);
    offset += 4;

    const numGaussians = view.getUint32(offset, true);
    offset += 4;

    const shDegree = view.getUint8(offset);
    offset += 1;

    const flags = view.getUint8(offset);
    offset += 1;

    offset += 2; // Padding

    const hasQuantization = (flags & 0x01) !== 0;
    const hasCompression = (flags & 0x02) !== 0;
    const hasFullSH = (flags & 0x04) !== 0 && options.loadFullSH;

    // Allocate arrays
    const positions = new Float32Array(numGaussians * 3);
    const scales = new Float32Array(numGaussians * 3);
    const rotations = new Float32Array(numGaussians * 4);
    const opacities = new Float32Array(numGaussians);
    const colors = new Float32Array(numGaussians * 3);

    let shCoefficients: Float32Array | undefined;
    if (hasFullSH && shDegree > 0) {
      const shCount = (shDegree + 1) * (shDegree + 1) - 1; // Exclude DC
      shCoefficients = new Float32Array(numGaussians * shCount * 3);
    }

    if (hasQuantization) {
      // Read quantization parameters
      const posMin = [
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ];
      offset += 12;

      const posMax = [
        view.getFloat32(offset, true),
        view.getFloat32(offset + 4, true),
        view.getFloat32(offset + 8, true),
      ];
      offset += 12;

      const scaleMin = view.getFloat32(offset, true);
      offset += 4;
      const scaleMax = view.getFloat32(offset, true);
      offset += 4;

      // Parse quantized data
      for (let i = 0; i < numGaussians; i++) {
        // Position (16-bit quantized)
        for (let j = 0; j < 3; j++) {
          const quantized = view.getUint16(offset, true);
          positions[i * 3 + j] = posMin[j] + (quantized / 65535) * (posMax[j] - posMin[j]);
          offset += 2;
        }

        // Scale (8-bit quantized log scale)
        for (let j = 0; j < 3; j++) {
          const quantized = view.getUint8(offset);
          const logScale = scaleMin + (quantized / 255) * (scaleMax - scaleMin);
          scales[i * 3 + j] = Math.exp(logScale);
          offset += 1;
        }

        // Rotation (8-bit quantized quaternion)
        const qx = (view.getInt8(offset) / 127);
        offset += 1;
        const qy = (view.getInt8(offset) / 127);
        offset += 1;
        const qz = (view.getInt8(offset) / 127);
        offset += 1;
        const qw = Math.sqrt(Math.max(0, 1 - qx * qx - qy * qy - qz * qz));

        rotations[i * 4 + 0] = qw;
        rotations[i * 4 + 1] = qx;
        rotations[i * 4 + 2] = qy;
        rotations[i * 4 + 3] = qz;

        // Opacity (8-bit)
        opacities[i] = view.getUint8(offset) / 255;
        offset += 1;

        // Color (8-bit RGB)
        colors[i * 3 + 0] = view.getUint8(offset) / 255;
        offset += 1;
        colors[i * 3 + 1] = view.getUint8(offset) / 255;
        offset += 1;
        colors[i * 3 + 2] = view.getUint8(offset) / 255;
        offset += 1;

        if (options.onProgress && i % 50000 === 0) {
          options.onProgress(i, numGaussians);
        }
      }
    } else {
      // Parse uncompressed data
      for (let i = 0; i < numGaussians; i++) {
        // Position (float32)
        positions[i * 3 + 0] = view.getFloat32(offset, true);
        offset += 4;
        positions[i * 3 + 1] = view.getFloat32(offset, true);
        offset += 4;
        positions[i * 3 + 2] = view.getFloat32(offset, true);
        offset += 4;

        // Scale (float32)
        scales[i * 3 + 0] = view.getFloat32(offset, true);
        offset += 4;
        scales[i * 3 + 1] = view.getFloat32(offset, true);
        offset += 4;
        scales[i * 3 + 2] = view.getFloat32(offset, true);
        offset += 4;

        // Rotation (float32 quaternion)
        rotations[i * 4 + 0] = view.getFloat32(offset, true);
        offset += 4;
        rotations[i * 4 + 1] = view.getFloat32(offset, true);
        offset += 4;
        rotations[i * 4 + 2] = view.getFloat32(offset, true);
        offset += 4;
        rotations[i * 4 + 3] = view.getFloat32(offset, true);
        offset += 4;

        // Opacity (float32)
        opacities[i] = view.getFloat32(offset, true);
        offset += 4;

        // Color (float32 RGB)
        colors[i * 3 + 0] = view.getFloat32(offset, true);
        offset += 4;
        colors[i * 3 + 1] = view.getFloat32(offset, true);
        offset += 4;
        colors[i * 3 + 2] = view.getFloat32(offset, true);
        offset += 4;

        if (options.onProgress && i % 50000 === 0) {
          options.onProgress(i, numGaussians);
        }
      }
    }

    options.onProgress?.(numGaussians, numGaussians);

    return {
      positions,
      scales,
      rotations,
      opacities,
      colors,
      shCoefficients,
      count: numGaussians,
      shDegree: hasFullSH ? shDegree : 0,
    };
  }

  /**
   * Parse KSPLAT (compressed) format
   */
  private parseKSplat(buffer: ArrayBuffer, options: SplatLoadOptions): SplatData {
    // KSplat uses section-based compression
    const view = new DataView(buffer);
    let offset = 4; // Skip magic

    const version = view.getUint32(offset, true);
    offset += 4;

    const numSections = view.getUint32(offset, true);
    offset += 4;

    let numGaussians = 0;
    const sections: Array<{ type: number; offset: number; size: number; count: number }> = [];

    // Read section headers
    for (let i = 0; i < numSections; i++) {
      const type = view.getUint8(offset);
      offset += 1;
      const sectionOffset = view.getUint32(offset, true);
      offset += 4;
      const size = view.getUint32(offset, true);
      offset += 4;
      const count = view.getUint32(offset, true);
      offset += 4;

      sections.push({ type, offset: sectionOffset, size, count });

      if (type === 0) { // Positions section
        numGaussians = count;
      }
    }

    // Allocate arrays
    const positions = new Float32Array(numGaussians * 3);
    const scales = new Float32Array(numGaussians * 3);
    const rotations = new Float32Array(numGaussians * 4);
    const opacities = new Float32Array(numGaussians);
    const colors = new Float32Array(numGaussians * 3);

    // Parse each section
    for (const section of sections) {
      const sectionData = new DataView(buffer, section.offset, section.size);

      switch (section.type) {
        case 0: // Positions
          this.parsePositionsSection(sectionData, positions, section.count);
          break;
        case 1: // Scales
          this.parseScalesSection(sectionData, scales, section.count);
          break;
        case 2: // Rotations
          this.parseRotationsSection(sectionData, rotations, section.count);
          break;
        case 3: // Opacities
          this.parseOpacitiesSection(sectionData, opacities, section.count);
          break;
        case 4: // Colors
          this.parseColorsSection(sectionData, colors, section.count);
          break;
      }
    }

    return {
      positions,
      scales,
      rotations,
      opacities,
      colors,
      count: numGaussians,
      shDegree: 0,
    };
  }

  /**
   * Parse legacy splat format (raw packed data)
   * Format: [x, y, z, scale_x, scale_y, scale_z, r, g, b, a] per splat (32 bytes)
   */
  private parseLegacySplat(buffer: ArrayBuffer, options: SplatLoadOptions): SplatData {
    const BYTES_PER_SPLAT = 32;
    const numGaussians = Math.floor(buffer.byteLength / BYTES_PER_SPLAT);

    const positions = new Float32Array(numGaussians * 3);
    const scales = new Float32Array(numGaussians * 3);
    const rotations = new Float32Array(numGaussians * 4);
    const opacities = new Float32Array(numGaussians);
    const colors = new Float32Array(numGaussians * 3);

    const view = new DataView(buffer);

    for (let i = 0; i < numGaussians; i++) {
      const offset = i * BYTES_PER_SPLAT;

      // Position
      positions[i * 3 + 0] = view.getFloat32(offset, true);
      positions[i * 3 + 1] = view.getFloat32(offset + 4, true);
      positions[i * 3 + 2] = view.getFloat32(offset + 8, true);

      // Scale
      scales[i * 3 + 0] = view.getFloat32(offset + 12, true);
      scales[i * 3 + 1] = view.getFloat32(offset + 16, true);
      scales[i * 3 + 2] = view.getFloat32(offset + 20, true);

      // Color (RGBA as bytes 24-27)
      colors[i * 3 + 0] = view.getUint8(offset + 24) / 255;
      colors[i * 3 + 1] = view.getUint8(offset + 25) / 255;
      colors[i * 3 + 2] = view.getUint8(offset + 26) / 255;
      opacities[i] = view.getUint8(offset + 27) / 255;

      // Default rotation (identity quaternion)
      rotations[i * 4 + 0] = 1;
      rotations[i * 4 + 1] = 0;
      rotations[i * 4 + 2] = 0;
      rotations[i * 4 + 3] = 0;

      if (options.onProgress && i % 50000 === 0) {
        options.onProgress(i, numGaussians);
      }
    }

    options.onProgress?.(numGaussians, numGaussians);

    return {
      positions,
      scales,
      rotations,
      opacities,
      colors,
      count: numGaussians,
      shDegree: 0,
    };
  }

  private parsePositionsSection(view: DataView, positions: Float32Array, count: number): void {
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 0] = view.getFloat32(i * 12, true);
      positions[i * 3 + 1] = view.getFloat32(i * 12 + 4, true);
      positions[i * 3 + 2] = view.getFloat32(i * 12 + 8, true);
    }
  }

  private parseScalesSection(view: DataView, scales: Float32Array, count: number): void {
    for (let i = 0; i < count; i++) {
      scales[i * 3 + 0] = view.getFloat32(i * 12, true);
      scales[i * 3 + 1] = view.getFloat32(i * 12 + 4, true);
      scales[i * 3 + 2] = view.getFloat32(i * 12 + 8, true);
    }
  }

  private parseRotationsSection(view: DataView, rotations: Float32Array, count: number): void {
    for (let i = 0; i < count; i++) {
      rotations[i * 4 + 0] = view.getFloat32(i * 16, true);
      rotations[i * 4 + 1] = view.getFloat32(i * 16 + 4, true);
      rotations[i * 4 + 2] = view.getFloat32(i * 16 + 8, true);
      rotations[i * 4 + 3] = view.getFloat32(i * 16 + 12, true);
    }
  }

  private parseOpacitiesSection(view: DataView, opacities: Float32Array, count: number): void {
    for (let i = 0; i < count; i++) {
      opacities[i] = view.getFloat32(i * 4, true);
    }
  }

  private parseColorsSection(view: DataView, colors: Float32Array, count: number): void {
    for (let i = 0; i < count; i++) {
      colors[i * 3 + 0] = view.getFloat32(i * 12, true);
      colors[i * 3 + 1] = view.getFloat32(i * 12 + 4, true);
      colors[i * 3 + 2] = view.getFloat32(i * 12 + 8, true);
    }
  }

  /**
   * Upload parsed splat data to GPU buffers
   */
  uploadToGPU(data: SplatData): {
    positions: GPUBuffer;
    scales: GPUBuffer;
    rotations: GPUBuffer;
    opacities: GPUBuffer;
    colors: GPUBuffer;
    shCoefficients?: GPUBuffer;
  } {
    const positions = this.bufferManager.createStorageBuffer(
      data.positions.byteLength,
      'splat-positions'
    );
    this.gpu.writeBuffer(positions, 0, data.positions);

    const scales = this.bufferManager.createStorageBuffer(
      data.scales.byteLength,
      'splat-scales'
    );
    this.gpu.writeBuffer(scales, 0, data.scales);

    const rotations = this.bufferManager.createStorageBuffer(
      data.rotations.byteLength,
      'splat-rotations'
    );
    this.gpu.writeBuffer(rotations, 0, data.rotations);

    const opacities = this.bufferManager.createStorageBuffer(
      data.opacities.byteLength,
      'splat-opacities'
    );
    this.gpu.writeBuffer(opacities, 0, data.opacities);

    const colors = this.bufferManager.createStorageBuffer(
      data.colors.byteLength,
      'splat-colors'
    );
    this.gpu.writeBuffer(colors, 0, data.colors);

    let shCoefficients: GPUBuffer | undefined;
    if (data.shCoefficients) {
      shCoefficients = this.bufferManager.createStorageBuffer(
        data.shCoefficients.byteLength,
        'splat-sh-coefficients'
      );
      this.gpu.writeBuffer(shCoefficients, 0, data.shCoefficients);
    }

    return {
      positions,
      scales,
      rotations,
      opacities,
      colors,
      shCoefficients,
    };
  }
}
