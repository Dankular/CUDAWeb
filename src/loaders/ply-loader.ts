/**
 * PLY File Loader for Gaussian Splatting
 * Parses PLY point cloud files with Gaussian attributes
 */

import { GPUContext, BufferManager } from '../core';

export interface PLYHeader {
  format: 'ascii' | 'binary_little_endian' | 'binary_big_endian';
  vertexCount: number;
  properties: PLYProperty[];
  headerLength: number;
}

export interface PLYProperty {
  name: string;
  type: string;
  isList: boolean;
  countType?: string;
}

export interface GaussianData {
  positions: Float32Array;
  scales: Float32Array;
  rotations: Float32Array;
  opacities: Float32Array;
  shCoefficients: Float32Array;
  count: number;
  shDegree: number;
}

export interface PLYLoadOptions {
  onProgress?: (loaded: number, total: number) => void;
  computeNormals?: boolean;
}

const TYPE_SIZES: Record<string, number> = {
  char: 1, uchar: 1,
  short: 2, ushort: 2,
  int: 4, uint: 4,
  float: 4, double: 8,
  int8: 1, uint8: 1,
  int16: 2, uint16: 2,
  int32: 4, uint32: 4,
  float32: 4, float64: 8,
};

export class PLYLoader {
  private gpu: GPUContext;
  private bufferManager: BufferManager;

  constructor(gpu: GPUContext, bufferManager: BufferManager) {
    this.gpu = gpu;
    this.bufferManager = bufferManager;
  }

  /**
   * Load PLY file from URL
   */
  async load(url: string, options: PLYLoadOptions = {}): Promise<GaussianData> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load PLY: ${response.status}`);
    }

    const buffer = await response.arrayBuffer();
    return this.parse(buffer, options);
  }

  /**
   * Load PLY file from File object
   */
  async loadFromFile(file: File, options: PLYLoadOptions = {}): Promise<GaussianData> {
    const buffer = await file.arrayBuffer();
    return this.parse(buffer, options);
  }

  /**
   * Parse PLY buffer
   */
  parse(buffer: ArrayBuffer, options: PLYLoadOptions = {}): GaussianData {
    const header = this.parseHeader(buffer);

    if (header.format === 'ascii') {
      return this.parseASCII(buffer, header, options);
    } else {
      return this.parseBinary(buffer, header, options);
    }
  }

  /**
   * Parse PLY header
   */
  private parseHeader(buffer: ArrayBuffer): PLYHeader {
    const decoder = new TextDecoder();
    const headerText = decoder.decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 64 * 1024)));

    const endHeaderIndex = headerText.indexOf('end_header');
    if (endHeaderIndex === -1) {
      throw new Error('Invalid PLY: no end_header found');
    }

    const headerLines = headerText.substring(0, endHeaderIndex).split('\n');
    const header: PLYHeader = {
      format: 'binary_little_endian',
      vertexCount: 0,
      properties: [],
      headerLength: 0,
    };

    let inVertexElement = false;

    for (const line of headerLines) {
      const parts = line.trim().split(/\s+/);

      if (parts[0] === 'format') {
        header.format = parts[1] as PLYHeader['format'];
      } else if (parts[0] === 'element') {
        if (parts[1] === 'vertex') {
          header.vertexCount = parseInt(parts[2], 10);
          inVertexElement = true;
        } else {
          inVertexElement = false;
        }
      } else if (parts[0] === 'property' && inVertexElement) {
        if (parts[1] === 'list') {
          header.properties.push({
            name: parts[4],
            type: parts[3],
            isList: true,
            countType: parts[2],
          });
        } else {
          header.properties.push({
            name: parts[2],
            type: parts[1],
            isList: false,
          });
        }
      }
    }

    // Calculate header length (including "end_header\n")
    const headerBytes = new TextEncoder().encode(headerText.substring(0, endHeaderIndex + 11));
    header.headerLength = headerBytes.length;

    return header;
  }

  /**
   * Parse binary PLY format
   */
  private parseBinary(
    buffer: ArrayBuffer,
    header: PLYHeader,
    options: PLYLoadOptions
  ): GaussianData {
    const data = new DataView(buffer, header.headerLength);
    const littleEndian = header.format === 'binary_little_endian';

    // Find property indices
    const propMap = this.buildPropertyMap(header.properties);

    // Calculate row size
    let rowSize = 0;
    for (const prop of header.properties) {
      rowSize += TYPE_SIZES[prop.type] || 4;
    }

    // Determine SH degree from properties
    const shDegree = this.detectSHDegree(header.properties);
    const shCoeffCount = (shDegree + 1) * (shDegree + 1);

    // Allocate output arrays
    const positions = new Float32Array(header.vertexCount * 3);
    const scales = new Float32Array(header.vertexCount * 3);
    const rotations = new Float32Array(header.vertexCount * 4);
    const opacities = new Float32Array(header.vertexCount);
    const shCoefficients = new Float32Array(header.vertexCount * shCoeffCount * 3);

    // Parse vertices
    for (let i = 0; i < header.vertexCount; i++) {
      const offset = i * rowSize;

      // Position
      positions[i * 3 + 0] = this.readProperty(data, offset, propMap.x, littleEndian);
      positions[i * 3 + 1] = this.readProperty(data, offset, propMap.y, littleEndian);
      positions[i * 3 + 2] = this.readProperty(data, offset, propMap.z, littleEndian);

      // Scale (log scale in PLY)
      scales[i * 3 + 0] = Math.exp(this.readProperty(data, offset, propMap.scale_0, littleEndian));
      scales[i * 3 + 1] = Math.exp(this.readProperty(data, offset, propMap.scale_1, littleEndian));
      scales[i * 3 + 2] = Math.exp(this.readProperty(data, offset, propMap.scale_2, littleEndian));

      // Rotation (quaternion)
      rotations[i * 4 + 0] = this.readProperty(data, offset, propMap.rot_0, littleEndian);
      rotations[i * 4 + 1] = this.readProperty(data, offset, propMap.rot_1, littleEndian);
      rotations[i * 4 + 2] = this.readProperty(data, offset, propMap.rot_2, littleEndian);
      rotations[i * 4 + 3] = this.readProperty(data, offset, propMap.rot_3, littleEndian);

      // Normalize rotation quaternion
      const qlen = Math.sqrt(
        rotations[i * 4 + 0] ** 2 +
        rotations[i * 4 + 1] ** 2 +
        rotations[i * 4 + 2] ** 2 +
        rotations[i * 4 + 3] ** 2
      );
      if (qlen > 0) {
        rotations[i * 4 + 0] /= qlen;
        rotations[i * 4 + 1] /= qlen;
        rotations[i * 4 + 2] /= qlen;
        rotations[i * 4 + 3] /= qlen;
      }

      // Opacity (sigmoid-inverse in PLY)
      opacities[i] = this.readProperty(data, offset, propMap.opacity, littleEndian);

      // SH coefficients
      // DC term (f_dc_0, f_dc_1, f_dc_2)
      shCoefficients[i * shCoeffCount * 3 + 0] = this.readProperty(data, offset, propMap.f_dc_0, littleEndian);
      shCoefficients[i * shCoeffCount * 3 + 1] = this.readProperty(data, offset, propMap.f_dc_1, littleEndian);
      shCoefficients[i * shCoeffCount * 3 + 2] = this.readProperty(data, offset, propMap.f_dc_2, littleEndian);

      // Higher order SH coefficients
      for (let j = 1; j < shCoeffCount; j++) {
        const restIdx = j - 1;
        const rProp = propMap[`f_rest_${restIdx * 3 + 0}`];
        const gProp = propMap[`f_rest_${restIdx * 3 + 1}`];
        const bProp = propMap[`f_rest_${restIdx * 3 + 2}`];

        if (rProp && gProp && bProp) {
          shCoefficients[i * shCoeffCount * 3 + j * 3 + 0] = this.readProperty(data, offset, rProp, littleEndian);
          shCoefficients[i * shCoeffCount * 3 + j * 3 + 1] = this.readProperty(data, offset, gProp, littleEndian);
          shCoefficients[i * shCoeffCount * 3 + j * 3 + 2] = this.readProperty(data, offset, bProp, littleEndian);
        }
      }

      // Progress callback
      if (options.onProgress && i % 10000 === 0) {
        options.onProgress(i, header.vertexCount);
      }
    }

    options.onProgress?.(header.vertexCount, header.vertexCount);

    return {
      positions,
      scales,
      rotations,
      opacities,
      shCoefficients,
      count: header.vertexCount,
      shDegree,
    };
  }

  /**
   * Parse ASCII PLY format
   */
  private parseASCII(
    buffer: ArrayBuffer,
    header: PLYHeader,
    options: PLYLoadOptions
  ): GaussianData {
    const decoder = new TextDecoder();
    const text = decoder.decode(new Uint8Array(buffer));

    const endHeaderIndex = text.indexOf('end_header');
    const dataText = text.substring(endHeaderIndex + 11).trim();
    const lines = dataText.split('\n');

    // Find property indices
    const propIndices = new Map<string, number>();
    header.properties.forEach((prop, idx) => {
      propIndices.set(prop.name, idx);
    });

    const shDegree = this.detectSHDegree(header.properties);
    const shCoeffCount = (shDegree + 1) * (shDegree + 1);

    const positions = new Float32Array(header.vertexCount * 3);
    const scales = new Float32Array(header.vertexCount * 3);
    const rotations = new Float32Array(header.vertexCount * 4);
    const opacities = new Float32Array(header.vertexCount);
    const shCoefficients = new Float32Array(header.vertexCount * shCoeffCount * 3);

    for (let i = 0; i < header.vertexCount && i < lines.length; i++) {
      const values = lines[i].trim().split(/\s+/).map(parseFloat);

      positions[i * 3 + 0] = values[propIndices.get('x')!];
      positions[i * 3 + 1] = values[propIndices.get('y')!];
      positions[i * 3 + 2] = values[propIndices.get('z')!];

      scales[i * 3 + 0] = Math.exp(values[propIndices.get('scale_0')!]);
      scales[i * 3 + 1] = Math.exp(values[propIndices.get('scale_1')!]);
      scales[i * 3 + 2] = Math.exp(values[propIndices.get('scale_2')!]);

      rotations[i * 4 + 0] = values[propIndices.get('rot_0')!];
      rotations[i * 4 + 1] = values[propIndices.get('rot_1')!];
      rotations[i * 4 + 2] = values[propIndices.get('rot_2')!];
      rotations[i * 4 + 3] = values[propIndices.get('rot_3')!];

      opacities[i] = values[propIndices.get('opacity')!];

      // SH coefficients
      shCoefficients[i * shCoeffCount * 3 + 0] = values[propIndices.get('f_dc_0')!];
      shCoefficients[i * shCoeffCount * 3 + 1] = values[propIndices.get('f_dc_1')!];
      shCoefficients[i * shCoeffCount * 3 + 2] = values[propIndices.get('f_dc_2')!];

      if (options.onProgress && i % 10000 === 0) {
        options.onProgress(i, header.vertexCount);
      }
    }

    return {
      positions,
      scales,
      rotations,
      opacities,
      shCoefficients,
      count: header.vertexCount,
      shDegree,
    };
  }

  /**
   * Build property offset map
   */
  private buildPropertyMap(
    properties: PLYProperty[]
  ): Record<string, { offset: number; type: string }> {
    const map: Record<string, { offset: number; type: string }> = {};
    let offset = 0;

    for (const prop of properties) {
      map[prop.name] = { offset, type: prop.type };
      offset += TYPE_SIZES[prop.type] || 4;
    }

    return map;
  }

  /**
   * Read a property value from binary data
   */
  private readProperty(
    data: DataView,
    rowOffset: number,
    prop: { offset: number; type: string } | undefined,
    littleEndian: boolean
  ): number {
    if (!prop) return 0;

    const offset = rowOffset + prop.offset;

    switch (prop.type) {
      case 'float':
      case 'float32':
        return data.getFloat32(offset, littleEndian);
      case 'double':
      case 'float64':
        return data.getFloat64(offset, littleEndian);
      case 'int':
      case 'int32':
        return data.getInt32(offset, littleEndian);
      case 'uint':
      case 'uint32':
        return data.getUint32(offset, littleEndian);
      case 'short':
      case 'int16':
        return data.getInt16(offset, littleEndian);
      case 'ushort':
      case 'uint16':
        return data.getUint16(offset, littleEndian);
      case 'char':
      case 'int8':
        return data.getInt8(offset);
      case 'uchar':
      case 'uint8':
        return data.getUint8(offset);
      default:
        return data.getFloat32(offset, littleEndian);
    }
  }

  /**
   * Detect SH degree from properties
   */
  private detectSHDegree(properties: PLYProperty[]): number {
    const restProps = properties.filter((p) => p.name.startsWith('f_rest_'));

    if (restProps.length === 0) {
      return 0; // DC only
    }

    // f_rest has (degree+1)^2 - 1 coefficients per channel, times 3 channels
    const numRest = restProps.length / 3;
    const totalCoeffs = numRest + 1; // +1 for DC term

    // totalCoeffs = (degree + 1)^2
    const degree = Math.sqrt(totalCoeffs) - 1;
    return Math.floor(degree);
  }

  /**
   * Upload parsed Gaussian data to GPU buffers
   */
  uploadToGPU(data: GaussianData): {
    positions: GPUBuffer;
    scales: GPUBuffer;
    rotations: GPUBuffer;
    opacities: GPUBuffer;
    shCoefficients: GPUBuffer;
  } {
    const positions = this.bufferManager.createStorageBuffer(
      data.positions.byteLength,
      'gaussian-positions'
    );
    this.gpu.writeBuffer(positions, 0, data.positions);

    const scales = this.bufferManager.createStorageBuffer(
      data.scales.byteLength,
      'gaussian-scales'
    );
    this.gpu.writeBuffer(scales, 0, data.scales);

    const rotations = this.bufferManager.createStorageBuffer(
      data.rotations.byteLength,
      'gaussian-rotations'
    );
    this.gpu.writeBuffer(rotations, 0, data.rotations);

    const opacities = this.bufferManager.createStorageBuffer(
      data.opacities.byteLength,
      'gaussian-opacities'
    );
    this.gpu.writeBuffer(opacities, 0, data.opacities);

    const shCoefficients = this.bufferManager.createStorageBuffer(
      data.shCoefficients.byteLength,
      'gaussian-sh-coefficients'
    );
    this.gpu.writeBuffer(shCoefficients, 0, data.shCoefficients);

    return {
      positions,
      scales,
      rotations,
      opacities,
      shCoefficients,
    };
  }
}
