/**
 * Pipeline Builder
 * Fluent API for creating WebGPU compute and render pipelines
 */

import { GPUContext } from './gpu-context';

export interface ComputePipelineConfig {
  shader: string;
  entryPoint: string;
  bindings: BindingConfig[];
  label?: string;
  constants?: Record<string, number>;
}

export interface BindingConfig {
  binding: number;
  visibility: GPUShaderStageFlags;
  type: 'buffer' | 'sampler' | 'texture' | 'storageTexture';
  bufferType?: GPUBufferBindingType;
  samplerType?: GPUSamplerBindingType;
  textureType?: GPUTextureSampleType;
  textureDimension?: GPUTextureViewDimension;
  storageTextureAccess?: GPUStorageTextureAccess;
  storageTextureFormat?: GPUTextureFormat;
}

export class ComputePipelineBuilder {
  private gpu: GPUContext;
  private shader: string = '';
  private entryPoint: string = 'main';
  private bindings: BindingConfig[] = [];
  private label?: string;
  private constants: Record<string, number> = {};

  constructor(gpu: GPUContext) {
    this.gpu = gpu;
  }

  setShader(code: string, entryPoint = 'main'): this {
    this.shader = code;
    this.entryPoint = entryPoint;
    return this;
  }

  setLabel(label: string): this {
    this.label = label;
    return this;
  }

  addStorageBuffer(binding: number, readOnly = false): this {
    this.bindings.push({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      type: 'buffer',
      bufferType: readOnly ? 'read-only-storage' : 'storage',
    });
    return this;
  }

  addUniformBuffer(binding: number): this {
    this.bindings.push({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      type: 'buffer',
      bufferType: 'uniform',
    });
    return this;
  }

  addSampler(binding: number, type: GPUSamplerBindingType = 'filtering'): this {
    this.bindings.push({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      type: 'sampler',
      samplerType: type,
    });
    return this;
  }

  addTexture(
    binding: number,
    sampleType: GPUTextureSampleType = 'float',
    dimension: GPUTextureViewDimension = '2d'
  ): this {
    this.bindings.push({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      type: 'texture',
      textureType: sampleType,
      textureDimension: dimension,
    });
    return this;
  }

  addStorageTexture(
    binding: number,
    format: GPUTextureFormat,
    access: GPUStorageTextureAccess = 'write-only'
  ): this {
    this.bindings.push({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      type: 'storageTexture',
      storageTextureAccess: access,
      storageTextureFormat: format,
    });
    return this;
  }

  setConstant(name: string, value: number): this {
    this.constants[name] = value;
    return this;
  }

  build(): { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout } {
    const shaderModule = this.gpu.createShaderModule(this.shader, this.label);

    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = this.bindings.map((b) => {
      const entry: GPUBindGroupLayoutEntry = {
        binding: b.binding,
        visibility: b.visibility,
      };

      switch (b.type) {
        case 'buffer':
          entry.buffer = { type: b.bufferType };
          break;
        case 'sampler':
          entry.sampler = { type: b.samplerType };
          break;
        case 'texture':
          entry.texture = {
            sampleType: b.textureType,
            viewDimension: b.textureDimension,
          };
          break;
        case 'storageTexture':
          entry.storageTexture = {
            access: b.storageTextureAccess,
            format: b.storageTextureFormat!,
          };
          break;
      }

      return entry;
    });

    const bindGroupLayout = this.gpu.createBindGroupLayout({
      entries: bindGroupLayoutEntries,
      label: this.label ? `${this.label}-bind-group-layout` : undefined,
    });

    const pipelineLayout = this.gpu.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
      label: this.label ? `${this.label}-pipeline-layout` : undefined,
    });

    const pipeline = this.gpu.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: this.entryPoint,
        constants: Object.keys(this.constants).length > 0 ? this.constants : undefined,
      },
      label: this.label,
    });

    return { pipeline, bindGroupLayout };
  }

  async buildAsync(): Promise<{ pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout }> {
    const shaderModule = this.gpu.createShaderModule(this.shader, this.label);

    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = this.bindings.map((b) => {
      const entry: GPUBindGroupLayoutEntry = {
        binding: b.binding,
        visibility: b.visibility,
      };

      switch (b.type) {
        case 'buffer':
          entry.buffer = { type: b.bufferType };
          break;
        case 'sampler':
          entry.sampler = { type: b.samplerType };
          break;
        case 'texture':
          entry.texture = {
            sampleType: b.textureType,
            viewDimension: b.textureDimension,
          };
          break;
        case 'storageTexture':
          entry.storageTexture = {
            access: b.storageTextureAccess,
            format: b.storageTextureFormat!,
          };
          break;
      }

      return entry;
    });

    const bindGroupLayout = this.gpu.createBindGroupLayout({
      entries: bindGroupLayoutEntries,
      label: this.label ? `${this.label}-bind-group-layout` : undefined,
    });

    const pipelineLayout = this.gpu.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
      label: this.label ? `${this.label}-pipeline-layout` : undefined,
    });

    const pipeline = await this.gpu.createComputePipelineAsync({
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: this.entryPoint,
        constants: Object.keys(this.constants).length > 0 ? this.constants : undefined,
      },
      label: this.label,
    });

    return { pipeline, bindGroupLayout };
  }
}

export interface RenderPipelineConfig {
  vertexShader: string;
  fragmentShader: string;
  vertexBufferLayouts: GPUVertexBufferLayout[];
  colorTargetFormat: GPUTextureFormat;
  depthFormat?: GPUTextureFormat;
  primitive?: GPUPrimitiveState;
  label?: string;
}

export class RenderPipelineBuilder {
  private gpu: GPUContext;
  private vertexShader: string = '';
  private fragmentShader: string = '';
  private vertexEntryPoint: string = 'vs_main';
  private fragmentEntryPoint: string = 'fs_main';
  private vertexBufferLayouts: GPUVertexBufferLayout[] = [];
  private bindings: BindingConfig[] = [];
  private colorTargetFormat: GPUTextureFormat;
  private depthFormat?: GPUTextureFormat;
  private primitive: GPUPrimitiveState = { topology: 'triangle-list' };
  private blendState?: GPUBlendState;
  private label?: string;
  private multisampled = false;

  constructor(gpu: GPUContext) {
    this.gpu = gpu;
    this.colorTargetFormat = gpu.format;
  }

  setVertexShader(code: string, entryPoint = 'vs_main'): this {
    this.vertexShader = code;
    this.vertexEntryPoint = entryPoint;
    return this;
  }

  setFragmentShader(code: string, entryPoint = 'fs_main'): this {
    this.fragmentShader = code;
    this.fragmentEntryPoint = entryPoint;
    return this;
  }

  setShaders(vertex: string, fragment: string): this {
    this.vertexShader = vertex;
    this.fragmentShader = fragment;
    return this;
  }

  setLabel(label: string): this {
    this.label = label;
    return this;
  }

  addVertexBuffer(layout: GPUVertexBufferLayout): this {
    this.vertexBufferLayouts.push(layout);
    return this;
  }

  addStorageBuffer(binding: number, visibility: GPUShaderStageFlags, readOnly = false): this {
    this.bindings.push({
      binding,
      visibility,
      type: 'buffer',
      bufferType: readOnly ? 'read-only-storage' : 'storage',
    });
    return this;
  }

  addUniformBuffer(binding: number, visibility: GPUShaderStageFlags): this {
    this.bindings.push({
      binding,
      visibility,
      type: 'buffer',
      bufferType: 'uniform',
    });
    return this;
  }

  addSampler(
    binding: number,
    visibility: GPUShaderStageFlags,
    type: GPUSamplerBindingType = 'filtering'
  ): this {
    this.bindings.push({
      binding,
      visibility,
      type: 'sampler',
      samplerType: type,
    });
    return this;
  }

  addTexture(
    binding: number,
    visibility: GPUShaderStageFlags,
    sampleType: GPUTextureSampleType = 'float',
    dimension: GPUTextureViewDimension = '2d'
  ): this {
    this.bindings.push({
      binding,
      visibility,
      type: 'texture',
      textureType: sampleType,
      textureDimension: dimension,
    });
    return this;
  }

  setColorTarget(format: GPUTextureFormat): this {
    this.colorTargetFormat = format;
    return this;
  }

  setDepthFormat(format: GPUTextureFormat): this {
    this.depthFormat = format;
    return this;
  }

  setPrimitive(state: GPUPrimitiveState): this {
    this.primitive = state;
    return this;
  }

  setBlend(state: GPUBlendState): this {
    this.blendState = state;
    return this;
  }

  enableAlphaBlending(): this {
    this.blendState = {
      color: {
        srcFactor: 'src-alpha',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add',
      },
      alpha: {
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add',
      },
    };
    return this;
  }

  enableAdditiveBlending(): this {
    this.blendState = {
      color: {
        srcFactor: 'one',
        dstFactor: 'one',
        operation: 'add',
      },
      alpha: {
        srcFactor: 'one',
        dstFactor: 'one',
        operation: 'add',
      },
    };
    return this;
  }

  enableMultisampling(): this {
    this.multisampled = true;
    return this;
  }

  build(): { pipeline: GPURenderPipeline; bindGroupLayout: GPUBindGroupLayout } {
    const shaderModule = this.gpu.createShaderModule(
      this.vertexShader + '\n' + this.fragmentShader,
      this.label
    );

    const bindGroupLayoutEntries: GPUBindGroupLayoutEntry[] = this.bindings.map((b) => {
      const entry: GPUBindGroupLayoutEntry = {
        binding: b.binding,
        visibility: b.visibility,
      };

      switch (b.type) {
        case 'buffer':
          entry.buffer = { type: b.bufferType };
          break;
        case 'sampler':
          entry.sampler = { type: b.samplerType };
          break;
        case 'texture':
          entry.texture = {
            sampleType: b.textureType,
            viewDimension: b.textureDimension,
          };
          break;
        case 'storageTexture':
          entry.storageTexture = {
            access: b.storageTextureAccess,
            format: b.storageTextureFormat!,
          };
          break;
      }

      return entry;
    });

    const bindGroupLayout = this.gpu.createBindGroupLayout({
      entries: bindGroupLayoutEntries,
      label: this.label ? `${this.label}-bind-group-layout` : undefined,
    });

    const pipelineLayout = this.gpu.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
      label: this.label ? `${this.label}-pipeline-layout` : undefined,
    });

    const depthStencil: GPUDepthStencilState | undefined = this.depthFormat
      ? {
          format: this.depthFormat,
          depthWriteEnabled: true,
          depthCompare: 'less',
        }
      : undefined;

    const pipeline = this.gpu.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: this.vertexEntryPoint,
        buffers: this.vertexBufferLayouts,
      },
      fragment: {
        module: shaderModule,
        entryPoint: this.fragmentEntryPoint,
        targets: [
          {
            format: this.colorTargetFormat,
            blend: this.blendState,
          },
        ],
      },
      primitive: this.primitive,
      depthStencil,
      multisample: this.multisampled ? { count: 4 } : undefined,
      label: this.label,
    });

    return { pipeline, bindGroupLayout };
  }
}

// Factory function for creating pipeline builders
export function createComputePipeline(gpu: GPUContext): ComputePipelineBuilder {
  return new ComputePipelineBuilder(gpu);
}

export function createRenderPipeline(gpu: GPUContext): RenderPipelineBuilder {
  return new RenderPipelineBuilder(gpu);
}
