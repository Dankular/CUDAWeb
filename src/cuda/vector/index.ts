/**
 * CUDA Vector Types for WebGPU
 *
 * Provides CUDA-compatible vector types and operations
 */

import cudaVectorWGSL from './cuda-vector.wgsl?raw';

// Export the WGSL source code for inclusion in shaders
export const CUDA_VECTOR_WGSL = cudaVectorWGSL;

// ============================================================================
// Float Vector Types (CPU-side implementations)
// ============================================================================

export class Float2 {
  x: number;
  y: number;

  constructor(x: number = 0, y: number = 0) {
    this.x = x;
    this.y = y;
  }

  static make(x: number, y: number): Float2 {
    return new Float2(x, y);
  }

  add(other: Float2): Float2 {
    return new Float2(this.x + other.x, this.y + other.y);
  }

  sub(other: Float2): Float2 {
    return new Float2(this.x - other.x, this.y - other.y);
  }

  mul(other: Float2): Float2 {
    return new Float2(this.x * other.x, this.y * other.y);
  }

  div(other: Float2): Float2 {
    return new Float2(this.x / other.x, this.y / other.y);
  }

  scale(s: number): Float2 {
    return new Float2(this.x * s, this.y * s);
  }

  dot(other: Float2): number {
    return this.x * other.x + this.y * other.y;
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }

  normalize(): Float2 {
    const len = this.length();
    return new Float2(this.x / len, this.y / len);
  }

  lerp(other: Float2, t: number): Float2 {
    return new Float2(
      this.x + t * (other.x - this.x),
      this.y + t * (other.y - this.y)
    );
  }

  toArray(): [number, number] {
    return [this.x, this.y];
  }

  toFloat32Array(): Float32Array {
    return new Float32Array([this.x, this.y]);
  }
}

export class Float3 {
  x: number;
  y: number;
  z: number;

  constructor(x: number = 0, y: number = 0, z: number = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  static make(x: number, y: number, z: number): Float3 {
    return new Float3(x, y, z);
  }

  add(other: Float3): Float3 {
    return new Float3(this.x + other.x, this.y + other.y, this.z + other.z);
  }

  sub(other: Float3): Float3 {
    return new Float3(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  mul(other: Float3): Float3 {
    return new Float3(this.x * other.x, this.y * other.y, this.z * other.z);
  }

  div(other: Float3): Float3 {
    return new Float3(this.x / other.x, this.y / other.y, this.z / other.z);
  }

  scale(s: number): Float3 {
    return new Float3(this.x * s, this.y * s, this.z * s);
  }

  dot(other: Float3): number {
    return this.x * other.x + this.y * other.y + this.z * other.z;
  }

  cross(other: Float3): Float3 {
    return new Float3(
      this.y * other.z - this.z * other.y,
      this.z * other.x - this.x * other.z,
      this.x * other.y - this.y * other.x
    );
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  normalize(): Float3 {
    const len = this.length();
    return new Float3(this.x / len, this.y / len, this.z / len);
  }

  lerp(other: Float3, t: number): Float3 {
    return new Float3(
      this.x + t * (other.x - this.x),
      this.y + t * (other.y - this.y),
      this.z + t * (other.z - this.z)
    );
  }

  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  toFloat32Array(): Float32Array {
    return new Float32Array([this.x, this.y, this.z]);
  }
}

export class Float4 {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }

  static make(x: number, y: number, z: number, w: number): Float4 {
    return new Float4(x, y, z, w);
  }

  add(other: Float4): Float4 {
    return new Float4(
      this.x + other.x,
      this.y + other.y,
      this.z + other.z,
      this.w + other.w
    );
  }

  sub(other: Float4): Float4 {
    return new Float4(
      this.x - other.x,
      this.y - other.y,
      this.z - other.z,
      this.w - other.w
    );
  }

  mul(other: Float4): Float4 {
    return new Float4(
      this.x * other.x,
      this.y * other.y,
      this.z * other.z,
      this.w * other.w
    );
  }

  div(other: Float4): Float4 {
    return new Float4(
      this.x / other.x,
      this.y / other.y,
      this.z / other.z,
      this.w / other.w
    );
  }

  scale(s: number): Float4 {
    return new Float4(this.x * s, this.y * s, this.z * s, this.w * s);
  }

  dot(other: Float4): number {
    return (
      this.x * other.x +
      this.y * other.y +
      this.z * other.z +
      this.w * other.w
    );
  }

  length(): number {
    return Math.sqrt(
      this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w
    );
  }

  normalize(): Float4 {
    const len = this.length();
    return new Float4(this.x / len, this.y / len, this.z / len, this.w / len);
  }

  lerp(other: Float4, t: number): Float4 {
    return new Float4(
      this.x + t * (other.x - this.x),
      this.y + t * (other.y - this.y),
      this.z + t * (other.z - this.z),
      this.w + t * (other.w - this.w)
    );
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }

  toFloat32Array(): Float32Array {
    return new Float32Array([this.x, this.y, this.z, this.w]);
  }

  // Quaternion operations
  get xyz(): Float3 {
    return new Float3(this.x, this.y, this.z);
  }
}

// ============================================================================
// Double Precision Vector Types (CPU-side, using JavaScript's native float64)
// ============================================================================

export class Double2 {
  x: number;
  y: number;

  constructor(x: number = 0, y: number = 0) {
    this.x = x;
    this.y = y;
  }

  static make(x: number, y: number): Double2 {
    return new Double2(x, y);
  }

  add(other: Double2): Double2 {
    return new Double2(this.x + other.x, this.y + other.y);
  }

  sub(other: Double2): Double2 {
    return new Double2(this.x - other.x, this.y - other.y);
  }

  mul(other: Double2): Double2 {
    return new Double2(this.x * other.x, this.y * other.y);
  }

  div(other: Double2): Double2 {
    return new Double2(this.x / other.x, this.y / other.y);
  }

  scale(s: number): Double2 {
    return new Double2(this.x * s, this.y * s);
  }

  dot(other: Double2): number {
    return this.x * other.x + this.y * other.y;
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }

  normalize(): Double2 {
    const len = this.length();
    return new Double2(this.x / len, this.y / len);
  }

  lerp(other: Double2, t: number): Double2 {
    return new Double2(
      this.x + t * (other.x - this.x),
      this.y + t * (other.y - this.y)
    );
  }

  toArray(): [number, number] {
    return [this.x, this.y];
  }

  toFloat64Array(): Float64Array {
    return new Float64Array([this.x, this.y]);
  }

  toFloat32Array(): Float32Array {
    return new Float32Array([this.x, this.y]);
  }

  // Convert to emulated double format (two u32 per component)
  toEmulatedArray(): Uint32Array {
    const f64 = new Float64Array([this.x, this.y]);
    const u32 = new Uint32Array(f64.buffer);
    return u32;
  }
}

export class Double3 {
  x: number;
  y: number;
  z: number;

  constructor(x: number = 0, y: number = 0, z: number = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  static make(x: number, y: number, z: number): Double3 {
    return new Double3(x, y, z);
  }

  add(other: Double3): Double3 {
    return new Double3(this.x + other.x, this.y + other.y, this.z + other.z);
  }

  sub(other: Double3): Double3 {
    return new Double3(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  mul(other: Double3): Double3 {
    return new Double3(this.x * other.x, this.y * other.y, this.z * other.z);
  }

  div(other: Double3): Double3 {
    return new Double3(this.x / other.x, this.y / other.y, this.z / other.z);
  }

  scale(s: number): Double3 {
    return new Double3(this.x * s, this.y * s, this.z * s);
  }

  dot(other: Double3): number {
    return this.x * other.x + this.y * other.y + this.z * other.z;
  }

  cross(other: Double3): Double3 {
    return new Double3(
      this.y * other.z - this.z * other.y,
      this.z * other.x - this.x * other.z,
      this.x * other.y - this.y * other.x
    );
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  normalize(): Double3 {
    const len = this.length();
    return new Double3(this.x / len, this.y / len, this.z / len);
  }

  lerp(other: Double3, t: number): Double3 {
    return new Double3(
      this.x + t * (other.x - this.x),
      this.y + t * (other.y - this.y),
      this.z + t * (other.z - this.z)
    );
  }

  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  toFloat64Array(): Float64Array {
    return new Float64Array([this.x, this.y, this.z]);
  }

  toFloat32Array(): Float32Array {
    return new Float32Array([this.x, this.y, this.z]);
  }

  // Convert to emulated double format (two u32 per component)
  toEmulatedArray(): Uint32Array {
    const f64 = new Float64Array([this.x, this.y, this.z]);
    const u32 = new Uint32Array(f64.buffer);
    return u32;
  }
}

export class Double4 {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 0) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }

  static make(x: number, y: number, z: number, w: number): Double4 {
    return new Double4(x, y, z, w);
  }

  add(other: Double4): Double4 {
    return new Double4(
      this.x + other.x,
      this.y + other.y,
      this.z + other.z,
      this.w + other.w
    );
  }

  sub(other: Double4): Double4 {
    return new Double4(
      this.x - other.x,
      this.y - other.y,
      this.z - other.z,
      this.w - other.w
    );
  }

  mul(other: Double4): Double4 {
    return new Double4(
      this.x * other.x,
      this.y * other.y,
      this.z * other.z,
      this.w * other.w
    );
  }

  div(other: Double4): Double4 {
    return new Double4(
      this.x / other.x,
      this.y / other.y,
      this.z / other.z,
      this.w / other.w
    );
  }

  scale(s: number): Double4 {
    return new Double4(this.x * s, this.y * s, this.z * s, this.w * s);
  }

  dot(other: Double4): number {
    return (
      this.x * other.x +
      this.y * other.y +
      this.z * other.z +
      this.w * other.w
    );
  }

  length(): number {
    return Math.sqrt(
      this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w
    );
  }

  normalize(): Double4 {
    const len = this.length();
    return new Double4(this.x / len, this.y / len, this.z / len, this.w / len);
  }

  lerp(other: Double4, t: number): Double4 {
    return new Double4(
      this.x + t * (other.x - this.x),
      this.y + t * (other.y - this.y),
      this.z + t * (other.z - this.z),
      this.w + t * (other.w - this.w)
    );
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }

  toFloat64Array(): Float64Array {
    return new Float64Array([this.x, this.y, this.z, this.w]);
  }

  toFloat32Array(): Float32Array {
    return new Float32Array([this.x, this.y, this.z, this.w]);
  }

  // Convert to emulated double format (two u32 per component)
  toEmulatedArray(): Uint32Array {
    const f64 = new Float64Array([this.x, this.y, this.z, this.w]);
    const u32 = new Uint32Array(f64.buffer);
    return u32;
  }

  get xyz(): Double3 {
    return new Double3(this.x, this.y, this.z);
  }
}

// ============================================================================
// Integer Vector Types
// ============================================================================

export class Int2 {
  x: number;
  y: number;

  constructor(x: number = 0, y: number = 0) {
    this.x = Math.floor(x);
    this.y = Math.floor(y);
  }

  static make(x: number, y: number): Int2 {
    return new Int2(x, y);
  }

  add(other: Int2): Int2 {
    return new Int2(this.x + other.x, this.y + other.y);
  }

  sub(other: Int2): Int2 {
    return new Int2(this.x - other.x, this.y - other.y);
  }

  mul(other: Int2): Int2 {
    return new Int2(this.x * other.x, this.y * other.y);
  }

  div(other: Int2): Int2 {
    return new Int2(
      Math.floor(this.x / other.x),
      Math.floor(this.y / other.y)
    );
  }

  toArray(): [number, number] {
    return [this.x, this.y];
  }

  toInt32Array(): Int32Array {
    return new Int32Array([this.x, this.y]);
  }
}

export class Int3 {
  x: number;
  y: number;
  z: number;

  constructor(x: number = 0, y: number = 0, z: number = 0) {
    this.x = Math.floor(x);
    this.y = Math.floor(y);
    this.z = Math.floor(z);
  }

  static make(x: number, y: number, z: number): Int3 {
    return new Int3(x, y, z);
  }

  add(other: Int3): Int3 {
    return new Int3(this.x + other.x, this.y + other.y, this.z + other.z);
  }

  sub(other: Int3): Int3 {
    return new Int3(this.x - other.x, this.y - other.y, this.z - other.z);
  }

  mul(other: Int3): Int3 {
    return new Int3(this.x * other.x, this.y * other.y, this.z * other.z);
  }

  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  toInt32Array(): Int32Array {
    return new Int32Array([this.x, this.y, this.z]);
  }
}

export class Int4 {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 0) {
    this.x = Math.floor(x);
    this.y = Math.floor(y);
    this.z = Math.floor(z);
    this.w = Math.floor(w);
  }

  static make(x: number, y: number, z: number, w: number): Int4 {
    return new Int4(x, y, z, w);
  }

  add(other: Int4): Int4 {
    return new Int4(
      this.x + other.x,
      this.y + other.y,
      this.z + other.z,
      this.w + other.w
    );
  }

  sub(other: Int4): Int4 {
    return new Int4(
      this.x - other.x,
      this.y - other.y,
      this.z - other.z,
      this.w - other.w
    );
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }

  toInt32Array(): Int32Array {
    return new Int32Array([this.x, this.y, this.z, this.w]);
  }
}

// ============================================================================
// Unsigned Integer Vector Types
// ============================================================================

export class Uint2 {
  x: number;
  y: number;

  constructor(x: number = 0, y: number = 0) {
    this.x = Math.max(0, Math.floor(x)) >>> 0;
    this.y = Math.max(0, Math.floor(y)) >>> 0;
  }

  static make(x: number, y: number): Uint2 {
    return new Uint2(x, y);
  }

  toArray(): [number, number] {
    return [this.x, this.y];
  }

  toUint32Array(): Uint32Array {
    return new Uint32Array([this.x, this.y]);
  }
}

export class Uint3 {
  x: number;
  y: number;
  z: number;

  constructor(x: number = 0, y: number = 0, z: number = 0) {
    this.x = Math.max(0, Math.floor(x)) >>> 0;
    this.y = Math.max(0, Math.floor(y)) >>> 0;
    this.z = Math.max(0, Math.floor(z)) >>> 0;
  }

  static make(x: number, y: number, z: number): Uint3 {
    return new Uint3(x, y, z);
  }

  toArray(): [number, number, number] {
    return [this.x, this.y, this.z];
  }

  toUint32Array(): Uint32Array {
    return new Uint32Array([this.x, this.y, this.z]);
  }
}

export class Uint4 {
  x: number;
  y: number;
  z: number;
  w: number;

  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 0) {
    this.x = Math.max(0, Math.floor(x)) >>> 0;
    this.y = Math.max(0, Math.floor(y)) >>> 0;
    this.z = Math.max(0, Math.floor(z)) >>> 0;
    this.w = Math.max(0, Math.floor(w)) >>> 0;
  }

  static make(x: number, y: number, z: number, w: number): Uint4 {
    return new Uint4(x, y, z, w);
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w];
  }

  toUint32Array(): Uint32Array {
    return new Uint32Array([this.x, this.y, this.z, this.w]);
  }
}

// ============================================================================
// Matrix Types
// ============================================================================

export class Mat3 {
  data: Float32Array;

  constructor(data?: Float32Array | number[]) {
    if (data) {
      this.data = data instanceof Float32Array ? data : new Float32Array(data);
    } else {
      this.data = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]); // Identity
    }
  }

  static identity(): Mat3 {
    return new Mat3();
  }

  mul(other: Mat3): Mat3 {
    const a = this.data;
    const b = other.data;
    const result = new Float32Array(9);

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        result[i * 3 + j] =
          a[i * 3 + 0] * b[0 * 3 + j] +
          a[i * 3 + 1] * b[1 * 3 + j] +
          a[i * 3 + 2] * b[2 * 3 + j];
      }
    }

    return new Mat3(result);
  }

  mulVec3(v: Float3): Float3 {
    const m = this.data;
    return new Float3(
      m[0] * v.x + m[1] * v.y + m[2] * v.z,
      m[3] * v.x + m[4] * v.y + m[5] * v.z,
      m[6] * v.x + m[7] * v.y + m[8] * v.z
    );
  }

  transpose(): Mat3 {
    const m = this.data;
    return new Mat3([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
  }

  determinant(): number {
    const m = this.data;
    return (
      m[0] * (m[4] * m[8] - m[5] * m[7]) -
      m[1] * (m[3] * m[8] - m[5] * m[6]) +
      m[2] * (m[3] * m[7] - m[4] * m[6])
    );
  }
}

export class Mat4 {
  data: Float32Array;

  constructor(data?: Float32Array | number[]) {
    if (data) {
      this.data = data instanceof Float32Array ? data : new Float32Array(data);
    } else {
      this.data = new Float32Array([
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
      ]); // Identity
    }
  }

  static identity(): Mat4 {
    return new Mat4();
  }

  static perspective(
    fov: number,
    aspect: number,
    near: number,
    far: number
  ): Mat4 {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Mat4([
      f / aspect,
      0,
      0,
      0,
      0,
      f,
      0,
      0,
      0,
      0,
      (far + near) * nf,
      -1,
      0,
      0,
      2 * far * near * nf,
      0,
    ]);
  }

  static lookAt(eye: Float3, center: Float3, up: Float3): Mat4 {
    const f = center.sub(eye).normalize();
    const s = f.cross(up).normalize();
    const u = s.cross(f);

    return new Mat4([
      s.x,
      u.x,
      -f.x,
      0,
      s.y,
      u.y,
      -f.y,
      0,
      s.z,
      u.z,
      -f.z,
      0,
      -s.dot(eye),
      -u.dot(eye),
      f.dot(eye),
      1,
    ]);
  }

  static translation(x: number, y: number, z: number): Mat4 {
    return new Mat4([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
  }

  static scale(x: number, y: number, z: number): Mat4 {
    return new Mat4([x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1]);
  }

  static rotationX(angle: number): Mat4 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Mat4([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
  }

  static rotationY(angle: number): Mat4 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Mat4([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
  }

  static rotationZ(angle: number): Mat4 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Mat4([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }

  mul(other: Mat4): Mat4 {
    const a = this.data;
    const b = other.data;
    const result = new Float32Array(16);

    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        result[i * 4 + j] =
          a[i * 4 + 0] * b[0 * 4 + j] +
          a[i * 4 + 1] * b[1 * 4 + j] +
          a[i * 4 + 2] * b[2 * 4 + j] +
          a[i * 4 + 3] * b[3 * 4 + j];
      }
    }

    return new Mat4(result);
  }

  mulVec4(v: Float4): Float4 {
    const m = this.data;
    return new Float4(
      m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3] * v.w,
      m[4] * v.x + m[5] * v.y + m[6] * v.z + m[7] * v.w,
      m[8] * v.x + m[9] * v.y + m[10] * v.z + m[11] * v.w,
      m[12] * v.x + m[13] * v.y + m[14] * v.z + m[15] * v.w
    );
  }

  transpose(): Mat4 {
    const m = this.data;
    return new Mat4([
      m[0],
      m[4],
      m[8],
      m[12],
      m[1],
      m[5],
      m[9],
      m[13],
      m[2],
      m[6],
      m[10],
      m[14],
      m[3],
      m[7],
      m[11],
      m[15],
    ]);
  }
}

// ============================================================================
// Quaternion Type
// ============================================================================

export class Quat extends Float4 {
  constructor(x: number = 0, y: number = 0, z: number = 0, w: number = 1) {
    super(x, y, z, w);
  }

  static identity(): Quat {
    return new Quat(0, 0, 0, 1);
  }

  static fromAxisAngle(axis: Float3, angle: number): Quat {
    const halfAngle = angle * 0.5;
    const s = Math.sin(halfAngle);
    const normalizedAxis = axis.normalize();
    return new Quat(
      normalizedAxis.x * s,
      normalizedAxis.y * s,
      normalizedAxis.z * s,
      Math.cos(halfAngle)
    );
  }

  static fromEuler(pitch: number, yaw: number, roll: number): Quat {
    const cy = Math.cos(yaw * 0.5);
    const sy = Math.sin(yaw * 0.5);
    const cp = Math.cos(pitch * 0.5);
    const sp = Math.sin(pitch * 0.5);
    const cr = Math.cos(roll * 0.5);
    const sr = Math.sin(roll * 0.5);

    return new Quat(
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
      cr * cp * cy + sr * sp * sy
    );
  }

  mulQuat(other: Quat): Quat {
    return new Quat(
      this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
      this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
      this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w,
      this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z
    );
  }

  conjugate(): Quat {
    return new Quat(-this.x, -this.y, -this.z, this.w);
  }

  inverse(): Quat {
    const lenSq = this.dot(this);
    const conj = this.conjugate();
    return new Quat(
      conj.x / lenSq,
      conj.y / lenSq,
      conj.z / lenSq,
      conj.w / lenSq
    );
  }

  rotateVec3(v: Float3): Float3 {
    const qv = new Float4(v.x, v.y, v.z, 0);
    const qConj = this.conjugate();
    const result = this.mulQuat(new Quat(qv.x, qv.y, qv.z, qv.w)).mulQuat(qConj);
    return new Float3(result.x, result.y, result.z);
  }

  toMat3(): Mat3 {
    const xx = this.x * this.x;
    const xy = this.x * this.y;
    const xz = this.x * this.z;
    const xw = this.x * this.w;
    const yy = this.y * this.y;
    const yz = this.y * this.z;
    const yw = this.y * this.w;
    const zz = this.z * this.z;
    const zw = this.z * this.w;

    return new Mat3([
      1 - 2 * (yy + zz),
      2 * (xy + zw),
      2 * (xz - yw),
      2 * (xy - zw),
      1 - 2 * (xx + zz),
      2 * (yz + xw),
      2 * (xz + yw),
      2 * (yz - xw),
      1 - 2 * (xx + yy),
    ]);
  }

  toMat4(): Mat4 {
    const m3 = this.toMat3();
    return new Mat4([
      m3.data[0],
      m3.data[1],
      m3.data[2],
      0,
      m3.data[3],
      m3.data[4],
      m3.data[5],
      0,
      m3.data[6],
      m3.data[7],
      m3.data[8],
      0,
      0,
      0,
      0,
      1,
    ]);
  }

  slerp(other: Quat, t: number): Quat {
    let otherQ = other;
    let cosHalfTheta = this.dot(other);

    if (cosHalfTheta < 0) {
      otherQ = new Quat(-other.x, -other.y, -other.z, -other.w);
      cosHalfTheta = -cosHalfTheta;
    }

    if (cosHalfTheta > 0.9999) {
      return new Quat(
        this.x + t * (otherQ.x - this.x),
        this.y + t * (otherQ.y - this.y),
        this.z + t * (otherQ.z - this.z),
        this.w + t * (otherQ.w - this.w)
      ).normalize() as Quat;
    }

    const halfTheta = Math.acos(cosHalfTheta);
    const sinHalfTheta = Math.sqrt(1 - cosHalfTheta * cosHalfTheta);

    const ratioA = Math.sin((1 - t) * halfTheta) / sinHalfTheta;
    const ratioB = Math.sin(t * halfTheta) / sinHalfTheta;

    return new Quat(
      this.x * ratioA + otherQ.x * ratioB,
      this.y * ratioA + otherQ.y * ratioB,
      this.z * ratioA + otherQ.z * ratioB,
      this.w * ratioA + otherQ.w * ratioB
    );
  }
}

// ============================================================================
// Helper functions for shader inclusion
// ============================================================================

export function withCudaVector(shaderCode: string): string {
  return CUDA_VECTOR_WGSL + '\n\n' + shaderCode;
}

// Factory functions
export const make_float2 = Float2.make;
export const make_float3 = Float3.make;
export const make_float4 = Float4.make;
export const make_double2 = Double2.make;
export const make_double3 = Double3.make;
export const make_double4 = Double4.make;
export const make_int2 = Int2.make;
export const make_int3 = Int3.make;
export const make_int4 = Int4.make;
export const make_uint2 = Uint2.make;
export const make_uint3 = Uint3.make;
export const make_uint4 = Uint4.make;
