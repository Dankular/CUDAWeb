/**
 * Camera System for Gaussian Splatting
 * Handles view/projection matrices and camera controls
 */

export interface CameraState {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fov: number;
  near: number;
  far: number;
  aspect: number;
}

export class Camera {
  // Camera state
  position: Float32Array;
  target: Float32Array;
  up: Float32Array;

  // Projection parameters
  fov: number = Math.PI / 4; // 45 degrees
  near: number = 0.1;
  far: number = 1000;
  aspect: number = 1;

  // Computed matrices
  viewMatrix: Float32Array;
  projMatrix: Float32Array;
  viewProjMatrix: Float32Array;
  invViewMatrix: Float32Array;

  // Cached values for shaders
  focalX: number = 0;
  focalY: number = 0;
  tanFovX: number = 0;
  tanFovY: number = 0;

  constructor(state?: Partial<CameraState>) {
    this.position = new Float32Array(state?.position ?? [0, 0, 5]);
    this.target = new Float32Array(state?.target ?? [0, 0, 0]);
    this.up = new Float32Array(state?.up ?? [0, 1, 0]);

    if (state?.fov) this.fov = state.fov;
    if (state?.near) this.near = state.near;
    if (state?.far) this.far = state.far;
    if (state?.aspect) this.aspect = state.aspect;

    this.viewMatrix = new Float32Array(16);
    this.projMatrix = new Float32Array(16);
    this.viewProjMatrix = new Float32Array(16);
    this.invViewMatrix = new Float32Array(16);

    this.updateMatrices();
  }

  setPosition(x: number, y: number, z: number): void {
    this.position[0] = x;
    this.position[1] = y;
    this.position[2] = z;
    this.updateMatrices();
  }

  setTarget(x: number, y: number, z: number): void {
    this.target[0] = x;
    this.target[1] = y;
    this.target[2] = z;
    this.updateMatrices();
  }

  setAspect(aspect: number): void {
    this.aspect = aspect;
    this.updateMatrices();
  }

  setFov(fov: number): void {
    this.fov = fov;
    this.updateMatrices();
  }

  lookAt(
    eyeX: number, eyeY: number, eyeZ: number,
    targetX: number, targetY: number, targetZ: number,
    upX: number = 0, upY: number = 1, upZ: number = 0
  ): void {
    this.position[0] = eyeX;
    this.position[1] = eyeY;
    this.position[2] = eyeZ;
    this.target[0] = targetX;
    this.target[1] = targetY;
    this.target[2] = targetZ;
    this.up[0] = upX;
    this.up[1] = upY;
    this.up[2] = upZ;
    this.updateMatrices();
  }

  updateMatrices(): void {
    this.computeViewMatrix();
    this.computeProjectionMatrix();
    this.multiplyMatrices(this.viewProjMatrix, this.projMatrix, this.viewMatrix);
    this.invertMatrix(this.invViewMatrix, this.viewMatrix);

    // Compute focal lengths and tan values for shader
    const tanHalfFov = Math.tan(this.fov / 2);
    this.tanFovY = tanHalfFov;
    this.tanFovX = tanHalfFov * this.aspect;
  }

  /**
   * Set focal lengths based on image dimensions
   */
  setFocalFromDimensions(width: number, height: number): void {
    this.focalY = height / (2 * Math.tan(this.fov / 2));
    this.focalX = this.focalY; // Assuming square pixels
    this.aspect = width / height;
    this.updateMatrices();
  }

  private computeViewMatrix(): void {
    const eye = this.position;
    const target = this.target;
    const up = this.up;

    // Forward direction
    let fx = target[0] - eye[0];
    let fy = target[1] - eye[1];
    let fz = target[2] - eye[2];
    const fLen = Math.sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fLen;
    fy /= fLen;
    fz /= fLen;

    // Right direction (cross product of forward and up)
    let sx = fy * up[2] - fz * up[1];
    let sy = fz * up[0] - fx * up[2];
    let sz = fx * up[1] - fy * up[0];
    const sLen = Math.sqrt(sx * sx + sy * sy + sz * sz);
    sx /= sLen;
    sy /= sLen;
    sz /= sLen;

    // True up direction
    const ux = sy * fz - sz * fy;
    const uy = sz * fx - sx * fz;
    const uz = sx * fy - sy * fx;

    // View matrix (column-major)
    this.viewMatrix[0] = sx;
    this.viewMatrix[1] = ux;
    this.viewMatrix[2] = -fx;
    this.viewMatrix[3] = 0;

    this.viewMatrix[4] = sy;
    this.viewMatrix[5] = uy;
    this.viewMatrix[6] = -fy;
    this.viewMatrix[7] = 0;

    this.viewMatrix[8] = sz;
    this.viewMatrix[9] = uz;
    this.viewMatrix[10] = -fz;
    this.viewMatrix[11] = 0;

    this.viewMatrix[12] = -(sx * eye[0] + sy * eye[1] + sz * eye[2]);
    this.viewMatrix[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
    this.viewMatrix[14] = fx * eye[0] + fy * eye[1] + fz * eye[2];
    this.viewMatrix[15] = 1;
  }

  private computeProjectionMatrix(): void {
    const f = 1 / Math.tan(this.fov / 2);
    const nf = 1 / (this.near - this.far);

    this.projMatrix[0] = f / this.aspect;
    this.projMatrix[1] = 0;
    this.projMatrix[2] = 0;
    this.projMatrix[3] = 0;

    this.projMatrix[4] = 0;
    this.projMatrix[5] = f;
    this.projMatrix[6] = 0;
    this.projMatrix[7] = 0;

    this.projMatrix[8] = 0;
    this.projMatrix[9] = 0;
    this.projMatrix[10] = (this.far + this.near) * nf;
    this.projMatrix[11] = -1;

    this.projMatrix[12] = 0;
    this.projMatrix[13] = 0;
    this.projMatrix[14] = 2 * this.far * this.near * nf;
    this.projMatrix[15] = 0;
  }

  private multiplyMatrices(out: Float32Array, a: Float32Array, b: Float32Array): void {
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        let sum = 0;
        for (let k = 0; k < 4; k++) {
          sum += a[i + k * 4] * b[k + j * 4];
        }
        out[i + j * 4] = sum;
      }
    }
  }

  private invertMatrix(out: Float32Array, m: Float32Array): void {
    const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
    const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
    const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
    const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

    const b00 = m00 * m11 - m01 * m10;
    const b01 = m00 * m12 - m02 * m10;
    const b02 = m00 * m13 - m03 * m10;
    const b03 = m01 * m12 - m02 * m11;
    const b04 = m01 * m13 - m03 * m11;
    const b05 = m02 * m13 - m03 * m12;
    const b06 = m20 * m31 - m21 * m30;
    const b07 = m20 * m32 - m22 * m30;
    const b08 = m20 * m33 - m23 * m30;
    const b09 = m21 * m32 - m22 * m31;
    const b10 = m21 * m33 - m23 * m31;
    const b11 = m22 * m33 - m23 * m32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    if (!det) {
      return;
    }
    det = 1.0 / det;

    out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
    out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
    out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
    out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
    out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
    out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
    out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
    out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
    out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
    out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
    out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
    out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
    out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
    out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
    out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
    out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  }

  /**
   * Get camera uniforms for shader
   */
  getUniforms(width: number, height: number): Float32Array {
    const uniforms = new Float32Array(64); // Enough for all camera data

    // View matrix (16 floats)
    uniforms.set(this.viewMatrix, 0);

    // Projection matrix (16 floats)
    uniforms.set(this.projMatrix, 16);

    // View-projection matrix (16 floats)
    uniforms.set(this.viewProjMatrix, 32);

    // Camera position (3 floats)
    uniforms[48] = this.position[0];
    uniforms[49] = this.position[1];
    uniforms[50] = this.position[2];

    // Focal lengths
    uniforms[51] = this.focalX || width / (2 * this.tanFovX);
    uniforms[52] = this.focalY || height / (2 * this.tanFovY);

    // Tan FOV
    uniforms[53] = this.tanFovX;
    uniforms[54] = this.tanFovY;

    // Dimensions
    uniforms[55] = width;
    uniforms[56] = height;

    // Near/far
    uniforms[57] = this.near;
    uniforms[58] = this.far;

    return uniforms;
  }

  /**
   * Get state for serialization
   */
  getState(): CameraState {
    return {
      position: [this.position[0], this.position[1], this.position[2]],
      target: [this.target[0], this.target[1], this.target[2]],
      up: [this.up[0], this.up[1], this.up[2]],
      fov: this.fov,
      near: this.near,
      far: this.far,
      aspect: this.aspect,
    };
  }
}

/**
 * Orbit camera controller
 */
export class OrbitControls {
  private camera: Camera;
  private canvas: HTMLCanvasElement;

  // Orbit state
  private theta: number = 0; // Horizontal angle
  private phi: number = Math.PI / 4; // Vertical angle
  private radius: number = 5;
  private targetPos: [number, number, number] = [0, 0, 0];

  // Interaction state
  private isDragging: boolean = false;
  private lastX: number = 0;
  private lastY: number = 0;

  // Sensitivity
  rotateSpeed: number = 0.005;
  zoomSpeed: number = 0.1;
  panSpeed: number = 0.01;

  constructor(camera: Camera, canvas: HTMLCanvasElement) {
    this.camera = camera;
    this.canvas = canvas;

    this.setupEventListeners();
    this.updateCameraFromOrbit();
  }

  private setupEventListeners(): void {
    this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
    this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));
    this.canvas.addEventListener('wheel', this.onWheel.bind(this));
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    // Touch support
    this.canvas.addEventListener('touchstart', this.onTouchStart.bind(this));
    this.canvas.addEventListener('touchmove', this.onTouchMove.bind(this));
    this.canvas.addEventListener('touchend', this.onTouchEnd.bind(this));
  }

  private onMouseDown(e: MouseEvent): void {
    this.isDragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
  }

  private onMouseMove(e: MouseEvent): void {
    if (!this.isDragging) return;

    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;

    if (e.buttons === 1) {
      // Left button: rotate
      this.theta -= dx * this.rotateSpeed;
      this.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.phi - dy * this.rotateSpeed));
    } else if (e.buttons === 2) {
      // Right button: pan
      const right = this.getRightVector();
      const up = this.getUpVector();

      this.targetPos[0] -= (right[0] * dx + up[0] * dy) * this.panSpeed * this.radius;
      this.targetPos[1] -= (right[1] * dx + up[1] * dy) * this.panSpeed * this.radius;
      this.targetPos[2] -= (right[2] * dx + up[2] * dy) * this.panSpeed * this.radius;
    }

    this.updateCameraFromOrbit();
  }

  private onMouseUp(): void {
    this.isDragging = false;
  }

  private onWheel(e: WheelEvent): void {
    e.preventDefault();
    this.radius *= 1 + e.deltaY * this.zoomSpeed * 0.01;
    this.radius = Math.max(0.1, Math.min(1000, this.radius));
    this.updateCameraFromOrbit();
  }

  private onTouchStart(e: TouchEvent): void {
    if (e.touches.length === 1) {
      this.isDragging = true;
      this.lastX = e.touches[0].clientX;
      this.lastY = e.touches[0].clientY;
    }
  }

  private onTouchMove(e: TouchEvent): void {
    if (!this.isDragging || e.touches.length !== 1) return;

    const dx = e.touches[0].clientX - this.lastX;
    const dy = e.touches[0].clientY - this.lastY;
    this.lastX = e.touches[0].clientX;
    this.lastY = e.touches[0].clientY;

    this.theta -= dx * this.rotateSpeed;
    this.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.phi - dy * this.rotateSpeed));

    this.updateCameraFromOrbit();
  }

  private onTouchEnd(): void {
    this.isDragging = false;
  }

  private getRightVector(): [number, number, number] {
    return [
      Math.cos(this.theta),
      0,
      -Math.sin(this.theta),
    ];
  }

  private getUpVector(): [number, number, number] {
    return [
      Math.sin(this.phi) * Math.sin(this.theta),
      Math.cos(this.phi),
      Math.sin(this.phi) * Math.cos(this.theta),
    ];
  }

  private updateCameraFromOrbit(): void {
    const x = this.targetPos[0] + this.radius * Math.sin(this.phi) * Math.sin(this.theta);
    const y = this.targetPos[1] + this.radius * Math.cos(this.phi);
    const z = this.targetPos[2] + this.radius * Math.sin(this.phi) * Math.cos(this.theta);

    this.camera.lookAt(
      x, y, z,
      this.targetPos[0], this.targetPos[1], this.targetPos[2]
    );
  }

  setOrbit(theta: number, phi: number, radius: number): void {
    this.theta = theta;
    this.phi = phi;
    this.radius = radius;
    this.updateCameraFromOrbit();
  }

  setTarget(x: number, y: number, z: number): void {
    this.targetPos = [x, y, z];
    this.updateCameraFromOrbit();
  }

  dispose(): void {
    this.canvas.removeEventListener('mousedown', this.onMouseDown.bind(this));
    this.canvas.removeEventListener('mousemove', this.onMouseMove.bind(this));
    this.canvas.removeEventListener('mouseup', this.onMouseUp.bind(this));
    this.canvas.removeEventListener('wheel', this.onWheel.bind(this));
  }
}
