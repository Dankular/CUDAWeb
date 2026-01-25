/**
 * Main entry point for the Gaussian Splatting demo
 */

import { createGaussianSplattingApp, type LoadProgress, type RenderStats } from './index';
import { checkWebGPUSupport } from './core';

// DOM elements
const canvas = document.getElementById('render-canvas') as HTMLCanvasElement;
const webgpuStatus = document.getElementById('webgpu-status') as HTMLSpanElement;
const loadBtn = document.getElementById('load-btn') as HTMLButtonElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const bgColorInput = document.getElementById('bg-color') as HTMLInputElement;
const resetCameraBtn = document.getElementById('reset-camera') as HTMLButtonElement;

const statsPanel = document.getElementById('stats') as HTMLDivElement;
const statGaussians = document.getElementById('stat-gaussians') as HTMLSpanElement;
const statVisible = document.getElementById('stat-visible') as HTMLSpanElement;
const statFrametime = document.getElementById('stat-frametime') as HTMLSpanElement;
const statFps = document.getElementById('stat-fps') as HTMLSpanElement;

const progressOverlay = document.getElementById('progress-overlay') as HTMLDivElement;
const progressFill = document.getElementById('progress-fill') as HTMLDivElement;
const progressText = document.getElementById('progress-text') as HTMLDivElement;

const errorMessage = document.getElementById('error-message') as HTMLDivElement;

// App instance
let app: Awaited<ReturnType<typeof createGaussianSplattingApp>> | null = null;

// FPS tracking
let frameCount = 0;
let lastFpsUpdate = performance.now();

async function init(): Promise<void> {
  // Check WebGPU support
  const support = await checkWebGPUSupport();

  if (!support.supported) {
    webgpuStatus.textContent = `WebGPU not supported`;
    webgpuStatus.style.color = '#e74c3c';
    showError(`WebGPU is not supported in this browser. ${support.reason}`);
    loadBtn.disabled = true;
    return;
  }

  webgpuStatus.textContent = 'WebGPU Ready';
  webgpuStatus.style.color = '#2ecc71';

  // Initialize app
  try {
    app = await createGaussianSplattingApp(canvas, {
      backgroundColor: hexToRgb(bgColorInput.value),
    });

    // Set up callbacks
    app.onProgress(handleProgress);
    app.onStats(handleStats);
    app.onErrorCallback(handleError);

    // Start rendering
    app.startRendering();

    console.log('Gaussian Splatting app initialized');
  } catch (error) {
    showError(`Failed to initialize: ${(error as Error).message}`);
  }

  // Set up event listeners
  setupEventListeners();
}

function setupEventListeners(): void {
  // File input
  fileInput.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file || !app) return;

    try {
      showProgress();
      await app.loadModelFromFile(file);
      hideProgress();
    } catch (error) {
      hideProgress();
      showError(`Failed to load model: ${(error as Error).message}`);
    }
  });

  // Background color
  bgColorInput.addEventListener('change', () => {
    if (!app) return;
    const [r, g, b] = hexToRgb(bgColorInput.value);
    app.setBackgroundColor(r, g, b);
  });

  // Reset camera
  resetCameraBtn.addEventListener('click', () => {
    if (!app) return;
    const controls = app.getControls();
    controls.setOrbit(Math.PI / 4, Math.PI / 3, 5);
    controls.setTarget(0, 0, 0);
  });

  // Handle resize
  window.addEventListener('resize', () => {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
  });

  // Initial canvas size
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
}

function handleProgress(progress: LoadProgress): void {
  const percent = progress.total > 0 ? (progress.loaded / progress.total) * 100 : 0;
  progressFill.style.width = `${percent}%`;

  let text = 'Loading...';
  if (progress.phase === 'metadata') {
    text = 'Loading metadata...';
  } else if (progress.phase === 'weights') {
    const loadedMB = (progress.loaded / (1024 * 1024)).toFixed(1);
    const totalMB = (progress.total / (1024 * 1024)).toFixed(1);
    text = `Loading weights: ${loadedMB} / ${totalMB} MB`;
    if (progress.currentWeight) {
      text += ` (${progress.currentWeight})`;
    }
  } else if (progress.phase === 'complete') {
    text = 'Complete!';
    setTimeout(hideProgress, 500);
  }

  progressText.textContent = text;
}

function handleStats(stats: RenderStats): void {
  statGaussians.textContent = formatNumber(stats.numGaussians);
  statVisible.textContent = formatNumber(stats.visibleGaussians);
  statFrametime.textContent = `${stats.totalTimeMs.toFixed(1)} ms`;

  // Update FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFpsUpdate > 1000) {
    const fps = Math.round(frameCount * 1000 / (now - lastFpsUpdate));
    statFps.textContent = String(fps);
    frameCount = 0;
    lastFpsUpdate = now;
  }
}

function handleError(error: Error): void {
  showError(error.message);
}

function showProgress(): void {
  progressOverlay.classList.add('visible');
  progressFill.style.width = '0%';
  progressText.textContent = 'Loading...';
}

function hideProgress(): void {
  progressOverlay.classList.remove('visible');
}

function showError(message: string): void {
  errorMessage.textContent = message;
  errorMessage.classList.add('visible');
  setTimeout(() => {
    errorMessage.classList.remove('visible');
  }, 5000);
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return [1, 1, 1];
  return [
    parseInt(result[1], 16) / 255,
    parseInt(result[2], 16) / 255,
    parseInt(result[3], 16) / 255,
  ];
}

function formatNumber(num: number): string {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return String(num);
}

// Handle drag and drop
document.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.stopPropagation();
});

document.addEventListener('drop', async (e) => {
  e.preventDefault();
  e.stopPropagation();

  const file = e.dataTransfer?.files?.[0];
  if (!file || !app) return;

  const validExtensions = ['.ply', '.splat', '.ksplat'];
  const hasValidExtension = validExtensions.some((ext) =>
    file.name.toLowerCase().endsWith(ext)
  );

  if (!hasValidExtension) {
    showError('Please drop a .ply or .splat file');
    return;
  }

  try {
    showProgress();
    await app.loadModelFromFile(file);
    hideProgress();
  } catch (error) {
    hideProgress();
    showError(`Failed to load model: ${(error as Error).message}`);
  }
});

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
