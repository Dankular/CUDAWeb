import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@shaders': resolve(__dirname, 'src/shaders'),
      '@core': resolve(__dirname, 'src/core'),
    },
  },
  build: {
    target: 'esnext',
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          'gaussian-shaders': [
            './src/shaders/gaussian/preprocess.wgsl',
            './src/shaders/gaussian/project.wgsl',
          ],
        },
      },
    },
  },
  assetsInclude: ['**/*.wgsl', '**/*.ply', '**/*.splat'],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@cudaweb/wasm'],
  },
});
