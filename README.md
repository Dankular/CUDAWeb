# CUDAWeb - WebGPU Gaussian Splatting

A high-performance WASM + WebGPU implementation of 3D Gaussian Splatting for browsers. This project reimplements CUDA-based Gaussian Splatting operations using WebGPU compute shaders, enabling full 360° 3D Gaussian splat rendering directly in the browser.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser Environment                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Web Frontend  │  │  Weight Loader  │  │  WASM Runtime   │  │
│  │   (TypeScript)  │  │  (Streaming)    │  │  (Rust/WASM)    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│  ┌─────────────────────────────┴─────────────────────────────┐  │
│  │                    WebGPU Abstraction Layer               │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │  │
│  │  │  Compute    │  │  Render     │  │  Memory         │    │  │
│  │  │  Pipeline   │  │  Pipeline   │  │  Management     │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                │                                │
│  ┌─────────────────────────────┴─────────────────────────────┐  │
│  │              WebGPU Compute Shaders (WGSL)                │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  • Gaussian Preprocessing    • Tile-based Rasterization  │  │
│  │  • Covariance Computation    • Alpha Blending/Sorting    │  │
│  │  • Spherical Harmonics       • Depth Sorting (Radix)     │  │
│  │  • nvdiffrast Operations     • Gradient Computation      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Full 3D Gaussian Splatting**: Complete implementation of 3DGS rendering
- **Large Model Support**: Streaming weight loader for 2GB+ models
- **WebGPU Compute Shaders**: CUDA kernels reimplemented in WGSL
- **nvdiffrast Operations**: Differentiable rasterization in WebGPU
- **WASM Acceleration**: CPU-intensive operations in Rust/WASM
- **360° View Synthesis**: Full novel view rendering capabilities

## Project Structure

```
CUDAWeb/
├── src/
│   ├── core/                 # Core WebGPU abstractions
│   │   ├── gpu-context.ts    # WebGPU device/context management
│   │   ├── buffer-manager.ts # GPU buffer allocation
│   │   └── pipeline.ts       # Compute/render pipeline builders
│   ├── shaders/              # WGSL compute shaders
│   │   ├── gaussian/         # Gaussian splatting kernels
│   │   ├── sorting/          # Radix sort for depth ordering
│   │   └── rasterize/        # Rasterization shaders
│   ├── loaders/              # Model/weight loading
│   │   ├── weight-loader.ts  # Streaming weight loader
│   │   ├── ply-loader.ts     # PLY point cloud loader
│   │   └── splat-loader.ts   # .splat format loader
│   ├── renderer/             # Rendering system
│   │   ├── gaussian-renderer.ts
│   │   └── camera.ts
│   └── wasm/                 # Rust/WASM modules
│       ├── src/
│       └── Cargo.toml
├── public/                   # Static assets
├── tests/                    # Test files
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Getting Started

```bash
# Install dependencies
npm install

# Build WASM module
cd src/wasm && wasm-pack build --target web

# Start development server
npm run dev

# Build for production
npm run build
```

## Browser Requirements

- Chrome 113+ or Firefox 121+ (WebGPU support required)
- 4GB+ GPU memory recommended for large scenes
- WebAssembly support

## Technical Details

### Gaussian Splatting Pipeline

1. **Preprocessing**: Transform 3D Gaussians to view space
2. **Covariance**: Compute 2D covariance from 3D + camera
3. **Tiling**: Assign Gaussians to screen tiles (16x16)
4. **Sorting**: Radix sort by depth within tiles
5. **Rasterization**: Alpha-blend Gaussians front-to-back

### Weight Loading Strategy

For 2GB+ models, weights are loaded in chunks:
- Streaming via Fetch API with range requests
- Chunked GPU buffer uploads
- Memory-mapped weight access
- Progressive loading with LOD support

## License

MIT License
