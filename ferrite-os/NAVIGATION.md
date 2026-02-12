# Ferrite-OS Repository Structure

This repository is organized into clear spatial layers:

## Top Layer: Public APIs

User-facing components and validation:

- **[ptx-runtime/](ptx-runtime/)** - Safe Rust runtime API for GPU operations
  - Memory allocation and management
  - Stream operations and parallelism
  - CUDA graphs and execution
  - Examples in `ptx-runtime/examples/`

- **[ptx-os/](ptx-os/)** - Operating system features
  - Virtual filesystem
  - Virtual memory management
  - Inter-process communication

- **[benchmarks/](benchmarks/)** - Benchmark outputs and run artifacts
  - Generated reports from benchmark runs
  - Historical perf snapshots

## Middle Layer: Implementation

Internal Rust crates (implementation details):

- **[internal/](internal/)** - All implementation crates
  - `ptx-sys/` - FFI bindings to C/CUDA core
  - `ptx-compute/` - High-level compute operations
  - `ptx-daemon/` - Multi-process daemon
  - `ptx-tensor/` - Tensor operations
  - `ptx-autograd/` - Automatic differentiation
  - `ptx-compiler/` - PTX compilation
  - `ptx-runner/` - Execution runner
  - `ptx-kernels/` - Kernel wrapper layer

## Bottom Layer: Core

C/CUDA implementation:

- **[core/](core/)** - Low-level GPU runtime
  - `core/memory/` - TLSF allocator implementation
  - `core/runtime/` - Hot runtime and execution
  - `core/os/` - OS services (VFS, VMM, IPC)
  - `core/kernels/` - CUDA kernel implementations
  - `core/hooks/` - Memory and lifecycle hooks
  - `core/include/` - C/C++ headers

## Documentation

- **[../docs/](../docs/)** - Repository-level documentation
  - `01-system-overview/README.md` - System map and contracts
  - `02-runtime-architecture/README.md` - Runtime internals
  - `03-build-and-portability/README.md` - Build/install portability
  - `04-llm-programming-guides/README.md` - Agent-oriented change guides

## Utilities

- **[scripts/](scripts/)** - Build and utility scripts
- **[lib/](lib/)** - Compiled shared libraries
- **[build/](build/)** - Build artifacts

## Quick Start

```bash
# Install and build
./setup.sh

# Run examples
cargo run --release --example telemetry_demo -p ptx-runtime

# Run benchmarks
scripts/ptx_bench_all.sh --no-build

# Start daemon
cargo run --release -p ferrite-daemon -- serve --config internal/ptx-daemon/dev-config.toml
```

## Documentation Guide

- **Getting Started**: See [../INSTALL.md](../INSTALL.md)
- **System Overview**: See [README.md](README.md)
- **Runtime Architecture**: See [../docs/02-runtime-architecture/README.md](../docs/02-runtime-architecture/README.md)
- **Build and Portability**: See [../docs/03-build-and-portability/README.md](../docs/03-build-and-portability/README.md)
