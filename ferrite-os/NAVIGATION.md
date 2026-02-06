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

- **[systemic_benchmarks/](systemic_benchmarks/)** - Comprehensive validation suite
  - Performance benchmarks
  - Stress tests
  - Stability validation

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

- **[docs/](docs/)** - All documentation
  - `API_REFERENCE.md` - Complete API documentation
  - `ARCHITECTURE.md` - System design and implementation
  - `INSTALL.md` - Installation guide
  - `DOCUMENTATION_INDEX.md` - Documentation index
  - `SAFETY.md` - Safety analysis and unsafe code audit

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
cargo run --release --bin allocation_comparison -p systemic_benchmarks

# Start daemon
cargo run --release -p ferrite-daemon -- serve --config internal/ptx-daemon/dev-config.toml
```

## Documentation Guide

- **Getting Started**: See [docs/INSTALL.md](docs/INSTALL.md)
- **System Overview**: See [README.md](README.md)
- **API Reference**: See [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
- **Architecture**: See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
