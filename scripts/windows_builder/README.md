# Windows Builder

Platform adjuster tooling for building Ferrite-OS on Windows.

## Architecture

Ferrite-OS uses a **thin platform boundary** (`ferrite-platform` crate) to
abstract OS-specific operations.  On Linux, the standard Unix domain socket,
signal, and process APIs are used.  On Windows, the platform crate maps to
named pipes, console events, and Win32 process APIs.

The `scripts/windows_builder/` directory owns the Windows build orchestration layer:

1. **`generate_manifest.py`** — Detects the Windows toolchain (Rust target,
   MSVC tools, CUDA paths) and writes a `FERRITE_PLATFORM_MANIFEST` TOML file
   that the Cargo build scripts consume.

2. **`build.ps1`** — Invokes `generate_manifest.py`, then runs `cargo build`
   with the manifest env var set.

3. **`smoke_test.ps1`** — Starts the daemon, connects a client, and verifies a
   basic IPC roundtrip.

## Prerequisites

- **Rust** (stable, `x86_64-pc-windows-msvc` target)
- **Python 3.8+** (for manifest generation)
- **MSVC Build Tools** (Visual Studio 2019+ or Build Tools)
- **CUDA Toolkit** (optional — only needed for GPU kernel compilation)

## Quick Start

```powershell
# From the repository root:
cd scripts/windows_builder

# 1. Generate the build manifest
python generate_manifest.py

# 2. Build
.\build.ps1

# 3. Smoke test
.\smoke_test.ps1
```

## Manifest Format

The generated `ferrite_platform_manifest.toml` follows this schema:

```toml
[ptx-sys]
lib_dirs = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\lib\\x64"]
link_libs = ["ptx_os", "cudart", "cublas"]
link_kind = "dylib"
include_dirs = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\include"]
gpu_sm = "sm_86"

[ptx-kernels]
cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6"
cuda_arch = "sm_86"
lib_dirs = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\lib\\x64"]
link_libs = ["cudart", "cuda"]
skip_kernel_build = false
```

If CUDA is not available, set `skip_kernel_build = true` in `[ptx-kernels]`
and provide pre-built kernel libraries via `lib_dirs`.

## How It Works

```
generate_manifest.py
        │
        ▼
ferrite_platform_manifest.toml
        │
        ▼  (FERRITE_PLATFORM_MANIFEST env var)
cargo build
        │
        ├──▶ ptx-sys/build.rs     → reads manifest → emits link directives
        └──▶ ptx-kernels/build.rs → reads manifest → emits link directives
```

The manifest seam is optional.  When `FERRITE_PLATFORM_MANIFEST` is **not**
set, both build scripts fall back to their existing Linux host-probing behavior.

## Remaining TODOs

- [ ] Named pipe IPC implementation in `ferrite-platform` (currently stubbed)
- [ ] Windows console event handler in `ferrite-platform::signals`
- [ ] TUI stdio redirection on Windows (`steal_stdio` is a no-op)
- [ ] CI pipeline for Windows builds
- [ ] Pre-built CUDA kernel libraries for skip_kernel_build mode
