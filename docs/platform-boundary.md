# Platform Boundary Architecture

## Overview

Ferrite-OS uses a **thin platform boundary** pattern to support multiple
operating systems without scattering `#[cfg(...)]` through business logic.

All OS-specific operations are routed through the `ferrite-platform` crate,
which exposes small, concrete function APIs.  Core crates (`ptx-daemon`,
`ptx-app`, `ferrite-connectors`, etc.) call these APIs instead of using
`std::os::unix`, `libc`, or conditional compilation directly.

## Module Layout

```
ferrite-platform/
├── src/
│   ├── lib.rs          # Public re-exports
│   ├── ipc.rs          # IPC: Unix domain sockets / Windows named pipes
│   ├── pid.rs          # PID file + process liveness check
│   ├── signals.rs      # Lifecycle signals (shutdown, reload)
│   ├── paths.rs        # Runtime dirs, temp paths, defaults
│   ├── tty.rs          # TTY detection, null device, stdio redirection
│   └── dylib_env.rs    # Dynamic library env var (LD_LIBRARY_PATH / PATH)
└── Cargo.toml
```

## Design Principles

1. **Function-first API** — Small standalone functions and simple structs.
   No large trait hierarchies or abstract factories.

2. **Compile-time abstraction** — Platform selection uses `#[cfg(unix)]` /
   `#[cfg(windows)]` *only inside `ferrite-platform`*.  Callers see a
   unified API.

3. **Narrow seams** — Each module abstracts exactly one OS concern.  New
   platform needs (e.g., macOS Keychain) get a new module rather than
   expanding existing ones.

4. **Zero overhead on Linux** — All Unix implementations call directly into
   `libc` / `std::os::unix` with no indirection layer.

## Build Manifest Seam

The native crates (`ptx-sys`, `ptx-kernels`) use Cargo build scripts that
probe the host for CUDA, libraries, and GPU capabilities.  On Linux this
works unchanged.

For Windows (or cross-compilation), the `FERRITE_PLATFORM_MANIFEST` env var
points to a TOML file declaring pre-resolved artifacts:

```
FERRITE_PLATFORM_MANIFEST=scripts/windows_builder/ferrite_platform_manifest.toml
```

Build scripts check for this var first; if absent, they fall back to the
existing Linux probing logic.

## Windows Builder

The `scripts/windows_builder/` directory owns Windows build orchestration:

- `generate_manifest.py` — Detects toolchain, writes manifest TOML
- `build.ps1` — Sets env, invokes cargo
- `smoke_test.ps1` — Verifies compile + basic runtime

See `scripts/windows_builder/README.md` for usage.

## Callsite Migration Pattern

Before (Unix-specific):
```rust
use std::os::unix::net::UnixStream;
let stream = UnixStream::connect(socket_path)?;
```

After (platform-neutral):
```rust
use ferrite_platform::ipc::{Endpoint, IpcStream};
let endpoint = Endpoint::new(socket_path);
let stream = IpcStream::connect(&endpoint)?;
```

The `Read`, `Write`, timeout, and shutdown semantics are identical.
