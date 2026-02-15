# Ferrite-OS Development Workflow

## Two-Repo Architecture

Ferrite-OS uses a two-repo model: a private dev monorepo for full-stack development, and a public prod repo that ships the clean, minimal OS.

```
ferriterc/ (dev)                    ferriteos/ (prod)
├── ferrite-os/                     ├── crates/
│   ├── crates/                     │   ├── public/
│   │   ├── public/                 │   ├── internal/
│   │   ├── internal/               │   └── external/
│   │   └── ...                     │       └── ferrite-gpu-lang/
│   ├── native/                     ├── native/
│   └── lib/                        ├── lib/
├── ferrite-gpu-lang/               ├── tooling/
├── external/                       ├── Makefile
│   ├── cudarc-ptx/     (22 MB)    └── README.md
│   ├── aten-ptx/
│   ├── candle-ptx/                 13 MB total source
│   ├── libtorch/       (5.2 GB)
│   ├── onnxruntime/    (685 MB)
│   └── ...
│
~18 GB with build cache
```

## What Lives Where

### Dev repo (ferriterc/)

Everything. This is the full development workspace with:

- All Ferrite-OS crates (runtime, tensor, compute, autograd, graphics, etc.)
- `ferrite-gpu-lang` as a sibling directory
- External ML framework forks (`cudarc-ptx`, `aten-ptx`, `candle-ptx`, `onnxruntime-ptx`)
- Pre-built libtorch binaries (~5.2 GB)
- The TUI daemon (`ptx-daemon`) and OS facade (`ptx-os`)
- Full build cache for fast incremental rebuilds

This is where active development happens. All experimental features, framework integrations, and heavy dependencies live here.

### Prod repo (ferriteos/)

The shippable product. Contains:

- All core crates (runtime, tensor, compute, autograd, compiler, kernels)
- Graphics stack (ptx-render, ferrite-graphics, ferrite-window)
- GPU language layer (ferrite-gpu-lang) with heavy deps commented out
- Native CUDA/C source and pre-built .so libraries
- Full build tooling (Makefile, scripts, env detection)
- Installation guide with four tiers

Does NOT contain:

- libtorch, onnxruntime, or other multi-GB binaries
- cudarc-ptx fork (optional, user clones separately)
- ptx-daemon (TUI dashboard -- corrupted source, to be restored)
- ptx-os (high-level facade -- corrupted source, to be restored)

## Syncing Dev to Prod

When ready to ship changes from dev to prod:

```bash
# 1. Sync source (excludes build cache and git history)
rsync -a --exclude='target' --exclude='.git' --delete \
  ~/testbuild/fresh/ferriterc/ferrite-os/ ~/ferrite-os-clean/

# 2. Review what changed
cd ~/ferrite-os-clean
git status
git diff

# 3. Commit and push
git add -A
git commit -m "description of changes"
git push
```

Important: after rsync, verify the workspace Cargo.toml still excludes broken crates and that ferrite-gpu-lang paths are correct. The dev repo has different path layouts than prod.

### Path differences

| Dependency | Dev path (from ptx-daemon) | Prod path (from ptx-daemon) |
|---|---|---|
| ferrite-gpu-lang | `../../../../ferrite-gpu-lang` | `../../external/ferrite-gpu-lang` |

| Dependency | Dev path (from ferrite-gpu-lang) | Prod path (from ferrite-gpu-lang) |
|---|---|---|
| ptx-runtime | `../ferrite-os/crates/public/ptx-runtime` | `../../public/ptx-runtime` |
| ptx-sys | `../ferrite-os/crates/internal/ptx-sys` | `../../internal/ptx-sys` |
| cudarc | `../external/cudarc-ptx` | commented out |

## Install Tiers (User-Facing)

The prod repo README documents four install tiers:

| Tier | What | Build command | Size |
|------|------|---------------|------|
| 1 | Core runtime + compute + autograd | `make && cargo build --release` | 13 MB src |
| 2 | + Graphics + windowing | same (included automatically) | same |
| 3 | + GPU language (JIT, vision) | same (included automatically) | same |
| 4 | + torch/candle/ONNX | uncomment features + clone external repos | +6 GB |

Tiers 1-3 build from the same command with no extra setup. Tier 4 requires the user to manually enable optional features and provide the external repositories.

## Design Rationale

**Why two repos?**

The dev monorepo has ~6 GB of ML framework binaries and experimental code that no end user needs. Shipping that as the install would be hostile. The prod repo gives users a 13 MB clone that builds in minutes.

**Why not git submodules?**

The external deps (cudarc-ptx, aten-ptx) are patched forks with TLSF integration. Submodules would expose internal fork URLs and add clone complexity. For Tier 4 users who want ML frameworks, explicit clone instructions are clearer.

**Why comment out instead of delete?**

The ML framework integration code is still in ferrite-gpu-lang behind `#[cfg(feature = "...")]` guards. Commenting out the Cargo.toml deps (instead of deleting them) means a Tier 4 user just uncomments four lines -- the code, tests, and examples are already there.

**Why exclude ptx-daemon and ptx-os?**

Their source files are corrupted (identifiers stripped). Once restored in the dev repo, they can be re-added to the prod workspace. The core OS functionality is unaffected -- the daemon is a TUI convenience layer, and ptx-os is a high-level facade.
