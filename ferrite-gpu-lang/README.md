# ferrite-gpu-lang

Rust-first GPU language runtime on Ferrite OS.

## What this crate provides

- A small graph DSL (`Program`) with compile-time shape checks.
- Execution on `ptx-runtime` with all allocations through TLSF.
- Candle kernel-backed ops (`relu`, `tanh`, `sigmoid`, `add`, `mul`).
- Optional Torch bridge (`--features torch`) for scriptable PyTorch checks on your custom stack.
- Optional CV scripting graph (`cv::CvProgram`) for Torch-backed CV ops:
  - `conv2d`, `upsample_nearest2d`, `concat`
  - `yolo_decode`, `nms`
- Explicit CPU/GPU execution contexts:
  - `cpu(|c| { ... })`
  - `gpu(0, |g| { ... })`
  - typed transfer traits: `ToGpu` / `ToCpu`

## Scripting Examples

```bash
cd ferrite-gpu-lang
export LD_LIBRARY_PATH=../ferrite-os/lib:$LD_LIBRARY_PATH

cargo run --release --example script_runtime
cargo run --release --example script_handoff
cargo run --release --features torch --example script_cv_depth
cargo run --release --features torch --example script_cv_detect
```

## CPU -> GPU -> CPU handoff

```rust
use ferrite_gpu_lang::{cpu, gpu, CpuTensor, ToCpu, ToGpu};

let cpu_in = cpu(|_| CpuTensor::new(vec![1,1,1,4096], data))?;
let cpu_out = gpu(0, |g| {
    let gpu_in = cpu_in.to_gpu(g)?;
    let gpu_out = gpu_in.relu(g)?; // GPU stage
    gpu_out.to_cpu()               // back to CPU
})?;
```

## Runtime Script Runner

```bash
./scripts/run_foundation_stack.sh
```

This runs only runtime scripting examples from `ferrite-gpu-lang`.
