# PLAN-CUDA-NEXT-08: Safe API Migration Completion (GPU Lang Runtime)

## Mission
Complete migration from deprecated runtime kernel paths to `ptx_kernels::safe_api`.

## Win Condition
- Build/tests pass.
- Deprecated runtime launch path is removed or isolated.
- Guard-enforced safe launches are default in gpu-lang runtime.

## Primary Targets
- `ferrite-gpu-lang/src/lib.rs`
- `ferrite-gpu-lang/src/runtime/tensor.rs`
- `ferrite-os/internal/ptx-kernels/src/safe_api/*`

## Steps
1. Map deprecated call sites to safe API equivalents.
2. Implement adapters for guarded buffer/context creation.
3. Remove `#[allow(deprecated)]` where migration complete.
4. Add regression tests for launch failures and success paths.

## Gates
```bash
cargo build --release -p ferrite-gpu-lang
cargo test -p ferrite-gpu-lang
cd ferrite-os && cargo test -p ptx-kernels
```

## Done
- [ ] Deprecated runtime launch paths addressed.
- [ ] Safe API path fully integrated.
- [ ] Build/test/integration pass.
