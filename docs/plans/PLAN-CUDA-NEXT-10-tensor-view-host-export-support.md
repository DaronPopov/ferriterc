# PLAN-CUDA-NEXT-10: Tensor View Host Export Support (Non-Contiguous)

## Mission
Enable `to_vec`/typed host export from non-contiguous tensors via safe contiguous materialization fallback.

## Win Condition
- Build/tests pass.
- Non-contiguous views export successfully with correct semantics.
- Contiguous fast path remains unchanged.

## Primary Targets
- `ferrite-os/internal/ptx-tensor/src/tensor/views_exec.rs`
- `ferrite-os/internal/ptx-tensor/src/tensor.rs`
- `ferrite-gpu-lang/examples/tests/test_harden.rs`

## Steps
1. Implement non-contiguous fallback in host export path.
2. Preserve dtype and shape semantics for typed exports.
3. Add tests for transpose/permute/sliced views.
4. Validate no regression for contiguous path.

## Gates
```bash
cd ferrite-os
cargo build --workspace --release
cargo test -p ptx-tensor
cd ..
cargo run --release -p ferrite-gpu-lang --example test_harden
```

## Done
- [ ] Non-contiguous export implemented.
- [ ] View tests expanded and passing.
- [ ] Build/test/integration pass.
