# PLAN-CUDA-NEXT-07: Compiler Pass Realization (Constant Fold + Dead Code)

## Mission
Implement real graph rewrites for constant folding and dead code elimination in `ptx-compiler`.

## Win Condition
- Build/tests pass.
- Passes produce structural graph changes when applicable.
- Backend compile path remains stable.

## Primary Targets
- `ferrite-os/internal/ptx-compiler/src/passes/constant_fold.rs`
- `ferrite-os/internal/ptx-compiler/src/passes/dead_code.rs`
- `ferrite-os/internal/ptx-compiler/src/ir/graph.rs`

## Steps
1. Implement minimal constant evaluator for supported ops.
2. Rewrite nodes/tensors with constant outputs.
3. Rebuild graph removing unreachable nodes.
4. Add invariants: output preservation, metadata correctness, idempotence.

## Gates
```bash
cd ferrite-os
cargo build --workspace --release
cargo test -p ptx-compiler
```

## Done
- [ ] Constant fold and DCE are no longer no-op.
- [ ] Tests cover mixed constant/non-constant graphs.
- [ ] Build/test/integration pass.
