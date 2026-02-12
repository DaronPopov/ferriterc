# 04.06 Task Routing and Validation Gates

## Routing by Plan/Subsystem
- Installer/compatibility/pinning: Plan D (`Subsystem D`)
- Daemon lifecycle/service scripts: Plan F (`Subsystem F`)
- Runtime/compiler/tensor/autograd Rust internals: Plan C (`Subsystem C`)
- CUDA core runtime/kernels/allocator internals: Plan B (`Subsystem B`)
- TUI-only behavior/layout changes: Plan A (`Subsystem A`)
- Integration crates (`external/**`): Plan E (`Subsystem E`)
- Diagnostics/error contracts: Plan G (`Subsystem G`)

## No-Break Editing Policy
1. Keep public CLI/FFI/crate APIs stable unless the task explicitly allows breaks.
2. Keep fallback and compatibility-selection behavior unchanged unless task explicitly changes policy.
3. Do not document unsupported commands or features.
4. Preserve `--core-only` mode behavior — it must skip libtorch download and torch-dependent build steps.

## Validation Gates
1. Run only subsystem-relevant checks first.
2. Run cross-boundary checks when changing shared contracts.
3. Include exact commands and outcomes in completion notes.
4. After installer changes, run `bash -n scripts/install/lib/*.sh`.

## Recommended Execution Order for Large Refactors
1. Inventory/drift list
2. Contract page updates
3. Mechanical code/doc split preserving behavior
4. Validation gates
5. Final contract alignment pass
