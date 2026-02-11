# Subsystem Contracts (A-G)

This directory defines architecture contracts for refactor plans A through G.
Each contract is behavior-preserving and maps to concrete repository paths.

## Contract Pages

- `subsystem-a-tui.md`
- `subsystem-b-cuda-os-layer.md`
- `subsystem-c-rust-runtime-core.md`
- `subsystem-d-installer-provisioner.md`
- `subsystem-e-external-integrations.md`
- `subsystem-f-daemon-service-lifecycle.md`
- `subsystem-g-diagnostics-error-contracts.md`

## Allowed Dependency Directions

1. Installer/compat (`D`) -> build/runtime artifacts (`B`, `C`, `F`, `E`)
2. CUDA runtime (`B`) -> exported C ABI only
3. Rust runtime/core (`C`) -> `ptx-sys` FFI and native ABI from `B`
4. Daemon/service (`F`) -> runtime APIs from `C`
5. Integrations (`E`) -> runtime/tensor APIs from `C`
6. TUI (`A`) -> daemon state/events (`F`), not CUDA core internals (`B`)
7. Diagnostics (`G`) -> all subsystems read-only for observation and error contracts

Forbidden direction baseline: lower-level runtime layers must not depend on higher-level UX/integration layers.
