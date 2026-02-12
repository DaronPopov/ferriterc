# 01.01 Subsystem Contracts (A-G)

This directory defines architecture contracts for refactor plans A through G.
Each contract is behavior-preserving and maps to concrete repository paths.

## Contract Pages

- `01.02` (`A`): `subsystem-a-tui.md`
- `01.03` (`B`): `subsystem-b-cuda-os-layer.md`
- `01.04` (`C`): `subsystem-c-rust-runtime-core.md`
- `01.05` (`D`): `subsystem-d-installer-provisioner.md`
- `01.06` (`E`): `subsystem-e-external-integrations.md`
- `01.07` (`F`): `subsystem-f-daemon-service-lifecycle.md`
- `01.08` (`G`): `subsystem-g-diagnostics-error-contracts.md`

## Allowed Dependency Directions

1. Installer/compat (`D`) -> build/runtime artifacts (`B`, `C`, `F`, `E`)
2. CUDA runtime (`B`) -> exported C ABI only
3. Rust runtime/core (`C`) -> `ptx-sys` FFI and native ABI from `B`
4. Daemon/service (`F`) -> runtime APIs from `C`
5. Integrations (`E`) -> runtime/tensor APIs from `C`
6. TUI (`A`) -> daemon state/events (`F`), not CUDA core internals (`B`)
7. Diagnostics (`G`) -> all subsystems read-only for observation and error contracts

Forbidden direction baseline: lower-level runtime layers must not depend on higher-level UX/integration layers.
