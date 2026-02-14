# 99.00 Ferrite Documentation Rewrite Plan

This plan is for an LLM agent to execute. Read it top-to-bottom, then work
through the tasks in order. Each task says exactly what file to edit, what
is wrong, and what the correct content should reflect.

---

## Ground Truth (read these first)

Before touching any doc, read these source-of-truth files to understand the
final system state. Do NOT guess — derive all facts from code.

| Source file | What it tells you |
|-------------|-------------------|
| `scripts/install/install.sh` | Main install flow, source order, CORE_ONLY conditionals |
| `scripts/install/lib/args.sh` | All CLI flags including `--core-only` |
| `scripts/install/lib/env.sh` | `usage()` output — canonical flag docs |
| `scripts/install/lib/cuda.sh` | `ensure_cuda_toolkit()`, `setup_cuda_env()`, CUPTI |
| `scripts/install/lib/preflight.sh` | Preflight calls `ensure_cuda_toolkit` not `need_cmd nvcc` |
| `scripts/install/lib/build.sh` | `run_build()` — 3-step core-only vs 9-step full, `print_success()` |
| `scripts/install/lib/service.sh` | Systemd unit generation, LIBTORCH guard for core-only |
| `ferrite-os/crates/internal/ptx-sys/build.rs` | Hard panic on missing `libptx_os.so` |
| `ferrite-os/crates/internal/ptx-kernels/build.rs` | `find_cuda_kernels()`, default `sm_75`, `CUDA_HOME` support |
| `ferrite-os/crates/internal/ptx-daemon/ferrite-daemon.toml` | Defaults: `pool_fraction=0.25`, `max_streams=128`, new sections |
| `ferrite-os/Cargo.toml` | `rust-version = "1.75"` |
| `ferrite-os/native/core/include/gpu/gpu_hot_runtime.h` | `GPU_HOT_IPC_KEY_PREFIX/SUFFIX/MAX_LEN` (per-UID) |
| `ferrite-os/native/core/runtime/modules/hot_runtime_init.inl` | `ipc_key` constructed with `getuid()` |
| `INSTALL.md` | Canonical install instructions (already updated) |

---

## What exists in the codebase (verified)

All of these paths are real and should be referenced accurately:

```
ferrite-os/tooling/scripts/ptx_doctor.sh   # exists
ferrite-daemon                            # exists (root wrapper; canonical daemon entrypoint)
ferrite-os/ferrite-daemon.sh              # exists (local wrapper)
ferrite-os/tooling/scripts/ferrite-run     # exists (local wrapper)
ferrite-run                               # exists (root wrapper)
scripts/install.sh                        # exists (installer entrypoint)
scripts/uninstall.sh                      # exists

ferrite-os/crates/internal/ptx-daemon/src/
  tui/                                    # extensive: app.rs, commands/, editor/, etc.
  commands.rs
  lifecycle.rs
  server/command_pipeline.rs
  policy/engine.rs, audit.rs, decision.rs
  scheduler_commands.rs
  supervisor.rs
  job_store.rs
  event_stream.rs

ferrite-os/native/core/
  hooks/    kernels/    memory/    os/    runtime/    include/

scripts/install/lib/
  args.sh  build.sh  cuda.sh  diag.sh  env.sh
  libtorch.sh  policy.sh  preflight.sh  rust.sh  service.sh
```

---

## What was deleted and must be de-referenced

The `docs/plans/` directory was deleted. All references to it must be removed:

- `docs/plans/PLAN-INDEX.txt`
- `docs/plans/PLAN-CUDA-HARDEN-01-runtime-safety.md`
- `docs/plans/PLAN-CUDA-HARDEN-02-control-plane-policy-enforcement.md`
- `docs/plans/PLAN-CUDA-HARDEN-03-multitenant-isolation-and-quotas.md`
- `docs/plans/PLAN-CUDA-HARDEN-04-recovery-and-determinism.md`
- `docs/plans/PLAN-CUDA-HARDEN-05-kernel-assurance-and-release-gates.md`
- `docs/plans/PLAN-CUDA-NEXT-06-scheduler-state-integration.md` through `10`
- `docs/plans/PLAN-UX-RUST-ENTRY-RUNNER.md`
- `docs/diagnostics-inventory-plan-g.md`
- `docs/integration-compat-contract-plan-e.md`

---

## Task 1: Fix `docs/README.md`

Problems:
- Reading order references `plans/PLAN-INDEX.txt` — deleted
- Structure section references `plans/` — deleted
- "CUDA Hardening Plans" section links 5 deleted files

Fix:
- Remove `plans/PLAN-INDEX.txt` from reading order
- Remove `plans/` from structure
- Remove entire "CUDA Hardening Plans" section
- Add `scripts/install/lib/` to "Source-of-Truth Files" list
- Add `ferrite-os/crates/internal/ptx-daemon/ferrite-daemon.toml` to source-of-truth list

---

## Task 2: Fix `docs/01-system-overview/README.md`

Problems:
- "Primary Runtime Path" step 2 says "CUDA runtime libraries and external libtorch
  are provisioned" — in core-only mode libtorch is skipped
- "Invariants" says "external binaries are fetched during install" — not true for
  core-only mode (libtorch is the only external binary, and it's optional now)
- Missing mention that CUDA toolkit is auto-installed if `nvcc` is absent
- Missing mention of `--core-only` mode

Fix:
- Update "Primary Runtime Path" to note the CUDA toolkit auto-install step and
  that libtorch provisioning is conditional on not using `--core-only`
- Update "Invariants" to say libtorch fetch is optional (skipped with `--core-only`)
- Add invariant: "Only the NVIDIA driver is required pre-installed; CUDA toolkit
  is auto-installed by the installer if absent"

---

## Task 3: Fix `docs/01-system-overview/contracts/subsystem-b-cuda-os-layer.md`

Problems:
- Missing the per-UID IPC key namespacing change
- Missing mention of `GPU_HOT_IPC_KEY_PREFIX/SUFFIX/MAX_LEN` replacing `GPU_HOT_IPC_KEY`

Fix:
- Add to "No-Break Rules": "IPC shared memory key is per-UID
  (`/ptx_os_{uid}_v1`) to prevent collisions on multi-user systems"
- Add to "Public Interfaces": mention the IPC key constants in
  `gpu_hot_runtime.h` (`GPU_HOT_IPC_KEY_PREFIX`, `GPU_HOT_IPC_KEY_SUFFIX`,
  `GPU_HOT_IPC_KEY_MAX_LEN`)

---

## Task 4: Fix `docs/01-system-overview/contracts/subsystem-d-installer-provisioner.md`

Problems:
- Missing `--core-only` flag
- Missing `ensure_cuda_toolkit` auto-install behavior
- Missing the modular installer lib structure (`scripts/install/lib/*.sh`)

Fix:
- Add to "Public Interfaces": `--core-only` flag skips libtorch and torch crates
- Add to "No-Break Rules": "CUDA toolkit auto-install via `ensure_cuda_toolkit()`
  when `nvcc` is absent — only NVIDIA driver required pre-installed"
- Update "Owned Paths" to explicitly list `scripts/install/lib/*.sh` with a note
  that these are the modular installer libraries (args, build, cuda, diag, env,
  libtorch, policy, preflight, rust, service)

---

## Task 5: Fix `docs/01-system-overview/contracts/subsystem-f-daemon-service-lifecycle.md`

Problems:
- Missing updated daemon config defaults (`pool_fraction=0.25`, `max_streams=128`)
- Missing new config sections (`[scheduler]`, `[control_plane]`, `[jobs]`)
- Missing that systemd unit generation guards LIBTORCH env for `--core-only`
- "Owned Paths" lists a legacy static daemon service path, but the actual
  service unit is generated by `scripts/install/lib/service.sh` at install time

Fix:
- Add to "Public Interfaces": daemon config defaults and new sections
- Add note that systemd unit is generated dynamically by `service.sh`, not from
  a static file
- Add to "No-Break Rules": "systemd unit must conditionally include LIBTORCH
  environment only when not in core-only mode"

---

## Task 6: Fix `docs/01-system-overview/contracts/subsystem-g-diagnostics-error-contracts.md`

Problems:
- References `diag_emit` but doesn't document the structured diagnostic codes
  (INS-CUDA-0010 through INS-CUDA-0012, etc.)
- Missing the new diagnostic codes from `ensure_cuda_toolkit()`

Fix:
- Add note that `scripts/install/lib/diag.sh` defines the `diag_emit` function
- Add the CUDA toolkit auto-install diagnostic codes:
  `INS-CUDA-0010` (cannot auto-install), `INS-CUDA-0011` (nvcc not on PATH),
  `INS-CUDA-0012` (auto-installed successfully)

---

## Task 7: Fix `docs/02-runtime-architecture/README.md`

Problems:
- "Critical Runtime Boundaries" → "Build boundary" should mention the hard panic
  in `ptx-sys/build.rs` when `libptx_os.so` is missing
- Missing mention of per-UID IPC key in "Memory Model" or a new "IPC Model" section
- Missing mention that `ptx-kernels` defaults to `sm_75` and has `find_cuda_kernels()`

Fix:
- Update "Build boundary" to say: "`ptx-sys` build script panics with actionable
  message if `libptx_os.so` is not found — enforces that `make all` runs first"
- Add "IPC Model" or update "Memory Model" to note: shared memory key is per-UID
  (`/ptx_os_{uid}_v1`), preventing collisions between users on shared machines
- Add to "Failure Surfaces": "Build with wrong default SM (ptx-kernels defaults
  to sm_75 if CUDA_ARCH is unset)"

---

## Task 8: Fix `docs/03-build-and-portability/README.md`

Problems:
- "Install Pipeline" stages are stale:
  - Missing stage: "CUDA toolkit auto-install (if nvcc absent)"
  - Stage 3 "Compatibility resolution" happens only in non-core-only mode
  - Stage 5 "libtorch provisioning" happens only in non-core-only mode
  - Stage 7 "Build Rust/Torch integration layers" only in non-core-only mode
- "Portability Checklist" doesn't account for core-only path
- Missing `--core-only` in "Pinning Modes" section

Fix:
- Rewrite "Install Pipeline" to show the actual stage order from `scripts/install.sh`:
  1. Platform/arch detection
  2. Defaults + CLI parsing
  3. Preflight checks (host tools, CUDA toolkit auto-install, Rust toolchain)
  4. Compatibility resolution (skipped if `--core-only`)
  5. CUDA env setup + SM detection
  6. Build env export
  7. Libtorch provisioning (skipped if `--core-only`)
  8. Build (`run_build` — 3 steps core-only, 9 steps full)
  9. Optional systemd service install
- Add `--core-only` example to "Pinning Modes" section
- Update "Portability Checklist" with a core-only variant

---

## Task 9: Fix `docs/03-build-and-portability/runbooks/install-and-provision.md`

Problems:
- "Preconditions" says "CUDA toolkit (nvcc available)" — stale; toolkit is now
  auto-installed, only NVIDIA driver is required
- Missing `--core-only` install example
- Missing validation commands for the new installer libs

Fix:
- Change preconditions to: "Linux (x86_64 or aarch64), NVIDIA driver installed.
  CUDA toolkit is auto-installed if absent."
- Add core-only install example:
  ```
  ./scripts/install.sh --core-only
  ./scripts/install.sh --core-only --sm 86
  ```
- Add validation for installer libs:
  ```
  bash -n scripts/install/lib/*.sh
  ```

---

## Task 10: Fix `docs/03-build-and-portability/runbooks/debugging-and-remediation.md`

Problems:
- Missing section for CUDA toolkit auto-install failures
- Missing section for per-UID IPC key debugging
- "If libptx_os.so Is Missing" section should mention the panic message from
  ptx-sys build.rs (since users will see it)

Fix:
- Update "If libptx_os.so Is Missing" to mention the build.rs panic message
  and that it enforces build order
- Add section: "If CUDA Toolkit Auto-Install Fails" — check diag codes
  INS-CUDA-0010/0011, manual install URL, PATH issues
- Add section: "If Shared Memory Collides Between Users" — explain per-UID
  key (`/ptx_os_{uid}_v1`), how to inspect with `ls /dev/shm/ptx_os_*`

---

## Task 11: Fix `docs/04-llm-programming-guides/architecture-walkthrough.md`

Problems:
- "Layer Map" → installer layer only lists `scripts/install.sh`, `compat.toml`,
  `scripts/resolve_cuda_compat.sh` — missing `scripts/install/lib/*.sh`
- Missing daemon config in any layer
- Missing mention of `ferrite-os/crates/internal/ptx-daemon/ferrite-daemon.toml`

Fix:
- Update layer 1 (Installer) to include `scripts/install/lib/*.sh` and
  `scripts/install/install.sh`
- Add daemon config `ferrite-os/crates/internal/ptx-daemon/ferrite-daemon.toml`
  to the daemon layer (or create a note alongside layer 5)
- Add to "Change Routing Rules": "If issue mentions --core-only or installer
  flags, start in `scripts/install/lib/args.sh` and `build.sh`"
- Add to "Change Routing Rules": "If issue mentions CUDA auto-install or
  toolkit, start in `scripts/install/lib/cuda.sh`"

---

## Task 12: Fix `docs/04-llm-programming-guides/change-playbooks.md`

Problems:
- Playbook A (Installer) doesn't mention `scripts/install/lib/*.sh`
- Playbook A validation doesn't include `bash -n scripts/install/lib/*.sh`
- Missing playbook for core-only mode changes

Fix:
- Update Playbook A to reference `scripts/install/lib/*.sh` for CLI/workflow
  changes (especially `args.sh`, `build.sh`, `cuda.sh`)
- Add `bash -n scripts/install/lib/*.sh` to validation commands
- Add Playbook F: "Core-Only Mode Change" — when modifying what gets built
  in core-only vs full mode, touch `args.sh`, `build.sh`, `install.sh`,
  `service.sh`, and `print_success()` in `build.sh`

---

## Task 13: Fix `docs/04-llm-programming-guides/debugging-playbook.md`

Problems:
- Missing "CUDA Toolkit Not Found / Auto-Install Failed" section
- Missing "Shared Memory IPC Collision" section
- "Linker/Shared Library Errors" should mention the ptx-sys hard panic

Fix:
- Add section: "CUDA Toolkit Auto-Install Failed" with checks for
  `ensure_cuda_toolkit`, PATH, diag codes INS-CUDA-0010/0011/0012
- Add section: "Shared Memory IPC Collision" with `ls /dev/shm/ptx_os_*`
  and explanation of per-UID key
- Update "Linker/Shared Library Errors" check 1 to note: "If missing,
  `cargo build` will panic with an actionable message from ptx-sys build.rs.
  Run `cd ferrite-os && make all` to fix."

---

## Task 14: Fix `docs/04-llm-programming-guides/safe-edit-rules.md`

Problems:
- Rule 4 validation commands don't include `scripts/install/lib/*.sh`

Fix:
- Add `bash -n scripts/install/lib/*.sh` to the validation commands in Rule 4

---

## Task 15: Fix `docs/04-llm-programming-guides/prompting-contract.md`

No major issues. Minor update:
- Example task template could mention `--core-only` as a scope option

---

## Task 16: Fix `docs/01-system-overview/contracts/README.md`

No structural issues. The contract index and dependency directions are still
correct. No changes needed unless other tasks surface new dependencies.

---

## Task 17: Fix `docs/04-llm-programming-guides/contracts/task-routing-and-gates.md`

Problems:
- "No-Break Editing Policy" should mention `--core-only` mode as a behavioral
  contract that must be preserved
- Missing validation gate for installer lib syntax

Fix:
- Add to "No-Break Editing Policy": "Preserve `--core-only` mode behavior —
  it must skip libtorch download and torch-dependent build steps"
- Add validation gate: "After installer changes, run
  `bash -n scripts/install/lib/*.sh`"

---

## Execution Rules for the Agent

1. **Read before write.** Read the source-of-truth file before editing any doc.
   Do not guess what the code does.
2. **Keep docs factual.** No aspirational language, no benchmarks, no "will be"
   or "planned" statements. Document what IS.
3. **Preserve structure.** Keep the existing section hierarchy (01-04). Do not
   create new top-level sections or reorganize the tree.
4. **Keep it concise.** These docs serve engineers and LLM agents. No filler
   paragraphs. Bullet points and tables are preferred.
5. **Validate after.** After all edits, run:
   ```bash
   # Confirm no broken internal links
   grep -rn 'plans/' docs/ | grep -v DOCS-REWRITE-PLAN
   # Should return nothing (all plan references removed)
   ```
6. **Do not touch code files.** This plan is docs-only. If you find a code issue
   while reading source files, note it but do not fix it.
7. **Work in task order.** Tasks 1-17 are ordered by dependency — the README
   files are fixed first so cross-references are consistent when you edit
   downstream files.
