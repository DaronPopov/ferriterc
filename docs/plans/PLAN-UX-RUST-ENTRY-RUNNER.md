# PLAN-UX-RUST-ENTRY-RUNNER

## Goal
Enable a JIT-like developer experience where users run daemon in one terminal, write normal Rust in any file, and execute it through Ferrite-OS from TUI/daemon commands without shell-centric workflows.

## Product Definition
"JIT-like" means:
- User edits any `.rs` file in workspace.
- User invokes run from TUI (or daemon command).
- System discovers an entrypoint, compiles incrementally, executes against daemon/runtime, and streams results.

This is not true Rust JIT; it is fast incremental compile+run with persistent runtime context.

## Win Condition
1. User can execute a Rust file outside `examples/` through daemon command.
2. User can execute by logical entry ID (not only by target name).
3. Build diagnostics and runtime logs stream back to TUI/CLI.
4. Re-run after edits is incremental and fast enough for interactive iteration.
5. Existing `ptx-runner` target workflows remain backward-compatible.

## UX Contract
- Command: `run-file <path> [--entry <name>] [-- <args...>]`
- Command: `run-entry <entry-id> [-- <args...>]`
- TUI:
  - workspace file focused -> `Enter` runs default entry
  - command palette includes discovered entries
  - build panel shows compile errors with file/line
  - run panel streams stdout/stderr + structured daemon events

## Technical Approach

### 1) Entrypoint Convention
Use lightweight convention first (no proc-macro dependency required initially):
- Detect function signature patterns in file:
  - `fn main()`
  - `fn ferrite_main(...)`
- Optional marker comment/attribute support in phase 2:
  - `#[ferrite::entry(name = "...")]`

### 2) Runner Pipeline
Add a new execution mode in `ptx-runner`:
- input: file path + optional entry + args
- resolution:
  - find workspace root
  - map file to crate context
  - compile/run with cargo command template
- environment injection:
  - daemon socket path
  - runtime context env (`PTX_CONTEXT_PTR`, etc. when available)
  - `LD_LIBRARY_PATH` merge behavior from existing runner

### 3) Daemon Integration
Add daemon command handlers:
- `run-file`
- `run-entry`
- `run-list` (discover available entries in workspace)

Add policy checks for mutating/execute actions and emit events:
- request accepted
- build started/finished
- run started/finished
- stdout/stderr chunks
- errors

### 4) TUI Integration
Use existing command/event plumbing:
- add `run-file`/`run-entry` commands
- add views:
  - build diagnostics panel
  - active run panel
  - historical runs list

### 5) Incremental Build/Cache
- Key by `(workspace, crate, file hash, entry, feature set)`
- Reuse cargo incremental artifacts; avoid bespoke compiler cache initially
- Record last successful run config for one-key re-run

## File Targets

### Runner
- `ferrite-os/internal/ptx-runner/src/main.rs`

### Daemon Command Path
- `ferrite-os/internal/ptx-daemon/src/commands.rs`
- `ferrite-os/internal/ptx-daemon/src/server/command_pipeline.rs`
- `ferrite-os/internal/ptx-daemon/src/event_stream.rs`
- `ferrite-os/internal/ptx-daemon/src/state.rs`

### TUI
- `ferrite-os/internal/ptx-daemon/src/tui/commands/run.rs`
- `ferrite-os/internal/ptx-daemon/src/tui/state/run_state.rs`
- `ferrite-os/internal/ptx-daemon/src/tui/layout/shell/*.rs`

### Docs
- `docs/02-runtime-architecture/runbooks/runtime-operations.md`
- `docs/04-llm-programming-guides/change-playbooks.md`

## Implementation Phases

### Phase A: Minimal End-to-End (`run-file`)
- Add `run-file <path>` daemon command.
- In runner, support file-path execution route.
- Stream completion + exit code.

Acceptance:
- Run a Rust file outside examples via daemon command.
- Failure reports compile errors with filename/line.

### Phase B: Entrypoint Discovery (`run-list`, `run-entry`)
- Add workspace scan for callable entries.
- Add entry IDs and command to run by ID.

Acceptance:
- `run-list` returns discoverable entries.
- `run-entry <id>` executes expected function.

### Phase C: TUI-First Flow
- Add keybind and palette actions.
- Add build/run output panes.

Acceptance:
- User can run current file from TUI with no shell.
- Build diagnostics visible and navigable.

### Phase D: Ergonomics + Reliability
- Last-run replay.
- Timeout/cancel support.
- Better policy denial UX in TUI.

Acceptance:
- Cancel works.
- Re-run uses prior config and is faster on no-op change.

## Non-Goals (for this plan)
- Full procedural macro platform launch.
- Remote/distributed execution.
- Replacing cargo with custom compiler pipeline.

## Risks and Mitigations
- Risk: Rust compile latency too high.
  - Mitigation: incremental mode + reuse crate graph + fast no-op runs.
- Risk: daemon command complexity growth.
  - Mitigation: isolate run orchestration module and typed events.
- Risk: security/policy bypass.
  - Mitigation: enforce policy in command handler before compile/run.

## Validation Commands

Core checks:
```bash
cd ferrite-os
cargo check -p ptx-runner -p ferrite-daemon
cargo test -p ferrite-daemon --lib
```

Integration checks:
```bash
# daemon running
ferrite-daemon run-list
ferrite-daemon run-file ferrite-gpu-lang/examples/scripts/script_runtime.rs
ferrite-daemon run-entry <id>
```

Regression checks:
```bash
cd ferrite-os
./scripts/release-gate.sh
```

## Completion Checklist
- [ ] `run-file` command implemented end-to-end.
- [ ] `run-list`/`run-entry` implemented with stable JSON schema.
- [ ] TUI run UX integrated (invoke + observe + cancel).
- [ ] Policy/audit/event integration complete.
- [ ] Documentation updated.
- [ ] Build/test/release gates pass.
