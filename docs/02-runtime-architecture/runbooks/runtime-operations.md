# 02.01 Runbook: Runtime Operations

## Run a Script (root wrapper)
```bash
./ferrite-run ferrite-os/workloads/finetune_engine/scripting_finetune.rs
```

## Run a Script with Hook Wrapper (ferrite-os local wrapper)
```bash
cd ferrite-os
tooling/scripts/ferrite-run ./build/test_hook
```

## Daemon Manual Run
```bash
./ferrite-daemon serve
```

## Daemon Client Commands
```bash
./ferrite-daemon ping
./ferrite-daemon metrics
./ferrite-daemon health
./ferrite-daemon run-list
./ferrite-daemon run-file crates/internal/ptx-runner/src/main.rs -- --list
./ferrite-daemon run-entry crates/internal/ptx-runner/src/main.rs#main -- --list
./ferrite-daemon run-entry crates/public/ferrite-apps/src/bin/ascii_tensor_orbit.rs#main
./ferrite-daemon run-entry crates/public/ferrite-apps/src/bin/gpu_heartbeat_journal.rs#main
./ferrite-daemon run-entry crates/public/ferrite-apps/src/bin/memory_churn_guard.rs#main
./ferrite-daemon run-entry crates/public/ferrite-apps/src/bin/ipc_ring_pipeline.rs#main
# endless mode:
DURATION=0 ferrite-daemon run-entry crates/public/ferrite-apps/src/bin/ascii_tensor_orbit.rs#main
DURATION=0 ferrite-daemon run-entry crates/public/ferrite-apps/src/bin/memory_churn_guard.rs#main
```

## Service-Oriented Entrypoint Scripts
```bash
bash -n ferrite-daemon
bash -n scripts/install/lib/service.sh
```
