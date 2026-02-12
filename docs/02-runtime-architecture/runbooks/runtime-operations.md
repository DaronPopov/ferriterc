# Runbook: Runtime Operations

## Run a Script (root wrapper)
```bash
./ferrite-run finetune_engine/scripting_finetune.rs
```

## Run a Script with Hook Wrapper (ferrite-os local wrapper)
```bash
cd ferrite-os
scripts/ferrite-run ./build/test_hook
```

## Daemon Manual Run
```bash
cd ferrite-os/internal/ptx-daemon
cargo run --bin ferrite-daemon -- serve
```

## Daemon Client Commands
```bash
cd ferrite-os/internal/ptx-daemon
cargo run --bin ferrite-daemon -- ping
cargo run --bin ferrite-daemon -- metrics
cargo run --bin ferrite-daemon -- health
cargo run --bin ferrite-daemon -- run-list
cargo run --bin ferrite-daemon -- run-file internal/ptx-runner/src/main.rs -- --list
cargo run --bin ferrite-daemon -- run-entry internal/ptx-runner/src/main.rs#main -- --list
```

## Service-Oriented Entrypoint Scripts
```bash
cd ferrite-os
bash -n scripts/ptx_daemon.sh
cat scripts/ptx_daemon.service
```
