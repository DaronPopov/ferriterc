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
```

## Service-Oriented Entrypoint Scripts
```bash
cd ferrite-os
bash -n scripts/ptx_daemon.sh
cat scripts/ptx_daemon.service
```
