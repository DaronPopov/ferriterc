# Ferrite-OS Daemon - Quick Start

## Build

```bash
# From repository root
cd /path/to/ferriterc/ferrite-os
cargo build --release -p ferrite-daemon
```

## Run in Development Mode

### Option 1: Direct binary + dev config (recommended)

```bash
cd /path/to/ferriterc/ferrite-os
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH \
  ./target/release/ferrite-daemon serve \
  --config internal/ptx-daemon/dev-config.toml
```

In another terminal:

```bash
cd /path/to/ferriterc/ferrite-os
./target/release/ferrite-daemon ping --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon status --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon metrics --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon health --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon watch --socket /tmp/ferrite-daemon.sock
```

### Option 2: Use launcher wrapper

```bash
cd /path/to/ferriterc/ferrite-os
./ferrite-daemon.sh serve --config internal/ptx-daemon/dev-config.toml

# In another terminal
./ferrite-daemon.sh ping --socket /tmp/ferrite-daemon.sock
```

## Example Session

Terminal 1:

```bash
cd /path/to/ferriterc/ferrite-os
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH \
  ./target/release/ferrite-daemon serve \
  --config internal/ptx-daemon/dev-config.toml
```

Terminal 2:

```bash
cd /path/to/ferriterc/ferrite-os
export DAEMON="./target/release/ferrite-daemon --socket /tmp/ferrite-daemon.sock"

$DAEMON ping
$DAEMON status
$DAEMON stats
$DAEMON metrics
$DAEMON health
$DAEMON watch
$DAEMON shutdown
```

## Production Service Installation

Use the installer-managed systemd flow:

```bash
cd /path/to/ferriterc
./install.sh --enable-service
```

This generates and installs `/etc/systemd/system/ferrite-daemon.service` from
`scripts/install/lib/service.sh`.

## Troubleshooting

- `command not found`
  - Build first: `cargo build --release -p ferrite-daemon`
  - Run binary directly: `./target/release/ferrite-daemon`
- `libptx_os.so: cannot open shared object file`
  - Run with `LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH`
  - Or use `./ferrite-daemon.sh`
- `Permission denied` for `/var/run`
  - Use dev config: `--config internal/ptx-daemon/dev-config.toml`
  - Or use a `/tmp` socket with `--socket /tmp/ferrite-daemon.sock`
- `Daemon already running`
  - Check PID file and socket in `/tmp` (dev) or `/var/run/ferrite-os` (service)
