# Ferrite-OS Daemon - Quick Start

## Build

```bash
cd ~/weird_dif
cargo build --release -p ferrite-daemon
```

## Run in Development Mode

**Option 1: Using dev config (recommended)**

```bash
# Start daemon (uses /tmp, no root needed)
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH \
  ./target/release/ferrite-daemon serve \
  --config internal/ptx-daemon/dev-config.toml

# In another terminal, test commands:
./target/release/ferrite-daemon ping --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon status --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon metrics --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon health --socket /tmp/ferrite-daemon.sock
./target/release/ferrite-daemon watch --socket /tmp/ferrite-daemon.sock
```

**Option 2: Using launcher script**

```bash
# Start daemon
./ferrite-daemon.sh serve --config internal/ptx-daemon/dev-config.toml

# In another terminal:
./ferrite-daemon.sh ping --socket /tmp/ferrite-daemon.sock
```

**Option 3: Command line only**

```bash
# Start daemon with custom settings
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH \
  ./target/release/ferrite-daemon serve \
  --socket /tmp/ferrite.sock \
  --device 0 \
  --streams 32 \
  --pool-fraction 0.7 \
  --watch

# Test it
./target/release/ferrite-daemon ping --socket /tmp/ferrite.sock
```

## Example Session

**Terminal 1 - Start daemon:**
```bash
cd ~/weird_dif
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH \
  ./target/release/ferrite-daemon serve \
  --config internal/ptx-daemon/dev-config.toml
```

**Terminal 2 - Test commands:**
```bash
cd ~/weird_dif
export DAEMON="./target/release/ferrite-daemon --socket /tmp/ferrite-daemon.sock"

# Test connectivity
$DAEMON ping

# Get pool status
$DAEMON status

# Get runtime stats
$DAEMON stats

# Get comprehensive metrics
$DAEMON metrics

# Health check
$DAEMON health

# Live watch (Ctrl+C to stop)
$DAEMON watch

# Shutdown daemon
$DAEMON shutdown
```

## Production Installation

```bash
# Build
cargo build --release -p ferrite-daemon

# Install binary
sudo cp target/release/ferrite-daemon /usr/local/bin/

# Install config
sudo mkdir -p /etc/ferrite-os
sudo cp internal/ptx-daemon/ferrite-daemon.toml /etc/ferrite-os/daemon.toml
sudo nano /etc/ferrite-os/daemon.toml  # Edit as needed

# Install systemd service
sudo cp internal/ptx-daemon/ferrite-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload

# Create user and directories
sudo useradd -r -s /bin/false -G video,render ferrite
sudo mkdir -p /var/run/ferrite-os /var/log/ferrite-os
sudo chown ferrite:ferrite /var/run/ferrite-os /var/log/ferrite-os

# Start service
sudo systemctl start ferrite-daemon
sudo systemctl status ferrite-daemon

# Test
ferrite-daemon ping
```

## Troubleshooting

**"command not found"**
- Build first: `cargo build --release -p ferrite-daemon`
- Run from project root with `./target/release/ferrite-daemon`
- Or install to PATH: `sudo cp target/release/ferrite-daemon /usr/local/bin/`

**"libptx_os.so: cannot open shared object file"**
- Set LD_LIBRARY_PATH: `LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH`
- Or use launcher script: `./ferrite-daemon.sh`

**"Permission denied" for /var/run**
- Use dev config: `--config internal/ptx-daemon/dev-config.toml`
- Or specify temp socket: `--socket /tmp/daemon.sock`
- Or run with sudo (not recommended for dev)

**"Daemon already running"**
- Check PID: `cat /tmp/ferrite-daemon.pid`
- Kill: `kill $(cat /tmp/ferrite-daemon.pid)`
- Clean up: `rm /tmp/ferrite-daemon.{sock,pid}`

**"Connection refused"**
- Check daemon is running: `ps aux | grep ferrite-daemon`
- Check socket exists: `ls -la /tmp/ferrite-daemon.sock`
- Start daemon if not running

## Example Output

**Daemon starting:**
```
INFO Starting Ferrite-OS daemon
INFO Configuration: DaemonConfig { device_id: 0, ... }
INFO PID file created: /tmp/ferrite-daemon.pid
INFO Initializing GPU runtime on device 0
[Ferrite-OS] Initializing GPU Hot Runtime on device 0
[Ferrite-OS] Device: NVIDIA GeForce RTX 3070
[Ferrite-OS] Total VRAM: 7.68 GB
INFO Runtime initialized with 16 streams, pool=60%
INFO Daemon listening on /tmp/ferrite-daemon.sock
INFO Keepalive thread started (interval: 5000ms)
INFO Watch thread started (interval: 1000ms)
INFO Daemon ready
```

**Live watch mode:**
```
util=5.2% frag=0.0000% vram=4MB streams=16 clients=0
```

**Health check:**
```json
{
  "ok": true,
  "healthy": true,
  "uptime_secs": 120,
  "pool_healthy": true,
  "active_clients": 0,
  "max_clients": 32
}
```
