# ferrite-daemon

Production-grade daemon for Ferrite-OS GPU runtime system.

## Overview

ferrite-daemon is a persistent system service that manages the Ferrite-OS GPU runtime, providing:

- **Signal Handling**: Graceful shutdown on SIGTERM/SIGINT, config reload on SIGHUP
- **Structured Logging**: Tracing integration with configurable output
- **Concurrent Clients**: Thread-per-client model with connection limits
- **Rate Limiting**: Maximum concurrent client enforcement
- **PID Management**: Automatic PID file creation and cleanup
- **Health Monitoring**: Built-in health checks and metrics
- **Configuration**: TOML configuration file and environment variable support
- **Systemd Integration**: Native systemd service with security hardening

## Features

### Production Hardening

- **Signal Handling**: Proper SIGTERM/SIGINT/SIGHUP handling
- **PID File Management**: Prevents duplicate daemon instances
- **Resource Limits**: Configurable connection limits
- **Graceful Shutdown**: Clean resource cleanup on exit
- **Client Timeouts**: Prevents hanging connections
- **Thread Safety**: Concurrent request handling

### Observability

- **Structured Logging**: JSON-formatted logs with tracing
- **Metrics Export**: Comprehensive runtime metrics
- **Health Checks**: Health endpoint for monitoring
- **Live Watch Mode**: Real-time metrics display

### TUI Visual System

- **Visual Design Guide**: `crates/internal/ptx-daemon/TUI_VISUAL_DESIGN_GUIDE.md`
- **Style Review Checklist**: `crates/internal/ptx-daemon/TUI_STYLE_REVIEW_CHECKLIST.md`

### Security

- **Unix Socket Communication**: Local-only IPC
- **Permission-Based Access**: File system permissions control access
- **No Network Exposure**: Unix sockets only, no TCP/IP
- **Systemd Security**: Sandboxing and privilege restrictions

## Installation

### Build

```bash
cargo build --release -p ferrite-daemon
```

### Install Binary

```bash
# System-wide installation
sudo cp target/release/ferrite-daemon /usr/local/bin/

# Or user installation
mkdir -p ~/.local/bin
cp target/release/ferrite-daemon ~/.local/bin/
```

### Install Configuration

```bash
# System-wide
sudo mkdir -p /etc/ferrite-os
sudo cp crates/internal/ptx-daemon/ferrite-daemon.toml /etc/ferrite-os/daemon.toml
sudo chmod 644 /etc/ferrite-os/daemon.toml

# Or user-specific
mkdir -p ~/.config/ferrite-os
cp crates/internal/ptx-daemon/ferrite-daemon.toml ~/.config/ferrite-os/daemon.toml
```

### Install Systemd Service

```bash
# From repository root, generate + install service using installer flow
./install.sh --enable-service
```

## Usage

### Server Mode

Start the daemon:

```bash
# Foreground
ferrite-daemon serve

# Background (systemd)
sudo systemctl start ferrite-daemon

# With configuration file
ferrite-daemon serve --config /etc/ferrite-os/daemon.toml

# With command-line options
ferrite-daemon serve \
  --device 0 \
  --streams 64 \
  --pool-fraction 0.7 \
  --watch
```

### Client Commands

#### Connectivity Test

```bash
ferrite-daemon ping
```

Output:
```json
{"ok":true,"message":"pong"}
```

#### Pool Status

```bash
ferrite-daemon status
```

Output:
```json
{
  "ok": true,
  "pool_total": 4324990976,
  "allocated": 0,
  "free": 4324990976,
  "utilization": 0.0,
  "fragmentation": 0.0,
  "healthy": true
}
```

#### Runtime Statistics

```bash
ferrite-daemon stats
```

Output:
```json
{
  "ok": true,
  "vram_allocated": 8192,
  "vram_used": 4096,
  "vram_free": 4096,
  "gpu_utilization": 45.2,
  "active_streams": 16,
  "registered_kernels": 0,
  "total_ops": 0
}
```

#### Comprehensive Metrics

```bash
ferrite-daemon metrics
```

Output:
```json
{
  "ok": true,
  "uptime_secs": 3600,
  "active_clients": 2,
  "total_requests": 150,
  "failed_requests": 3,
  "vram_allocated": 8192,
  "vram_used": 4096,
  "vram_free": 4096,
  "gpu_util": 45.2,
  "active_streams": 16,
  "total_ops": 12500,
  "pool_total": 4324990976,
  "pool_allocated": 1024000,
  "pool_free": 4323966976,
  "pool_util": 0.024,
  "fragmentation": 0.0
}
```

#### Health Check

```bash
ferrite-daemon health
```

Output:
```json
{
  "ok": true,
  "healthy": true,
  "uptime_secs": 3600,
  "pool_healthy": true,
  "active_clients": 2,
  "max_clients": 32
}
```

#### System Snapshot

```bash
ferrite-daemon snapshot
```

Output:
```json
{
  "ok": true,
  "total_ops": 12500,
  "active_processes": 3,
  "active_tasks": 5,
  "vram_used": 4096,
  "watchdog_alert": false,
  "kernel_running": false,
  "shutdown_requested": false,
  "queue_head": 0,
  "queue_tail": 0,
  "completion_head": 0,
  "completion_tail": 0,
  "completion_overruns": 0
}
```

#### Keepalive

```bash
ferrite-daemon keepalive
```

Output:
```json
{"ok":true,"message":"keepalive sent"}
```

#### Task Submit (ABI v1)

```bash
ferrite-daemon task-submit-v1 [opcode] [priority] [tenant_id] [flags] [depends_on_task_id] [continuation_opcode]
```

Output:
```json
{
  "ok": true,
  "task_id": 1,
  "opcode": 0,
  "priority": 1,
  "tenant_id": 7,
  "flags": 0,
  "depends_on_task_id": null,
  "continuation_opcode": null
}
```

Example DAG edge with continuation:
```bash
# Submit B that waits for task 10 and enqueues opcode 0 continuation on completion.
ferrite-daemon task-submit-v1 0 1 7 0 10 0
```

#### Cooperative Time-Sliced Task

```bash
ferrite-daemon task-submit-coop 42 20000 64 1
```

Output:
```json
{
  "ok": true,
  "task_id": 12,
  "opcode": 9,
  "tenant_id": 42,
  "priority": 1,
  "work_units": 20000,
  "quantum": 64
}
```

#### Cooperative Batch Submit

```bash
ferrite-daemon task-submit-coop-batch 42 24 256 256 1
```

Output:
```json
{
  "ok": true,
  "requested": 24,
  "submitted": 24,
  "first_task_id": 100,
  "last_task_id": 123,
  "tenant_id": 42,
  "priority": 1,
  "work_units": 256,
  "quantum": 256
}
```

#### ISA v0 Task Submit

```bash
ferrite-daemon task-submit-isa-v0 42 halt 64 1
```

Modes:

- `halt`: execute `HALT` immediately
- `trap`: execute `TRAP` and complete with runtime error
- `yield`: execute `YIELD`, then resume and halt
- `movi`: execute `MOVI` + `ASSERT_EQI` + `HALT`
- `arith`: execute `ADD` + `ASSERT_EQI` + `HALT`
- `branch`: execute `BR_EQ` to skip `TRAP`, then `HALT`
- `jmp`: execute `JMP` to skip `TRAP`, then `HALT`
- `pc_oob`: execute out-of-bounds `JMP` and trap with runtime error
- `mem_ld`: execute `LD_U32` + `ASSERT_EQI` + `HALT`
- `mem_oob`: execute out-of-bounds `LD_U32` and trap with runtime error
- `sys_yield`: execute syscall `SYS_YIELD`, then resume and halt
- `sys_bad`: execute invalid syscall id and trap with runtime error

Output:
```json
{
  "ok": true,
  "task_id": 200,
  "opcode": 10,
  "tenant_id": 42,
  "priority": 1,
  "slice_steps": 64,
  "mode": "halt",
  "code_words": 1
}
```

#### Task Poll (ABI v1)

```bash
ferrite-daemon task-poll-v1
```

Output (when completion is available):
```json
{
  "ok": true,
  "empty": false,
  "result": {
    "abi_version": 1,
    "task_id": 1,
    "opcode": 0,
    "priority": 1,
    "tenant_id": 7,
    "status": 0
  }
}
```

#### Live Watch Mode

```bash
ferrite-daemon watch
```

Output (updates in real-time):
```
util=24.5% frag=0.0000% vram=4MB streams=16 clients=2
```

#### Shutdown

```bash
ferrite-daemon shutdown
```

Output:
```json
{"ok":true,"message":"shutting down"}
```

#### Help

```bash
ferrite-daemon help
```

## Configuration

### Configuration File

TOML format at `/etc/ferrite-os/daemon.toml` or `~/.config/ferrite-os/daemon.toml`:

```toml
device_id = 0
socket_path = "/var/run/ferrite-os/daemon.sock"
pid_file = "/var/run/ferrite-os/daemon.pid"
max_clients = 32
max_streams = 128
pool_fraction = 0.25
keepalive_ms = 5000
watch_ms = 1000
watch_enabled = false
enable_leak_detection = false
boot_kernel = false
log_dir = "/var/log/ferrite-os"
client_timeout_secs = 30
```

### Environment Variables

- `FERRITE_DEVICE`: GPU device ID
- `FERRITE_SOCKET`: Socket path
- `FERRITE_MAX_STREAMS`: Maximum streams
- `FERRITE_BOOT_KERNEL`: Boot persistent kernel
- `FERRITE_WATCH`: Enable watch mode

### Command-Line Options

```
OPTIONS:
    --config FILE         Load configuration from file
    --socket PATH         Unix socket path
    --device N            GPU device ID
    --streams N           Maximum streams
    --pool-fraction F     VRAM pool fraction (0.0-1.0)
    --boot-kernel         Boot persistent kernel
    --watch               Enable watch mode
    --watch-ms N          Watch interval (milliseconds)
    --log-dir DIR         Log directory
```

## Monitoring

### Systemd Integration

```bash
# Status
sudo systemctl status ferrite-daemon

# Logs
sudo journalctl -u ferrite-daemon -f

# Restart
sudo systemctl restart ferrite-daemon

# Stop
sudo systemctl stop ferrite-daemon
```

### Health Monitoring

Use the health endpoint in monitoring systems:

```bash
#!/bin/bash
# health-check.sh

STATUS=$(ferrite-daemon health | jq -r .healthy)
if [ "$STATUS" != "true" ]; then
  echo "Daemon unhealthy"
  exit 1
fi
echo "Daemon healthy"
exit 0
```

### Prometheus Integration

Export metrics in Prometheus format:

```bash
#!/bin/bash
# metrics-exporter.sh

METRICS=$(ferrite-daemon metrics)
echo "# HELP ferrite_uptime_seconds Daemon uptime"
echo "# TYPE ferrite_uptime_seconds gauge"
echo "ferrite_uptime_seconds $(echo $METRICS | jq .uptime_secs)"

echo "# HELP ferrite_pool_utilization Pool utilization percentage"
echo "# TYPE ferrite_pool_utilization gauge"
echo "ferrite_pool_utilization $(echo $METRICS | jq .pool_util)"

# Add more metrics...
```

## Troubleshooting

### Daemon Won't Start

**Check if already running**:
```bash
ps aux | grep ferrite-daemon
```

**Check socket**:
```bash
ls -la /var/run/ferrite-os/daemon.sock
```

**Check PID file**:
```bash
cat /var/run/ferrite-os/daemon.pid
```

**Remove stale files**:
```bash
sudo rm /var/run/ferrite-os/daemon.sock
sudo rm /var/run/ferrite-os/daemon.pid
```

### Permission Denied

**Check user permissions**:
```bash
groups $USER
# Should include 'video' and 'render' groups
```

**Add user to groups**:
```bash
sudo usermod -aG video,render $USER
# Logout and login again
```

### Connection Refused

**Check socket path**:
```bash
ferrite-daemon ping --socket /var/run/ferrite-os/daemon.sock
```

**Check daemon is running**:
```bash
sudo systemctl status ferrite-daemon
```

### High Memory Usage

**Check pool configuration**:
```toml
pool_fraction = 0.6  # Reduce if too high
```

**Monitor pool usage**:
```bash
ferrite-daemon status
```

### GPU Not Found

**Check CUDA availability**:
```bash
nvidia-smi
```

**Check device ID**:
```toml
device_id = 0  # Ensure correct device
```

## Security Considerations

### Access Control

The daemon uses Unix socket permissions for access control:

```bash
# Restrict socket to specific group
sudo chown root:ferrite-users /var/run/ferrite-os/daemon.sock
sudo chmod 660 /var/run/ferrite-os/daemon.sock
```

### Systemd Sandboxing

The systemd service includes security hardening:

- `NoNewPrivileges`: Prevents privilege escalation
- `PrivateTmp`: Private /tmp directory
- `ProtectSystem=strict`: Read-only filesystem
- `ProtectHome`: No access to home directories
- `RestrictAddressFamilies`: Limited network access
- `LimitNOFILE`: File descriptor limits

### Best Practices

1. **Run as dedicated user**: Use `ferrite` user, not root
2. **Limit socket permissions**: Only allow required users/groups
3. **Monitor logs**: Watch for unusual activity
4. **Regular updates**: Keep daemon and dependencies updated
5. **Resource limits**: Configure appropriate client limits

## Performance Tuning

### Stream Count

Balance parallelism vs overhead:

```toml
# Low parallelism (< 100 operations)
max_streams = 16

# Medium parallelism (100-1000 operations)
max_streams = 64

# High parallelism (1000+ operations)
max_streams = 1024
```

### Pool Fraction

Adjust based on workload:

```toml
# Conservative (shared GPU)
pool_fraction = 0.5

# Balanced (dedicated GPU)
pool_fraction = 0.7

# Aggressive (GPU exclusively for Ferrite-OS)
pool_fraction = 0.9
```

### Client Limits

Prevent resource exhaustion:

```toml
# Conservative
max_clients = 16

# Standard
max_clients = 32

# High concurrency
max_clients = 64
```

## Development

### Building

```bash
cargo build -p ferrite-daemon
```

### Testing

```bash
# Start test daemon
cargo run -p ferrite-daemon -- serve --socket /tmp/test.sock

# Test commands
cargo run -p ferrite-daemon -- ping --socket /tmp/test.sock
cargo run -p ferrite-daemon -- status --socket /tmp/test.sock
```

### Debugging

Enable debug logging:

```bash
RUST_LOG=debug cargo run -p ferrite-daemon -- serve
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../../../LICENSE-MIT))

at your option.

## See Also

- [Ferrite-OS Main Documentation](../../../README.md)
- [Runtime Architecture Guide](../../../../docs/02-runtime-architecture/README.md)
- [ptx-runtime Documentation](../../public/ptx-runtime/README.md)
