# Runbook: Debugging and Remediation

## Doctor Check
```bash
cd ferrite-os
scripts/ptx_doctor.sh
```

## If `libptx_os.so` Is Missing
```bash
cd ferrite-os
make all
ls -l lib/libptx_os.so
```

## If Compatibility Selection Looks Wrong
```bash
./scripts/resolve_cuda_compat.sh --format env
cat compat.toml
```

## If Daemon Service Fails
```bash
systemctl status ferrite-daemon
journalctl -u ferrite-daemon -n 200
```

## If Script Execution Fails
```bash
./ferrite-run --help
cd ferrite-gpu-lang && cargo check
```
