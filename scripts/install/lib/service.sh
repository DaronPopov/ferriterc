# service.sh — systemd unit installation
# Sourced by scripts/install/install.sh

install_systemd_service() {
  local daemon_bin="$ROOT/ferrite-os/target/release/ferrite-daemon"
  local cfg_src="$ROOT/ferrite-os/crates/internal/ptx-daemon/ferrite-daemon.toml"
  local cfg_dst="/etc/ferrite-os/daemon.toml"
  local unit_dst="/etc/systemd/system/ferrite-daemon.service"

  if ! cmd_exists systemctl; then
    echo "[error] --enable-service requested but systemctl was not found"
    diag_emit "installer.service" "FAIL" "INS-SVC-0001" "systemctl not found for --enable-service" "install systemd or run without --enable-service"
    exit 1
  fi
  if [[ ! -f "$daemon_bin" ]]; then
    echo "[error] ferrite-daemon binary not found: $daemon_bin"
    diag_emit "installer.service" "FAIL" "INS-SVC-0002" "ferrite-daemon binary missing" "build ferrite-os daemon before enabling service"
    exit 1
  fi
  if [[ ! -f "$cfg_src" ]]; then
    echo "[error] daemon config template not found: $cfg_src"
    diag_emit "installer.service" "FAIL" "INS-SVC-0003" "daemon config template missing" "restore ferrite-daemon.toml template"
    exit 1
  fi

  echo "[info] installing ferrite-daemon systemd service"
  sudo_cmd install -d -m 0755 /etc/ferrite-os
  sudo_cmd install -d -m 0755 /var/run/ferrite-os
  sudo_cmd install -d -m 0755 /var/log/ferrite-os
  sudo_cmd cp "$cfg_src" "$cfg_dst"

  local libtorch_env=""
  local ld_path
  if [[ "${CORE_ONLY}" != "true" && -n "${LIBTORCH:-}" ]]; then
    libtorch_env="Environment=LIBTORCH=${LIBTORCH}
Environment=LIBTORCH_BYPASS_VERSION_CHECK=1"
    ld_path="${ROOT}/ferrite-os/lib:${LIBTORCH}/lib"
  else
    ld_path="${ROOT}/ferrite-os/lib"
  fi

  local unit_tmp
  unit_tmp="$(mktemp -t ferrite-daemon-service.XXXXXX)"
  cat > "$unit_tmp" <<EOF
[Unit]
Description=Ferrite-OS GPU Runtime Daemon
After=network.target

[Service]
Type=simple
WorkingDirectory=${ROOT}/ferrite-os
Environment=CUDA_PATH=${CUDA_PATH}
Environment=CUDA_HOME=${CUDA_PATH}
Environment=SM=${SM}
Environment=GPU_SM=${SM}
Environment=CUDA_SM=${SM}
Environment=PTX_GPU_SM=sm_${SM}
Environment=CUDA_ARCH=sm_${SM}
Environment=CUDARC_CUDA_FEATURE=${CUDARC_CUDA_FEATURE:-}
${libtorch_env}
Environment=LD_LIBRARY_PATH=${ld_path}
ExecStart=${daemon_bin} serve --config ${cfg_dst}
Restart=on-failure
RestartSec=3
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ferrite-daemon

[Install]
WantedBy=multi-user.target
EOF

  sudo_cmd install -m 0644 "$unit_tmp" "$unit_dst"
  rm -f "$unit_tmp"

  sudo_cmd systemctl daemon-reload
  sudo_cmd systemctl enable ferrite-daemon.service
  sudo_cmd systemctl restart ferrite-daemon.service
  echo "[info] ferrite-daemon service enabled and started"
  diag_emit "installer.service" "PASS" "INS-SVC-0004" "ferrite-daemon service enabled and started" "none"
}
