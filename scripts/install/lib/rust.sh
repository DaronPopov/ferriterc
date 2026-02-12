# rust.sh — Rust toolchain bootstrap
# Sourced by scripts/install/install.sh

ensure_rust_toolchain() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  echo "[info] cargo not found; installing rust toolchain via rustup"
  ensure_fetch_tools
  need_cmd curl
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  need_cmd cargo
}
