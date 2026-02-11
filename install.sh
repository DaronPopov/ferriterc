#!/usr/bin/env bash
# Preserved entrypoint — delegates to modular installer.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/install/install.sh" "$@"
