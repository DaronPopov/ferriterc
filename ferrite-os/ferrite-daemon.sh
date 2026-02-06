#!/bin/bash
# Ferrite-OS Daemon Launcher
#
# This script sets up the environment and runs the daemon.

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set library path
export LD_LIBRARY_PATH="${SCRIPT_DIR}/ferrite-os/lib:${LD_LIBRARY_PATH}"

# Run daemon
exec "${SCRIPT_DIR}/ferrite-os/target/release/ferrite-daemon" "$@"
