#!/usr/bin/env bash
set -euo pipefail

echo "This is a source distribution; uninstall is manual."
echo "Remove this folder to uninstall:"
echo "  rm -rf $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
