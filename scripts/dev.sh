#!/usr/bin/env bash
# Usage: scripts/dev.sh <command...>
# Sets VCPKG_ROOT when it is not already set, then runs the command.
set -euo pipefail

if [[ -z "${VCPKG_ROOT:-}" ]]; then
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    candidates=(
        "$HOME/vcpkg"
        /opt/vcpkg
        /usr/local/vcpkg
        "$root/vcpkg"
    )
    for dir in "${candidates[@]}"; do
        if [[ -x "$dir/vcpkg" ]]; then
            export VCPKG_ROOT=$dir
            break
        fi
    done
fi
if [[ -z "${VCPKG_ROOT:-}" ]]; then
    echo "Warning: VCPKG_ROOT was not found. The CMake presets need it." >&2
fi
if [[ $# -eq 0 ]]; then
    exit 0
fi
exec "$@"
