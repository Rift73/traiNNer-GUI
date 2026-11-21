#!/bin/bash
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x venv/bin/python ]]; then
    if [[ -e venv ]]; then
        echo "venv exists but has no Linux Python executable. Select the checkout for this platform." >&2
        exit 1
    fi
    python3.12 -m venv venv
fi
exec venv/bin/python scripts/install_runtime.py "$@"
