#!/usr/bin/env bash
# Source from the checkout root. First-time browser login needs a terminal (SSH: -t).
_remote_tailnet_run() {
    # Isolated Python excludes the checkout, so add it explicitly.
    command python3 -I -B -c '
import runpy
import sys

sys.path.insert(0, ".")
runpy.run_module("studio.remote_tailnet", run_name="__main__")
' "$@"
}

remote_tailnet_setup() {
    local selected
    selected=$(_remote_tailnet_run setup) || return "$?"
    export TRAINNER_GUI_TAILSCALE_CLI="${selected%%$'\n'*}"
    if [[ $selected == *$'\n'* ]]; then
        export TRAINNER_GUI_TAILSCALE_SOCKET="${selected#*$'\n'}"
    else
        unset TRAINNER_GUI_TAILSCALE_SOCKET
    fi
}

# --logout requests immediate removal of this checkout's ephemeral device.
remote_tailnet_cleanup() {
    local mode=cleanup closed
    if [[ $# == 1 && ${1:-} == --logout ]]; then
        mode=logout
    elif (( $# )); then
        printf '%s\n' 'Usage: remote_tailnet_cleanup [--logout]' >&2
        return 2
    fi
    closed=$(_remote_tailnet_run "$mode") || return "$?"
    if [[ -n $closed && $closed == "${TRAINNER_GUI_TAILSCALE_SOCKET:-}" ]]; then
        unset TRAINNER_GUI_TAILSCALE_CLI TRAINNER_GUI_TAILSCALE_SOCKET
    fi
}
