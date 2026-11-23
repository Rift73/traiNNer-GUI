#!/usr/bin/env bash
set -euo pipefail
root=$(dirname -- "$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")")
cd -- "$root"

usage() {
    echo "Usage: ./trainner-gui remote [off]"
    echo "Set up this Linux host and print private desktop and phone links."
    echo "Use off to close remote access and sign out a checkout-owned rental node."
}
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    ""|off) ;;
    *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then usage >&2; exit 2; fi

source "$root/scripts/remote-tailnet.sh"
if [[ ${1:-} == off ]]; then
    status=0
    ./trainner-gui tailscale off || status=$?
    remote_tailnet_cleanup --logout
    exit "$status"
fi
source "$root/scripts/remote-runtime.sh"

remote_tailnet_setup
# Reconnect to a running workspace before probing training libraries or invoking pip.
if ./trainner-gui tailscale status --links 2>/dev/null; then
    exit 0
fi
remote_runtime_setup
exec ./trainner-gui tailscale on --links
