#!/bin/bash
# Packaged as trainner-gui at the installed source root.
managed_root=$(dirname -- "$(realpath -- "${BASH_SOURCE[0]}")")
managed_python="$managed_root/venv/bin/python"
export PATH="$managed_root/venv/bin:/usr/bin:/bin:/usr/lib/wsl/lib"
export CUDA_HOME="$managed_root/venv/lib/python3.12/site-packages/nvidia/cu13"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib"
export CPATH="$CUDA_HOME/include/cccl"
unset PYTHONHOME
export PYTHONPATH="$managed_root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    if ! read -r _ trainer uid mode < "$managed_root/.managed-ready"; then
        echo "Managed traiNNer-GUI setup is incomplete. Rerun Windows setup." >&2
        exit 1
    fi
    if [[ $(id -u) != "$uid" || $EUID == 0 ]]; then
        echo "Launch managed traiNNer-GUI as the Linux user $trainer." >&2
        exit 1
    fi
    case "$mode" in
        cpu) export CUDA_VISIBLE_DEVICES= ;;
        cuda) ;;
        *) echo "Managed trainer mode is missing. Rerun Windows setup." >&2; exit 1 ;;
    esac
    export TRAINNER_GUI_RUNTIME_MODE="$mode"
    cd -- "$managed_root" || exit
    if [[ $# == 0 ]]; then set -- serve; fi
    exec "$managed_python" -P -m studio "$@"
fi
