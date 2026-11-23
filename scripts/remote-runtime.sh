#!/usr/bin/env bash
# Source in Bash from the checkout root. On success remote_runtime_setup
# exports the interpreter used by ./trainner-gui.

_remote_runtime_nvidia_visible() {
    case "${CUDA_VISIBLE_DEVICES-unset}" in ''|-1|-1,*|none|NoDevFiles) return 1 ;; esac
    case "${NVIDIA_VISIBLE_DEVICES-unset}" in ''|none|void) return 1 ;; esac
    # Use device files to recognize an allocation without calling CUDA or NVML.
    if [[ -c /dev/nvidiactl ]] && compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
        return 0
    fi
    [[ -c /dev/dxg && -r /usr/lib/wsl/lib/libcuda.so.1 ]]
}

_remote_runtime_probe() {
    CUDA_VISIBLE_DEVICES= "$1" -I -B - "$2" "$3" "$4" <<'PYTHON'
import sys
from pathlib import Path

if sys.platform != "linux" or sys.version_info < (3, 11):
    sys.exit("The trainer requires Linux Python 3.11 or newer.")
if sys.argv[2] and (sys.prefix == sys.base_prefix or
                    Path(sys.prefix).resolve() != Path(sys.argv[2]).resolve()):
    sys.exit("Python did not start in the selected virtual environment.")

sys.path.insert(0, sys.argv[1])
try:
    from traiNNer.check.check_dependencies import check_dependencies

    check_dependencies()
except (ImportError, RuntimeError) as error:
    sys.exit(str(error))

import torch

build = f"CUDA {torch.version.cuda}" if torch.version.cuda else (
    f"ROCm {torch.version.hip}" if torch.version.hip else "CPU-only"
)
if sys.argv[3] != "hidden" and not torch.version.cuda:
    print(f"{sys.executable} uses a {build} PyTorch build on a visible NVIDIA host.", file=sys.stderr)
    if sys.argv[3] == "auto":
        sys.exit(1)

import chainner_ext
import pyvips
import torchvision

torch.ops.torchvision.nms
pyvips.Image.black(1, 1)
# FFmpeg and NPP are needed by recipe-specific codec paths.
try:
    import cv2
except ImportError as exc:
    if "libGL.so.1:" not in str(exc):
        raise
    print(exc, file=sys.stderr)
    sys.exit(3)
print(f"Python {sys.version.split()[0]}, torch {torch.__version__} ({build}), torchvision {torchvision.__version__}")
PYTHON
}

_remote_runtime_missing_native() {
    local module
    local -A packages=(
        [gdk-pixbuf-2.0]=libgdk-pixbuf-2.0-dev [glib-2.0]=libglib2.0-dev
        [libtiff-4]=libtiff-dev [libwebp]=libwebp-dev [libpng]=libpng-dev
    )
    command -v make >/dev/null || printf 'make '
    command -v "${CXX:-c++}" >/dev/null || printf 'build-essential '
    if ! command -v "${CLANG:-/usr/bin/clang}" >/dev/null ||
       ! command -v "${CLANGXX:-/usr/bin/clang++}" >/dev/null; then
        printf 'clang '
    fi
    command -v pkg-config >/dev/null || printf 'pkg-config '
    for module in gdk-pixbuf-2.0 glib-2.0 libtiff-4 libwebp libpng; do
        pkg-config --exists "$module" 2>/dev/null || printf '%s ' "${packages[$module]}"
    done
    return 0
}

_remote_runtime_offer_apt() {
    (( $# )) || return 0
    local answer
    local -a apt=(apt-get)
    if ! command -v apt-get >/dev/null || ! grep -Eq '^ID="?ubuntu"?$' /etc/os-release; then
        printf 'Supply the equivalents of these Ubuntu packages and retry: %s\n' "$*" >&2
        return 1
    fi
    if (( EUID != 0 )); then apt=(sudo apt-get); fi
    printf 'Ubuntu prerequisites are missing. To install them, run\n  %s update && %s install --no-install-recommends --no-remove %s\n' \
        "${apt[*]}" "${apt[*]}" "$*" >&2
    if (( EUID != 0 )) && ! command -v sudo >/dev/null; then
        printf 'Ask the host administrator to install the packages.\n' >&2
        return 1
    fi
    if [[ ! -t 0 ]] || ! read -r -p 'Install the listed Ubuntu packages now? [y/N] ' answer ||
        [[ ! $answer =~ ^([yY]|[yY][eE][sS])$ ]]; then
        return 1
    fi
    "${apt[@]}" update </dev/null >&2 &&
        "${apt[@]}" install -y --no-install-recommends --no-remove "$@" </dev/null >&2
}

_remote_runtime_select() {
    local root=$1 venv=$1/venv marker=$1/venv/.remote-runtime-pending
    local python= candidate expected provider native status=0 lock_fd owned=no nvidia=hidden
    local -a candidates missing
    # The command substitution owns the directory lock until setup returns.
    if ! command -v flock >/dev/null; then
        printf 'Install util-linux for the flock command, then retry.\n' >&2
        return 1
    fi
    exec {lock_fd}<"$root" || return 1
    if ! flock -n "$lock_fd"; then
        printf 'Runtime setup is already active, or this filesystem cannot lock the checkout. Retry on a local Linux filesystem.\n' >&2
        return 1
    fi

    if _remote_runtime_nvidia_visible; then
        nvidia=auto
        [[ -z ${TRAINNER_GUI_PYTHON:-} ]] || nvidia=explicit
    fi
    if [[ -n ${TRAINNER_GUI_PYTHON:-} ]]; then
        candidate=$TRAINNER_GUI_PYTHON
        if [[ $candidate != */* ]] && ! candidate=$(type -P -- "$candidate"); then
            printf 'TRAINNER_GUI_PYTHON does not name an available executable.\n' >&2
            return 1
        fi
        candidates=("$candidate")
    else
        candidates=("$venv/bin/python" "$root/.venv/bin/python")
        if provider=$(type -P python3) &&
           [[ $provider != "$venv"/bin/* && $provider != "$root"/.venv/bin/* ]]; then
            candidates+=("$provider")
        fi
    fi
    for candidate in "${candidates[@]}"; do
        [[ -f $candidate && -x $candidate ]] || continue
        expected=
        case "$candidate" in
            "$venv/bin/python"|"$root/.venv/bin/python")
                expected=${candidate%/bin/python}
                [[ -f $expected/pyvenv.cfg ]] || continue
                ;;
        esac
        status=0
        _remote_runtime_probe "$candidate" "$root" "$expected" "$nvidia" >&2 || status=$?
        if (( status == 0 || status == 3 )); then
            python=$candidate
            break
        fi
    done
    if [[ -n ${TRAINNER_GUI_PYTHON:-} && -z $python ]]; then
        printf 'Choose a compatible prepared TRAINNER_GUI_PYTHON or unset it to discover the checkout runtime.\n' >&2
        return 1
    fi

    if [[ -d $venv && ! -L $venv && -O $venv && -f $marker && ! -L $marker &&
          -O $marker && $(<"$marker") == "$root" ]]; then
        owned=yes
    fi
    native=$(_remote_runtime_missing_native) || return 1
    read -r -a missing <<< "$native"
    if [[ -z $python ]]; then
        if [[ $owned == no && ( -e $venv || -L $venv ) ]]; then
            printf 'The existing %q did not pass the runtime check. Select a prepared interpreter or move that directory aside.\n' "$venv" >&2
            return 1
        fi
        if [[ $(uname -m) != x86_64 ]]; then
            printf 'The reference install targets x86-64 Linux. Select a prepared interpreter with TRAINNER_GUI_PYTHON.\n' >&2
            return 1
        fi
        if [[ ! -x $venv/bin/python ]] &&
           ! python3.12 -I -B -c 'import ensurepip, sys, venv; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
            if ! grep -Eq '^VERSION_ID="?24\.04"?$' /etc/os-release &&
               ! apt-cache show python3.12-venv >/dev/null 2>&1; then
                printf 'Supply Python 3.12 with venv support or set TRAINNER_GUI_PYTHON to a prepared interpreter.\n' >&2
                return 1
            fi
            missing+=(python3.12-venv)
        fi
        if ! /sbin/ldconfig -p 2>/dev/null | grep -F 'libGL.so.1 ' >/dev/null &&
           ! python3 -I -B -c 'from ctypes import CDLL; CDLL("libGL.so.1")' 2>/dev/null; then
            missing+=(libgl1)
        fi
    elif (( status == 3 )); then
        missing+=(libgl1)
    fi
    _remote_runtime_offer_apt "${missing[@]}" || return 1

    if [[ -z $python ]]; then
        python=$venv/bin/python
        local -a clean=(env -u PYTHONPATH -u PYTHONHOME -u PIP_TARGET -u PIP_PREFIX -u PIP_ROOT -u PIP_PYTHON
            PIP_CONFIG_FILE=/dev/null PIP_USER=0 PIP_REQUIRE_VIRTUALENV=1 PIP_NO_INPUT=1
            PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1)
        if [[ $owned == no ]]; then
            mkdir -- "$venv" && printf '%s\n' "$root" >"$marker" || return 1
            owned=yes
        fi
        if [[ -x $python ]]; then
            "$python" -I -B -c '
import sys
from pathlib import Path
from sysconfig import get_path

venv = Path(sys.argv[1])
if sys.version_info[:2] != (3, 12) or sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != venv:
    sys.exit("The pending install needs its original isolated Python 3.12 venv.")
if any(not Path(get_path(kind)).resolve().is_relative_to(venv) for kind in ("purelib", "platlib", "scripts")):
    sys.exit("The pending venv has install directories outside the checkout venv.")
' "$venv" || return 1
            if ! "$python" -I -B -c 'import pip' 2>/dev/null; then
                "${clean[@]}" "$python" -I -m ensurepip </dev/null >&2 || return 1
            fi
        else
            "${clean[@]}" python3.12 -m venv "$venv" </dev/null >&2 || return 1
        fi
        printf 'Installing the reference Python packages in %s.\n' "$venv" >&2
        "${clean[@]}" bash "$root/install.sh" </dev/null >&2 || return 1
        _remote_runtime_probe "$python" "$root" "$venv" "$nvidia" >&2 || return 1
    elif (( status == 3 )); then
        "$python" -I -B -c 'import cv2' >&2 || return 1
    fi

    [[ $python == /* ]] || python=$root/$python
    if [[ $owned == yes && $python == "$venv/bin/python" ]]; then
        rm -- "$marker" || return 1
    fi
    printf '%s\n' "$python"
}

remote_runtime_setup() {
    local root python
    root=$(pwd -P) || return 1
    if [[ $(uname -s) != Linux || ! -r $root/install.sh || ! -r $root/pyproject.toml ||
          ! -r $root/scripts/install_runtime.py ]]; then
        printf 'Run remote_runtime_setup in the traiNNer-GUI checkout root on Linux.\n' >&2
        return 1
    fi
    python=$(_remote_runtime_select "$root") || return 1
    export TRAINNER_GUI_PYTHON="$python"
    printf 'Launch with TRAINNER_GUI_PYTHON=%q.\n' "$TRAINNER_GUI_PYTHON" >&2
}
