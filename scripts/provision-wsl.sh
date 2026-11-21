#!/bin/bash
# Run as root in the provisioned Ubuntu 24.04 WSL 2 distro.
set -Eeuo pipefail
umask 022
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
export LC_ALL=C.UTF-8
unset CDPATH

root=/opt/trainner-gui
manifest=.trainner-gui-payload.sha256
trainer=trainer
source_root=
mode=
offline=no
check=no
phase="checking the setup arguments"

usage() {
    echo "Usage: bash scripts/provision-wsl.sh [--source DIR] [--user trainer] [--cpu-only] [--offline | --check]"
    echo "CPU-only mode shares the installed wheels and disables CUDA in the workbench."
    echo "Run as Linux root from the extracted tarball or the installed managed checkout."
}

fail() {
    local code=$1
    shift
    printf 'traiNNer-GUI setup: %s\n' "$*" >&2
    exit "$code"
}

trap 'status=$?; fail 6 "Failed while $phase (exit $status). Rerun setup with the same bundle after correcting the error above."' ERR

while (( $# )); do
    case "$1" in
        --user)
            (( $# >= 2 )) || fail 2 "--user needs a Linux username."
            trainer=$2
            shift
            ;;
        --source)
            (( $# >= 2 )) || fail 2 "--source needs the extracted source directory."
            source_root=$2
            shift
            ;;
        --cpu-only) mode=cpu ;;
        --offline) offline=yes ;;
        --check) check=yes ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; fail 2 "Unknown argument $1." ;;
    esac
    shift
done
[[ $trainer =~ ^[a-z_][a-z0-9_-]{0,31}$ && $trainer != root ]] ||
    fail 2 "Select an ordinary Linux trainer account."
[[ $offline == no || $check == no ]] || fail 2 "Use --check or --offline."
(( EUID == 0 )) || fail 2 "Run this bootstrap as Linux root from Windows setup."

phase="checking WSL"
. /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 24.04 && $(uname -m) == x86_64 ]] ||
    fail 2 "Use the x86-64 Ubuntu 24.04 distro created by Windows setup."
grep -qi 'microsoft.*wsl2' /proc/sys/kernel/osrelease ||
    fail 2 "Use WSL 2. In Windows, check the distro version with wsl.exe --list --verbose."
[[ $(findmnt -n -o FSTYPE -T /opt) == ext4 ]] ||
    fail 2 "/opt must be on the distro's ext4 filesystem."
check_nvidia() {
    if [[ ! -c /dev/dxg || ! -r /usr/lib/wsl/lib/libcuda.so.1 ]] ||
        ! timeout -k 2s 12s /usr/lib/wsl/lib/nvidia-smi -L < /dev/null; then
        fail 4 "NVIDIA CUDA is unavailable in WSL. Update WSL and the NVIDIA Windows driver, then restart the distro. For workbench and CPU recipes, choose --cpu-only."
    fi
    capability=$(timeout -k 2s 12s /usr/lib/wsl/lib/nvidia-smi \
        --query-gpu=compute_cap --format=csv,noheader,nounits < /dev/null) || capability=
    capability=${capability%%$'\n'*}
    if [[ $capability =~ ^([0-9]+)\.[0-9]+$ ]] && (( BASH_REMATCH[1] < 8 )); then
        fail 4 "The first NVIDIA GPU has compute capability $capability. CUDA training requires Ampere or newer (SM 8.0+). For workbench and CPU recipes, choose --cpu-only."
    fi
}

if [[ $check == no && $mode != cpu ]]; then
    check_nvidia
fi

phase="checking the Linux payload"
if [[ -z $source_root ]]; then
    source_root=$(dirname -- "${BASH_SOURCE[0]}")/..
fi
source_root=$(realpath -e -- "$source_root") || fail 3 "The extracted source directory does not exist."
[[ -f $source_root/$manifest && ! -L $source_root/$manifest ]] ||
    fail 3 "The payload manifest is missing. Extract the source archive from the matching Windows bundle."
files=()
declare -A listed=()
while IFS= read -r line; do
    [[ $line =~ ^[a-f0-9]{64}\ \ ([A-Za-z0-9_.+/-]+)$ ]] ||
        fail 3 "Invalid entry in $manifest. Extract the matching source archive again."
    path=${BASH_REMATCH[1]}
    [[ $path != /* && /$path/ != *'/.'* && /$path/ != *'//'* &&
        $path != venv && $path != venv/* && $path != experiments && $path != experiments/* ]] ||
        fail 3 "Invalid installed path in $manifest."
    [[ ! -v 'listed[$path]' && -f $source_root/$path &&
        $(realpath -e -- "$source_root/$path") == "$source_root/$path" ]] ||
        fail 3 "Duplicate, missing or redirected payload file $path."
    listed[$path]=1
    files+=("$path")
done < "$source_root/$manifest"
for path in trainner-gui train.py pyproject.toml scripts/provision-wsl.sh \
    scripts/check-managed-wsl-runtime.py studio/desktop.py \
    requirements/managed-wsl-py312.txt; do
    [[ -v 'listed[$path]' ]] || fail 3 "Incomplete Linux payload. Missing $path."
done
(cd -- "$source_root"; sha256sum --check --strict --quiet -- "$manifest") ||
    fail 3 "The extracted source failed its checksum. Extract the matching archive again."
digest=$(sha256sum < "$source_root/$manifest")
payload_id=${digest%% *}
install_id="$payload_id $trainer"
files+=("$manifest")

phase="checking the managed installation identity"
if [[ $check == yes ]]; then
    [[ -d $root ]] || fail 3 "Managed setup has not run."
elif [[ ! -e $root && ! -L $root ]]; then
    mkdir -- "$root" || [[ -d $root ]]
fi
[[ -d $root && ! -L $root && $(stat -c '%u:%a' -- "$root") == 0:755 &&
    $(findmnt -n -o FSTYPE -T "$root") == ext4 ]] ||
    fail 3 "$root must be a root-owned 0755 directory on ext4."
exec 9< "$root"
flock -n 9 || fail 3 "Another setup is running in this distro. Wait for it to finish."
if [[ -f $root/.managed-id ]]; then
    [[ $(cat -- "$root/.managed-id") == "$install_id" ]] ||
        fail 3 "This distro belongs to a different bundle or trainer. Rerun the original bundle for this install or use a separate distro."
else
    [[ $check == no &&
        -z $(find "$root" -mindepth 1 -maxdepth 1 ! -name .managed-id.pending -print -quit) ]] ||
        fail 3 "$root does not identify a managed install. Select a dedicated setup distro."
    identity_file=$root/.managed-id.pending
    [[ ! -L $identity_file ]] || fail 3 "The pending installation identity is redirected."
    printf '%s\n' "$install_id" > "$identity_file"
    chmod 0644 -- "$identity_file"
    mv -T -- "$identity_file" "$root/.managed-id"
fi

phase="checking the trainer account"
if ! passwd_entry=$(getent passwd "$trainer"); then
    [[ $check == no && ! -f $root/.managed-ready ]] ||
        fail 3 "The Linux account $trainer is missing. Restore that account for this installation."
    useradd --create-home --user-group --shell /bin/bash -- "$trainer"
    passwd_entry=$(getent passwd "$trainer")
fi
IFS=: read -r _ _ trainer_uid trainer_gid _ trainer_home _ <<< "$passwd_entry"
[[ $trainer_uid =~ ^[0-9]+$ ]] && (( trainer_uid >= 1000 && trainer_uid < 65534 )) &&
    [[ $trainer_home == /* && $trainer_home != / && -d $trainer_home &&
        $(stat -c '%u' -- "$trainer_home") == "$trainer_uid" &&
        $(findmnt -n -o FSTYPE -T "$trainer_home") == ext4 ]] ||
    fail 2 "The trainer needs an ordinary Linux account with a user-owned home directory on ext4."

ready_id="$install_id $trainer_uid"
installed_mode=
if [[ -f $root/.managed-ready ]]; then
    read -r installed_id installed_user installed_uid installed_mode < "$root/.managed-ready"
    [[ "$installed_id $installed_user $installed_uid" == "$ready_id" &&
        ( $installed_mode == cuda || $installed_mode == cpu ) ]] ||
        fail 3 "The installed Linux trainer identity has changed. Restore the original account."
elif [[ $check == yes ]]; then
    fail 3 "Managed setup is incomplete. Rerun it with the same source bundle."
fi
if [[ $check == yes ]]; then
    [[ -z $mode || $mode == "$installed_mode" ]] ||
        fail 3 "The installed trainer is in $installed_mode mode. Run setup to change modes."
    mode=$installed_mode
    if [[ $mode == cuda ]]; then
        check_nvidia
    fi
else
    mode=${mode:-cuda}
fi

phase="copying the source and verifying its installed checksum"
if ! cmp -s -- "$source_root/$manifest" "$root/$manifest" ||
    ! (cd -- "$root"; sha256sum --check --strict --quiet -- "$manifest"); then
    [[ $check == no && -z $installed_mode && $source_root != "$root" ]] ||
        fail 3 "The installed source has changed. Inspect $root and retain the original bundle and workspace."
    # A previous copy may have stopped after writing part of a member.
    printf '%s\0' "${files[@]}" | tar -C "$source_root" --no-recursion --null -T - -cf - |
        tar -C "$root" --no-same-owner -xf -
    if ! cmp -s -- "$source_root/$manifest" "$root/$manifest" ||
        ! (cd -- "$root"; sha256sum --check --strict --quiet -- "$manifest"); then
        fail 3 "The source could not be copied intact."
    fi
fi

phase="installing Ubuntu build and media packages"
packages=(python3.12 python3.12-venv python3.12-dev ca-certificates build-essential
    clang make pkg-config cmake ninja-build ffmpeg libgl1 libgdk-pixbuf-2.0-dev
    libglib2.0-dev libtiff-dev libwebp-dev libpng-dev)
missing=()
for package in "${packages[@]}"; do
    if [[ $(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null) != installed ]]; then
        missing+=("$package")
    fi
done
if (( ${#missing[@]} )); then
    [[ $check == no ]] || fail 5 "Missing Ubuntu packages: ${missing[*]}. Rerun setup to restore them."
    export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a
    apt=(apt-get -o Dpkg::Use-Pty=0 -o DPkg::Lock::Timeout=30 -o Acquire::Retries=2
        -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
    if [[ $offline == yes ]]; then
        apt+=(--no-download)
    else
        "${apt[@]}" update < /dev/null ||
            fail 5 "Could not refresh Ubuntu package lists. Check Internet access and the Ubuntu 24.04 package sources."
    fi
    "${apt[@]}" install -y --no-install-recommends "${missing[@]}" < /dev/null ||
        fail 5 "Ubuntu packages could not be installed. Offline setup needs them installed or cached. If dpkg reports an interrupted transaction, run dpkg --configure -a in this dedicated distro, then retry."
fi

as_trainer() {
    runuser -u "$trainer" -- env -i HOME="$trainer_home" PATH="$PATH" LC_ALL="$LC_ALL" \
        bash -c 'set -e; source "$1/trainner-gui"; shift; cd -- "$managed_root"; exec "$@"' \
        -- "$root" "$@" < /dev/null
}

phase="installing the isolated Python runtime"
venv=$root/venv
python=$venv/bin/python
wheels=$root/.download/wheels
requirements=$root/requirements/managed-wsl-py312.txt
if [[ $check == no && ! -f $root/.python-ready ]]; then
    [[ -f $venv/pyvenv.cfg && -x $python ]] || /usr/bin/python3.12 -m venv "$venv"
    install -d -m 0700 -- "$root/.download" "$wheels"
    echo "Available space for the managed runtime and wheel cache:"
    df -h -- "$root"
    export PIP_CONFIG_FILE=/dev/null
    pip=("$python" -I -m pip --isolated --disable-pip-version-check --no-input)
    if ! "${pip[@]}" --version > /dev/null; then
        "$python" -I -m ensurepip --upgrade
    fi
    dependencies=$("$python" -I -c 'import sys,tomllib; print(*tomllib.load(open(sys.argv[1],"rb"))["project"]["dependencies"], sep="\n")' \
        "$root/pyproject.toml")
    mapfile -t dependencies <<< "$dependencies"
    requested=(-r "$requirements" "${dependencies[@]}")
    # A killed pip process may leave distribution metadata before every
    # file is installed. Keep the wheel cache and overlay the same packages.
    local_pip=(--no-index --find-links "$wheels" --only-binary=:all: --ignore-installed)
    if ! "${pip[@]}" install --no-cache-dir "${local_pip[@]}" "${requested[@]}"; then
        [[ $offline == no ]] ||
            fail 6 "The offline wheel cache is incomplete. Rerun setup online with the same bundle to populate $wheels."
        # pyvips 3.1.1 is published as an sdist. Cache its built wheel so
        # offline retries require only the wheelhouse and no build isolation.
        "${pip[@]}" --cache-dir "$root/.download/pip" --retries 2 --timeout 30 wheel \
            --index-url https://pypi.org/simple \
            --extra-index-url https://download.pytorch.org/whl/nightly/cu134 \
            --wheel-dir "$wheels" --find-links "$wheels" \
            --only-binary=:all: --no-binary=pyvips "${requested[@]}" ||
            fail 6 "The pinned packages could not be staged from PyPI and the CUDA 13.4 nightly index. Check connectivity and wheel availability. Cached downloads are retained."
        "${pip[@]}" install --no-cache-dir "${local_pip[@]}" "${requested[@]}" ||
            fail 6 "The staged wheels could not be installed. Inspect the Python package errors above."
    fi
    cuda_home=$venv/lib/python3.12/site-packages/nvidia/cu13
    [[ -x $cuda_home/bin/nvcc && -r $cuda_home/lib/libcudart.so.13 ]] ||
        fail 6 "The pinned NVIDIA compiler or CUDA runtime wheel is incomplete."
    # torch.utils.cpp_extension links with -lcudart, but the runtime wheel
    # supplies only the SONAME. Keep the linker name inside this venv.
    if [[ ! -e $cuda_home/lib/libcudart.so ]]; then
        ln -s -- libcudart.so.13 "$cuda_home/lib/libcudart.so"
    fi
    "$python" -I -m pip --disable-pip-version-check check
    as_trainer env CUDA_VISIBLE_DEVICES= timeout -k 5s 90s \
        "$python" -P "$root/scripts/check-managed-wsl-runtime.py" ||
        fail 6 "Python, torchvision or FFmpeg failed the runtime smoke check. Inspect the import or codec error above."
    touch -- "$root/.python-ready"
fi
[[ -x $python && -f $root/.python-ready ]] ||
    fail 6 "The managed Python environment is incomplete. Rerun setup with the original bundle."

phase="preparing the native tools"
if [[ $check == no && ( ! -f $root/.native-ready ||
    ! -x $root/native/studio/data/build/clang/inspect ||
    ! -r $root/traiNNer/csrc/webp_reconstruct/build/libwebp_reconstruct.so ) ]]; then
    install -d -o "$trainer_uid" -g "$trainer_gid" -m 0700 -- "$root/.cache" "$root/experiments"
    for tool in alignment data detection files history image_difference metrics \
        paired_patches provenance resample split_audit tfrecord weights; do
        install -d -o "$trainer_uid" -g "$trainer_gid" -m 0755 -- "$root/native/studio/$tool/build"
    done
    as_trainer timeout -k 5s 180s "$python" -P -m studio prepare ||
        fail 6 "The Studio tools could not be built. Inspect the compiler or development package error above."
    if [[ ! -f $root/traiNNer/csrc/webp_reconstruct/build/libwebp_reconstruct.so ]]; then
        timeout -k 5s 180s bash "$root/traiNNer/csrc/webp_reconstruct/build.sh" ||
            fail 6 "The bundled libwebp reconstruction tool could not be built."
    fi
    touch -- "$root/.native-ready"
fi
[[ -f $root/.native-ready &&
    -x $root/native/studio/data/build/clang/inspect &&
    -r $root/traiNNer/csrc/webp_reconstruct/build/libwebp_reconstruct.so ]] ||
    fail 6 "The native tools are incomplete. Check the build directories in the managed distro."

phase="validating the installed trainer"
if [[ $mode == cuda ]]; then
    as_trainer timeout -k 2s 12s "$venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc" --version ||
        fail 6 "The isolated NVIDIA CUDA compiler cannot run. Inspect the pinned runtime and compiler files."
    as_trainer timeout -k 5s 90s "$python" -P "$root/scripts/check-managed-wsl-runtime.py" --gpu ||
        fail 4 "The installed PyTorch or PTX kernels could not run on the first GPU. Update the NVIDIA Windows driver for CUDA 13.4 and restart the distro, or choose --cpu-only. The runtime and wheels are retained."
elif [[ $check == yes ]]; then
    as_trainer env CUDA_VISIBLE_DEVICES= timeout -k 5s 90s \
        "$python" -P "$root/scripts/check-managed-wsl-runtime.py" ||
        fail 6 "The installed CPU runtime failed its smoke check. Inspect the import or codec error above."
else
    echo "CPU-only setup enables the workbench and CPU recipes. A successful CUDA setup is required for GPU training."
fi
if [[ $check == no && $mode != "$installed_mode" ]]; then
    ready_file=$(mktemp -- "$root/.managed-ready.XXXXXXXX")
    printf '%s %s\n' "$ready_id" "$mode" > "$ready_file"
    chmod 0644 -- "$ready_file"
    mv -T -- "$ready_file" "$root/.managed-ready"
fi
printf 'TRAINNER_GUI_READY %s %s %s %s\n' "$payload_id" "$trainer" "$root" "$mode"
