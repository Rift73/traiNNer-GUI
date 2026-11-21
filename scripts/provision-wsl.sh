#!/bin/bash
# Run as root in the provisioned Ubuntu 24.04 WSL 2 distro.
set -Eeuo pipefail
umask 022
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
export LC_ALL=C.UTF-8
unset CDPATH

legacy_root=/opt/trainner-gui
release_parent=/opt/trainner-gui-releases
root=$legacy_root
upgrade_from=
archive_sha256=
reconfigure=no
probe=no
probe_state_args=()
manifest=.trainner-gui-payload.sha256
python_requirements=requirements/managed-wsl-py312.txt
python_inputs=(pyproject.toml "$python_requirements" requirements/live-runtime.txt requirements/live-wsl-py312.txt)
trainer=trainer
source_root=
mode=
offline=no
check=no
phase="checking the setup arguments"

usage() {
    echo "Usage: bash scripts/provision-wsl.sh [--source DIR] [--user trainer] [--cpu-only] [--offline | --check]"
    echo "       bash scripts/provision-wsl.sh --upgrade-from ROOT --release SHA256 [--source DIR] [--user trainer] [--offline]"
    echo "       bash scripts/provision-wsl.sh --upgrade-from ROOT --release SHA256 --probe [--state-dir DIR]"
    echo "       bash scripts/provision-wsl.sh --release SHA256 [--source DIR] [--user trainer] [--cpu-only | --check]"
    echo "An upgrade prepares a separate release with the installed trainer mode."
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
        --upgrade-from)
            (( $# >= 2 )) && [[ -n $2 ]] || fail 2 "--upgrade-from needs the active managed Linux root."
            upgrade_from=$2
            shift
            ;;
        --release)
            (( $# >= 2 )) && [[ $2 =~ ^[0-9a-f]{64}$ ]] || fail 2 "--release needs the lowercase Windows archive SHA-256."
            archive_sha256=$2
            shift
            ;;
        --probe) probe=yes ;;
        --state-dir)
            (( $# >= 2 )) && [[ $2 == /* ]] || fail 2 "--state-dir needs an absolute Linux path."
            probe_state_args+=(--state-dir "$2")
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
[[ $probe == no || -n $upgrade_from ]] || fail 2 "--probe requires --upgrade-from and --release."
[[ $probe == yes || ${#probe_state_args[@]} == 0 ]] || fail 2 "--state-dir applies only to --probe."
(( EUID == 0 )) || fail 2 "Run this bootstrap as Linux root from Windows setup."

is_managed_root() {
    [[ $1 == "$legacy_root" ||
        ( $1 == "$release_parent/"* && ${1#"$release_parent/"} =~ ^[0-9a-f]{64}$ ) ]]
}

if [[ -n $upgrade_from ]]; then
    [[ -n $archive_sha256 && $check == no && -z $mode ]] ||
        fail 2 "An upgrade requires --release and inherits the installed mode."
    is_managed_root "$upgrade_from" || fail 2 "Invalid active managed root: $upgrade_from."
elif [[ -n $archive_sha256 && $check == no ]]; then
    reconfigure=yes
fi
if [[ -n $archive_sha256 ]]; then
    root=$release_parent/$archive_sha256
    [[ $root != "$upgrade_from" ]] || fail 2 "The candidate must differ from the active managed root."
fi

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

if [[ $check == no && -z $archive_sha256 && $mode != cpu ]]; then
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
for path in trainner-gui train.py scripts/provision-wsl.sh scripts/check-managed-wsl-runtime.py \
    scripts/check-managed-studio-update.py studio/desktop.py studio/update_gate.py "${python_inputs[@]}"; do
    [[ -v 'listed[$path]' ]] || fail 3 "Incomplete Linux payload. Missing $path."
done
(cd -- "$source_root"; sha256sum --check --strict --quiet -- "$manifest") ||
    fail 3 "The extracted source failed its checksum. Extract the matching archive again."
digest=$(sha256sum < "$source_root/$manifest")
payload_id=${digest%% *}
install_id="$payload_id $trainer"
files+=("$manifest")

trusted_directory() {
    [[ -d $1 && ! -L $1 && $(realpath -e -- "$1") == "$1" &&
        $(stat -c '%u:%a' -- "$1") == 0:755 && $(findmnt -n -o FSTYPE -T "$1") == ext4 ]]
}

managed_record() {
    [[ -f $1 && ! -L $1 && $(stat -c '%u:%a' -- "$1") == 0:644 ]]
}

resolve_runtime() {
    local path owner
    path=$(realpath -e -- "$1/venv") || return 1
    owner=${path%/venv}
    [[ $path == "$owner/venv" ]] && is_managed_root "$owner" && trusted_directory "$owner" &&
        [[ -d $path && $(stat -c '%u:%a' -- "$path") == 0:755 &&
        -f $path/pyvenv.cfg && -x $path/bin/python ]] || return 1
    printf '%s\n' "$path"
}

phase="checking the previous managed installation"
previous_venv=
previous_uid=
previous_mode=
if [[ -n $upgrade_from ]]; then
    trusted_directory "$upgrade_from" || fail 3 "The active managed root is missing or redirected: $upgrade_from."
    exec 8< "$upgrade_from"
    flock -n 8 || fail 3 "Another setup is running in this distro. Wait for it to finish."
    for path in .managed-id .managed-ready .python-ready .native-ready "$manifest"; do
        managed_record "$upgrade_from/$path" || fail 3 "The active managed install is missing a trusted $path."
    done
    previous_digest=$(sha256sum < "$upgrade_from/$manifest")
    previous_id="${previous_digest%% *} $trainer"
    [[ $(cat -- "$upgrade_from/.managed-id") == "$previous_id" ]] &&
        read -r previous_payload previous_user previous_uid previous_mode < "$upgrade_from/.managed-ready" &&
        [[ "$previous_payload $previous_user" == "$previous_id" && $previous_uid =~ ^[0-9]+$ &&
        ( $previous_mode == cpu || $previous_mode == cuda ) ]] ||
        fail 3 "The active managed install does not identify the selected trainer and mode."
    previous_venv=$(resolve_runtime "$upgrade_from") ||
        fail 3 "The active managed Python runtime is missing or redirected."
    managed_record "${previous_venv%/venv}/.python-ready" ||
        fail 3 "The active Python runtime owner is incomplete."
    for path in "${python_inputs[@]}"; do
        [[ -f $upgrade_from/$path && $(realpath -e -- "$upgrade_from/$path") == "$upgrade_from/$path" ]] ||
            fail 3 "The active managed install is missing a trusted $path."
        digest=$(sha256sum < "$upgrade_from/$path")
        grep -Fxq -- "${digest%% *}  $path" "$upgrade_from/$manifest" ||
            fail 3 "The active managed install has modified dependency inputs: $path."
    done
fi

check_trainer_account() {
    phase="checking the trainer account"
    if ! passwd_entry=$(getent passwd "$trainer"); then
        [[ $check == no && -z $upgrade_from && ! -f $root/.managed-ready ]] ||
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

    [[ -z $upgrade_from || $trainer_uid == "$previous_uid" ]] ||
        fail 3 "The installed Linux trainer identity has changed. Restore the original account."
}

if [[ -n $upgrade_from ]]; then
    check_trainer_account
fi

phase="checking the managed installation identity"
if [[ $check == no && $reconfigure == no && ! -e $release_parent && ! -L $release_parent ]]; then
    mkdir -- "$release_parent" || [[ -d $release_parent ]]
fi
if [[ $check == no || -n $archive_sha256 ]]; then
    trusted_directory "$release_parent" ||
        fail 3 "$release_parent must be a root-owned 0755 directory on ext4."
fi
if [[ $probe == yes ]]; then
    probe_env=(HOME="$trainer_home" PATH="$PATH" LC_ALL="$LC_ALL")
    for name in TRAINNER_GUI_STATE_DIR XDG_STATE_HOME; do
        if [[ -v $name ]]; then
            probe_env+=("$name=${!name}")
        fi
    done
    # Keep the verified staging directory private. The gate reads the
    # trainer's process and state information as that account.
    exec runuser -u "$trainer" -- env -i "${probe_env[@]}" \
        /usr/bin/python3.12 -I - probe --old-root "$upgrade_from" "${probe_state_args[@]}" \
        < "$source_root/studio/update_gate.py"
fi
if [[ $check == yes || $reconfigure == yes ]]; then
    [[ -d $root ]] || fail 3 "The selected managed checkout has not been prepared."
elif [[ ! -e $root && ! -L $root ]]; then
    mkdir -- "$root" || [[ -d $root ]]
fi
trusted_directory "$root" || fail 3 "$root must be a root-owned 0755 directory on ext4."
exec 9< "$root"
flock -n 9 || fail 3 "Another setup is running in this distro. Wait for it to finish."
if [[ -f $root/.managed-id ]]; then
    managed_record "$root/.managed-id" && [[ $(cat -- "$root/.managed-id") == "$install_id" ]] ||
        fail 3 "This distro belongs to a different bundle or trainer. Rerun the original bundle for this install or use a separate distro."
else
    [[ $check == no && $reconfigure == no &&
        -z $(find "$root" -mindepth 1 -maxdepth 1 ! -name .managed-id.pending -print -quit) ]] ||
        fail 3 "$root does not identify a managed install. Select a dedicated setup distro."
    identity_file=$root/.managed-id.pending
    [[ ! -L $identity_file ]] || fail 3 "The pending installation identity is redirected."
    printf '%s\n' "$install_id" > "$identity_file"
    chmod 0644 -- "$identity_file"
    mv -T -- "$identity_file" "$root/.managed-id"
fi

if [[ -z $upgrade_from ]]; then
    check_trainer_account
fi

ready_id="$install_id $trainer_uid"
installed_mode=
if [[ -f $root/.managed-ready ]]; then
    managed_record "$root/.managed-ready" &&
        read -r installed_id installed_user installed_uid installed_mode < "$root/.managed-ready" &&
        [[ "$installed_id $installed_user $installed_uid" == "$ready_id" &&
        ( $installed_mode == cuda || $installed_mode == cpu ) ]] ||
        fail 3 "The installed Linux trainer identity has changed. Restore the original account."
elif [[ $check == yes || $reconfigure == yes ]]; then
    fail 3 "Managed setup is incomplete. Rerun it with the same source bundle."
fi
if [[ $check == yes ]]; then
    [[ -z $mode || $mode == "$installed_mode" ]] ||
        fail 3 "The installed trainer is in $installed_mode mode. Run setup to change modes."
    mode=$installed_mode
    if [[ $mode == cuda && -z $archive_sha256 ]]; then
        check_nvidia
    fi
elif [[ -n $upgrade_from ]]; then
    mode=$previous_mode
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

if [[ $reconfigure == yes && $mode == cuda ]]; then
    check_nvidia
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
    [[ $check == no && -z $archive_sha256 ]] ||
        fail 5 "Missing Ubuntu packages: ${missing[*]}. Restore the distro's packages before continuing."
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

python_dependencies() {
    /usr/bin/python3.12 -I -c 'import sys,tomllib; print(*tomllib.load(open(sys.argv[1],"rb"))["project"]["dependencies"], sep="\n")' \
        "$1/pyproject.toml"
}

same_python_inputs() {
    local path old_deps new_deps old_pins new_pins
    old_deps=$(python_dependencies "$1" | sort -u) || return 1
    new_deps=$(python_dependencies "$root" | sort -u) || return 1
    [[ $old_deps == "$new_deps" ]] || return 1
    for path in "${python_inputs[@]:1}"; do
        # Compare pins and resolver directives independent of comments.
        old_pins=$(sed -E 's/(^|[[:space:]]+)#.*$//; s/[[:space:]]+$//; /^$/d' "$1/$path") || return 1
        new_pins=$(sed -E 's/(^|[[:space:]]+)#.*$//; s/[[:space:]]+$//; /^$/d' "$root/$path") || return 1
        [[ $old_pins == "$new_pins" ]] || return 1
    done
}

phase="preparing the isolated Python runtime"
venv=$root/venv
wheels=$root/.download/wheels
requirements=$root/$python_requirements
if [[ -n $upgrade_from && ! -e $venv && ! -L $venv && ! -e $root/.python-ready ]] &&
    same_python_inputs "$upgrade_from"; then
    ln -sT -- "$previous_venv" "$venv"
fi
if [[ -L $venv ]]; then
    [[ -n $archive_sha256 ]] && venv=$(resolve_runtime "$root") ||
        fail 6 "The managed Python runtime link is invalid."
    runtime_owner=${venv%/venv}
    managed_record "$runtime_owner/.python-ready" && same_python_inputs "$runtime_owner" ||
        fail 6 "The shared Python runtime does not match this release."
    echo "Reusing the pinned Python runtime at $venv."
fi
python=$venv/bin/python
cuda_home=$venv/lib/python3.12/site-packages/nvidia/cu13
if [[ $check == no && $reconfigure == no && ! -f $root/.python-ready && $venv == "$root/venv" ]]; then
    [[ -f $venv/pyvenv.cfg && -x $python ]] || /usr/bin/python3.12 -m venv "$venv"
    install -d -m 0700 -- "$root/.download" "$wheels"
    echo "Available space for the managed runtime and wheel cache:"
    df -h -- "$root"
    export PIP_CONFIG_FILE=/dev/null
    pip=("$python" -I -m pip --isolated --disable-pip-version-check --no-input)
    if ! "${pip[@]}" --version > /dev/null; then
        "$python" -I -m ensurepip --upgrade
    fi
    dependencies=$(python_dependencies "$root")
    requested=(-r "$requirements")
    if [[ -n $dependencies ]]; then
        mapfile -t dependencies <<< "$dependencies"
        requested+=("${dependencies[@]}")
    fi
    wheel_sources=(--find-links "$wheels")
    if [[ -n $previous_venv ]]; then
        previous_cache=${previous_venv%/venv}/.download
        if [[ -d $previous_cache/wheels && $(realpath -e -- "$previous_cache/wheels") == "$previous_cache/wheels" &&
            $(stat -c '%u:%a' -- "$previous_cache") == 0:700 &&
            $(stat -c '%u:%a' -- "$previous_cache/wheels") == 0:700 ]]; then
            wheel_sources+=(--find-links "$previous_cache/wheels")
        fi
    fi
    # A killed pip process may leave distribution metadata before every
    # file is installed. Keep the wheel cache and overlay the same packages.
    local_pip=(--no-index "${wheel_sources[@]}" --only-binary=:all: --ignore-installed)
    if ! "${pip[@]}" install --no-cache-dir "${local_pip[@]}" "${requested[@]}"; then
        [[ $offline == no ]] ||
            fail 6 "The offline wheel cache is incomplete. Rerun setup online with the same bundle to populate $wheels."
        # pyvips 3.1.1 is published as an sdist. Cache its built wheel so
        # offline retries require only the wheelhouse and no build isolation.
        "${pip[@]}" --cache-dir "$root/.download/pip" --retries 2 --timeout 30 wheel \
            --index-url https://pypi.org/simple \
            --extra-index-url https://download.pytorch.org/whl/nightly/cu134 \
            --wheel-dir "$wheels" "${wheel_sources[@]}" \
            --only-binary=:all: --no-binary=pyvips "${requested[@]}" ||
            fail 6 "The pinned packages could not be staged from PyPI and the CUDA 13.4 nightly index. Check connectivity and wheel availability. Cached downloads are retained."
        "${pip[@]}" install --no-cache-dir "${local_pip[@]}" "${requested[@]}" ||
            fail 6 "The staged wheels could not be installed. Inspect the Python package errors above."
    fi
    [[ -x $cuda_home/bin/nvcc && -r $cuda_home/lib/libcudart.so.13 ]] ||
        fail 6 "The pinned NVIDIA compiler or CUDA runtime wheel is incomplete."
    # torch.utils.cpp_extension links with -lcudart, but the runtime wheel
    # supplies only the SONAME. Keep the linker name inside this venv.
    if [[ ! -e $cuda_home/lib/libcudart.so ]]; then
        ln -s -- libcudart.so.13 "$cuda_home/lib/libcudart.so"
    fi
    "$python" -I -m pip --disable-pip-version-check check
fi
if [[ $check == no && $reconfigure == no && ! -f $root/.python-ready ]]; then
    verified_root=
    if [[ $venv != "$root/venv" ]]; then
        verified_root=${venv%/venv}
        if [[ $venv == "$previous_venv" ]]; then
            verified_root=$upgrade_from
        fi
    fi
    # The previous marker records a successful smoke check of this venv.
    if [[ -n $verified_root ]] &&
        cmp -s -- "$verified_root/scripts/check-managed-wsl-runtime.py" "$root/scripts/check-managed-wsl-runtime.py" &&
        cmp -s -- "$verified_root/traiNNer/check/check_dependencies.py" "$root/traiNNer/check/check_dependencies.py" &&
        cmp -s -- "$verified_root/traiNNer/__init__.py" "$root/traiNNer/__init__.py"; then
        echo "The pinned runtime and its smoke check are unchanged."
    else
        as_trainer env CUDA_VISIBLE_DEVICES= timeout -k 5s 90s \
            "$python" -P "$root/scripts/check-managed-wsl-runtime.py" ||
            fail 6 "Python, torchvision or FFmpeg failed the runtime smoke check. Inspect the import or codec error above."
    fi
    touch -- "$root/.python-ready"
fi
[[ -x $python ]] && managed_record "$root/.python-ready" ||
    fail 6 "The managed Python environment is incomplete. Rerun setup with the original bundle."

phase="preparing the native tools"
if [[ $check == no && ( ! -f $root/.native-ready ||
    ! -x $root/native/studio/data/build/clang/inspect ||
    ! -r $root/traiNNer/csrc/webp_reconstruct/build/libwebp_reconstruct.so ) ]]; then
    install -d -o "$trainer_uid" -g "$trainer_gid" -m 0700 -- "$root/.cache"
    if [[ -z $archive_sha256 ]]; then
        install -d -o "$trainer_uid" -g "$trainer_gid" -m 0700 -- "$root/experiments"
    fi
    for tool in alignment data detection files history image_difference metrics \
        paired_patches provenance resample split_audit tfrecord weights; do
        install -d -o "$trainer_uid" -g "$trainer_gid" -m 0755 -- "$root/native/studio/$tool/build"
    done
    as_trainer env CUDA_VISIBLE_DEVICES= timeout -k 5s 180s "$python" -P -m studio prepare ||
        fail 6 "The Studio tools could not be built. Inspect the compiler or development package error above."
    if [[ ! -f $root/traiNNer/csrc/webp_reconstruct/build/libwebp_reconstruct.so ]]; then
        timeout -k 5s 180s bash "$root/traiNNer/csrc/webp_reconstruct/build.sh" ||
            fail 6 "The bundled libwebp reconstruction tool could not be built."
    fi
    touch -- "$root/.native-ready"
fi
managed_record "$root/.native-ready" &&
    [[ -x $root/native/studio/data/build/clang/inspect &&
    -r $root/traiNNer/csrc/webp_reconstruct/build/libwebp_reconstruct.so ]] ||
    fail 6 "The native tools are incomplete. Check the build directories in the managed distro."

phase="validating the installed trainer"
if [[ $mode == cuda ]]; then
    as_trainer timeout -k 2s 12s "$cuda_home/bin/nvcc" --version ||
        fail 6 "The isolated NVIDIA CUDA compiler cannot run. Inspect the pinned runtime and compiler files."
fi
if [[ $mode == cuda && ( -z $archive_sha256 || $reconfigure == yes ) ]]; then
    as_trainer timeout -k 5s 90s "$python" -P "$root/scripts/check-managed-wsl-runtime.py" --gpu ||
        fail 4 "The installed PyTorch or PTX kernels could not run on the first GPU. Update the NVIDIA Windows driver for CUDA 13.4 and restart the distro, or choose --cpu-only. The runtime and wheels are retained."
elif [[ $check == yes ]]; then
    as_trainer env CUDA_VISIBLE_DEVICES= timeout -k 5s 90s \
        "$python" -P "$root/scripts/check-managed-wsl-runtime.py" ||
        fail 6 "The installed CPU runtime failed its smoke check. Inspect the import or codec error above."
elif [[ $mode == cpu && -z $archive_sha256 ]]; then
    echo "CPU-only setup enables the workbench and CPU recipes. A successful CUDA setup is required for GPU training."
fi
if [[ $check == no && -z $installed_mode ]]; then
    phase="checking the isolated Studio service"
    # The readiness marker also enables Studio's state registration.
    # Start the service before writing the marker or sharing the legacy lease.
    as_trainer timeout -k 8s 35s "$python" -P "$root/scripts/check-managed-studio-update.py" ||
        fail 6 "The installed Studio service failed its /api/status smoke check."
fi
if [[ -n $archive_sha256 ]]; then
    phase="checking the shared trainer state"
    legacy_experiments=$legacy_root/experiments
    legacy_lease=$legacy_root/.cache/studio-trainer.lock
    candidate_lease=$root/.cache/studio-trainer.lock
    [[ -d $legacy_experiments && $(realpath -e -- "$legacy_experiments") == "$legacy_experiments" ]] ||
        fail 6 "The original managed experiments directory is missing or redirected."
    if [[ $check == no ]]; then
        if [[ ! -e $root/experiments && ! -L $root/experiments ]]; then
            ln -sT -- "$legacy_experiments" "$root/experiments"
        fi
        # Both checkouts require a local regular path. Share the inode so an
        # in-flight trainer keeps its flock through activation or rollback.
        as_trainer bash -ec '
            for path; do
                [[ $(realpath -e -- "${path%/*}") == "${path%/*}" ]]
            done
            if [[ ! -e $1 && ! -L $1 ]]; then
                (umask 077; set -o noclobber; : > "$1") || [[ -f $1 && ! -L $1 ]]
            fi
            [[ -f $1 && ! -L $1 ]]
            if [[ ! -e $2 && ! -L $2 ]]; then
                ln -T -- "$1" "$2"
            fi
        ' -- "$legacy_lease" "$candidate_lease" ||
            fail 6 "The candidate could not share the existing trainer lease."
    fi
    [[ -L $root/experiments && $(readlink -- "$root/experiments") == "$legacy_experiments" ]] ||
        fail 6 "The candidate does not share the original experiments directory."
    [[ -f $legacy_lease && ! -L $legacy_lease && -f $candidate_lease &&
        ! -L $candidate_lease && $legacy_lease -ef $candidate_lease ]] ||
        fail 6 "The candidate does not share the existing trainer lease."
fi
if [[ $check == no && $mode != "$installed_mode" ]]; then
    ready_file=$(mktemp -- "$root/.managed-ready.XXXXXXXX")
    printf '%s %s\n' "$ready_id" "$mode" > "$ready_file"
    chmod 0644 -- "$ready_file"
    mv -T -- "$ready_file" "$root/.managed-ready"
fi
printf 'TRAINNER_GUI_READY %s %s %s %s\n' "$payload_id" "$trainer" "$root" "$mode"
