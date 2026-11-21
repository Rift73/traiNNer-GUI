#!/usr/bin/env bash
# Builds build/libwebp_reconstruct.so from reconstruct.c, loop_params.c and the
# libwebp 1.6.0 source archive next to them. Needs a C compiler, cmake and
# ninja; downloads nothing. The default is a portable x86-64 build; pass a CPU
# target as the first argument to build for that CPU instead, e.g.
#   ./build.sh skylake-avx512
set -euo pipefail
cd -- "$(dirname -- "$0")"

archive=libwebp-1.6.0.tar.gz
digest=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
source_dir=$PWD/build/libwebp-1.6.0
lib_dir=$PWD/build/libwebp
compiler=$(command -v clang-18 || command -v clang || command -v cc)
flags=(-O3 -DNDEBUG)
if [[ -n ${1:-} ]]; then
    flags+=("-march=$1")
fi

mkdir -p build
if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
    echo "$digest  $archive" | sha256sum -c -
    tar -xzf "$archive" -C build
fi

options=()
for tool in ANIM_UTILS CWEBP DWEBP GIF2WEBP IMG2WEBP VWEBP WEBPINFO LIBWEBPMUX WEBPMUX EXTRAS; do
    options+=("-DWEBP_BUILD_$tool=OFF")
done
cmake -S "$source_dir" -B "$lib_dir" -G Ninja -DCMAKE_C_COMPILER="$compiler" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="${flags[*]}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF "${options[@]}" >/dev/null
cmake --build "$lib_dir" --target webp sharpyuv >/dev/null

"$compiler" "${flags[@]}" -DHAVE_CONFIG_H -I"$lib_dir" -I"$source_dir" -fPIC -shared \
    -fvisibility=hidden -Wl,-Bsymbolic -Wl,--exclude-libs,ALL \
    reconstruct.c loop_params.c "$lib_dir/libwebp.a" "$lib_dir/libsharpyuv.a" \
    -lpthread -lm -o build/libwebp_reconstruct.so
echo "built $PWD/build/libwebp_reconstruct.so (${flags[*]})"
