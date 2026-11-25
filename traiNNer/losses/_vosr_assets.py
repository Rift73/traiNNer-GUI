"""Fetch the pinned Linux teacher bundle and unpack individual stages in host memory."""

import ctypes
import hashlib
import json
import mmap
import os
import platform
from functools import cache
from pathlib import Path

import torch

from trainner_config.official_weights import ensure_asset

MANIFEST_SHA256 = "56ec71afc0982ecf14bc0e8a7738fe64601df9edacd284973582da6a6bc49464"
BUNDLE = "vosr2-linux-" + MANIFEST_SHA256[:12]
BASE_URL = f"https://rift.steropes.feralhosting.com/{BUNDLE}"
GPU = "NVIDIA GeForce RTX 5090"
TRT_VERSION = "11.4.0.106"


@cache
def _zstd():
    # libzstd is part of the Ubuntu base system used by the WSL trainer.
    lib = ctypes.CDLL("libzstd.so.1")
    lib.ZSTD_decompress.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
    ]
    lib.ZSTD_decompress.restype = ctypes.c_size_t
    return lib


class ManagedVosrAssets:
    def __init__(self, device: torch.device, trt_version: str) -> None:
        gpu = torch.cuda.get_device_name(device)
        if (platform.system(), platform.machine(), gpu, trt_version) != (
            "Linux", "x86_64", GPU, TRT_VERSION,
        ):
            raise RuntimeError(
                f"No managed VOSR bundle for {platform.system()} {platform.machine()}, "
                f"{gpu}, TensorRT {trt_version}. Available bundle: Linux x86_64, "
                f"{GPU}, TensorRT {TRT_VERSION}. A compatible local engine_dir "
                "and plugin can be supplied as overrides."
            )
        cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        self.directory = cache_root / "trainner" / BUNDLE
        manifest = ensure_asset(self.directory / "manifest.json", f"{BASE_URL}/manifest.json", MANIFEST_SHA256)
        self.files = json.loads(manifest.read_bytes())["files"]

    def unpack(self, name: str) -> bytearray:
        spec = self.files[name]
        compressed = name + ".zst"
        path = ensure_asset(self.directory / compressed, f"{BASE_URL}/{compressed}", spec["compressed_sha256"])
        size = spec["size"]
        raw = bytearray(size)
        with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY) as source:
            if len(source) != spec["compressed_size"]:
                raise ValueError(f"Unexpected compressed size for {name}.")
            written = _zstd().ZSTD_decompress(
                ctypes.addressof(ctypes.c_char.from_buffer(raw)), size,
                ctypes.addressof(ctypes.c_char.from_buffer(source)), len(source),
            )
        if written != size or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError(f"Unpacked VOSR asset did not match the pinned build: {name}.")
        return raw

    def load_plugin(self):
        name = "vosr2_qkv.so"
        data = self.unpack(name)
        # dlopen needs a pathname. Keep the anonymous file descriptor alive with
        # its handle, so this pathname cannot be recycled for a different library.
        file = os.fdopen(os.memfd_create(name, os.MFD_CLOEXEC), "wb")
        try:
            file.write(data)
            file.flush()
            library = ctypes.CDLL(f"/proc/self/fd/{file.fileno()}", mode=ctypes.RTLD_GLOBAL)
        except BaseException:
            file.close()
            raise
        return library, file
