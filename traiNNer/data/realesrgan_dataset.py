import math
import os
import os.path as osp
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing
from functools import cache
from multiprocessing.util import Finalize

import numpy as np
import pyvips
import torch
from trainner_config.options import OTF_IMAGE_BACKENDS
from trainner_data.lmdb import PairedLmdbReader
from trainner_data.manifests import lmdb_keys, manifest_filename
from trainner_data.zip_images import ZipImageReader

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from traiNNer.data.dvd_response import DvdGeometry, DvdSample, Window
from traiNNer.data.still_source import (
    is_still_source,
    orientation_flags,
    source_bytes,
)
from traiNNer.data.transforms import (
    augment_vips,
    draw_augment_vips,
    single_crop_vips,
)
from traiNNer.utils import RNG, img2tensor, scandir
from traiNNer.utils._drvfs import close_drvfs_directories, pin_drvfs_directories
from traiNNer.utils.img_util import (
    img2rgb,
    is_drvfs_directory,
    read_drvfs_image_bytes,
    vipsimfrombuffer,
    vipsimfrompath,
    vipsimfromprefetched,
)
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed

# A 21x21 unit pulse: convolving with it leaves the image as it is. Shared by
# every sample that carries a kernel no stage applies; nothing writes to it.
_UNIT_PULSE = torch.zeros(21, 21)
_UNIT_PULSE[10, 10] = 1

# Project RNG draws random_mixed_kernels takes per kernel type (no kernel
# noise): sigma_x, then sigma_y and rotation if anisotropic, then the beta
# branch and beta for generalized and plateau kernels.
_MIXED_KERNEL_DRAWS = {
    "iso": 1,
    "aniso": 3,
    "generalized_iso": 3,
    "generalized_aniso": 5,
    "plateau_iso": 3,
    "plateau_aniso": 5,
}


def _skip_blur_kernel_draws(
    sinc_prob: float, kernel_list: list[str], kernel_prob: list[float]
) -> None:
    """Take the draws of one blur kernel after its size without building it.

    Matches the kernel branch of sample_degradation_kernels draw for draw: the
    sinc choice, then the cutoff, or the kernel type and its parameters.
    """
    if RNG.get_rng().uniform() < sinc_prob:
        RNG.get_rng().random()
        return
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type not in _MIXED_KERNEL_DRAWS:
        raise ValueError("Unknown kernel type")
    RNG.get_rng().random(_MIXED_KERNEL_DRAWS[kernel_type])


def sample_degradation_kernels(
    opt: DatasetOptions,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the first and second blur kernels and the final sinc kernel, 21x21 each.

    Uses Python's random module and the project RNG in a fixed order, so one
    call per image keeps the image dataset's random stream. A blur stage whose
    probability is 0 never applies its kernel: its draws are taken all the same
    and the shared unit pulse stands in for the kernel, as it does for the final
    sinc kernel whenever that is not drawn.
    """
    # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
    kernel_size = random.choice(range(opt.kernel_range[0], opt.kernel_range[1] + 1, 2))
    if opt.blur_prob <= 0:
        _skip_blur_kernel_draws(opt.sinc_prob, opt.kernel_list, opt.kernel_prob)
        kernel = _UNIT_PULSE
    else:
        if RNG.get_rng().uniform() < opt.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            else:
                omega_c = RNG.get_rng().uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                opt.kernel_list,
                opt.kernel_prob,
                kernel_size,
                opt.blur_sigma,
                opt.blur_sigmay or opt.blur_sigma,
                opt.blur_rotation or (-math.pi, math.pi),
                opt.betag_range,
                opt.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = torch.FloatTensor(
            np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        )

    # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
    kernel_size = random.choice(
        range(opt.kernel_range2[0], opt.kernel_range2[1] + 1, 2)
    )
    if opt.blur_prob2 <= 0:
        _skip_blur_kernel_draws(opt.sinc_prob2, opt.kernel_list2, opt.kernel_prob2)
        kernel2 = _UNIT_PULSE
    else:
        if RNG.get_rng().uniform() < opt.sinc_prob2:
            if kernel_size < 13:
                omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
            else:
                omega_c = RNG.get_rng().uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                opt.kernel_list2,
                opt.kernel_prob2,
                kernel_size,
                opt.blur_sigma2,
                opt.blur_sigmay2 or opt.blur_sigma2,
                opt.blur_rotation or (-math.pi, math.pi),
                opt.betag_range2,
                opt.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = torch.FloatTensor(
            np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        )

    # ------------------------------------- the final sinc kernel ------------------------------------- #
    if RNG.get_rng().uniform() < opt.final_sinc_prob:
        kernel_size = random.choice(
            range(opt.final_kernel_range[0], opt.final_kernel_range[1] + 1, 2)
        )
        omega_c = RNG.get_rng().uniform(np.pi / 3, np.pi)
        sinc_kernel = torch.FloatTensor(
            circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        )
    else:
        # convolving with a pulse brings no blurry effect
        sinc_kernel = _UNIT_PULSE

    # Sample this gate after all three kernels, as in NeoSR. The ordinary
    # trainer and either endpoint retain their existing random stream.
    p = opt.sample_blur_prob
    if p <= 0 or (p < 1 and RNG.get_rng().uniform() >= p):
        kernel = _UNIT_PULSE
    return kernel, kernel2, sinc_kernel


def _downsample_input_vips(
    img: pyvips.Image, algorithm: str = "catrom"
) -> pyvips.Image:
    """Halve HR in encoded RGB on CPU, independently of LR degradation."""
    from chainner_ext import ResizeFilter, resize

    pixels = img.numpy()
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    if pixels.shape[2] > 3:
        pixels = pixels[:, :, :3]  # Match the loader's existing alpha handling.
    dtype = pixels.dtype
    # Normalize in place; keep grayscale at one channel until the existing crop
    # conversion, and do not allocate a second full-size float image.
    pixels = pixels.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        pixels *= 1.0 / np.iinfo(dtype).max
    size = (max(1, img.width // 2), max(1, img.height // 2))
    pixels = np.ascontiguousarray(pixels)
    if algorithm == "dpid":
        from traiNNer.data.dpid import dpid_half

        pixels = dpid_half(pixels)
    else:
        filters = {
            "catrom": ResizeFilter.CubicCatrom,
            "mitchell": ResizeFilter.CubicMitchell,
            "hermite": ResizeFilter.Hermite,
        }
        if algorithm not in filters:
            raise ValueError(f"Unsupported HR downscale_algorithm: {algorithm!r}")
        pixels = resize(pixels, size, filters[algorithm], False)
    # Keep sub-byte precision and clip overshoot in place at the reduced size.
    np.clip(pixels, 0.0, 1.0, out=pixels)
    return pyvips.Image.new_from_array(pixels)


def _reflect_pad_vips(img: pyvips.Image, min_size: int) -> pyvips.Image:
    height: int = img.height  # type: ignore
    width: int = img.width  # type: ignore
    if height >= min_size and width >= min_size:
        return img

    pad_height = max(0, min_size - height)
    pad_width = max(0, min_size - width)
    return img.embed(
        pad_width // 2,
        pad_height // 2,
        width + pad_width,
        height + pad_height,
        extend=pyvips.enums.Extend.MIRROR,
    )


def _neosr_pad_vips(img: pyvips.Image, side: int) -> pyvips.Image:
    """OpenCV REFLECT_101 past the right and bottom, using lazy libvips regions."""
    width, height = img.width, img.height
    if width < side:
        if width > 2:
            interior = img.crop(1, 0, width - 2, height).fliphor()
            img = img.join(interior, "horizontal")
        img = img.replicate((side + img.width - 1) // img.width, 1).crop(0, 0, side, height)
    if height < side:
        width = img.width
        if height > 2:
            interior = img.crop(0, 1, width, height - 2).flipver()
            img = img.join(interior, "vertical")
        img = img.replicate(1, (side + img.height - 1) // img.height).crop(0, 0, width, side)
    return img


def _window_tensor(img: pyvips.Image, window: Window) -> torch.Tensor:
    """A window of an image as float32 RGB CHW, read like the work canvas."""
    region = img.crop(window.left, window.top, window.width, window.height)
    return img2tensor(img2rgb(region.numpy()), from_bgr=False, float32=True)


@cache
def _configure_windows_reader_worker() -> None:
    if os.environ.get("VIPS_CONCURRENCY"):
        return
    # libvips is initialized before DataLoader forks. Its environment default
    # would create a machine-sized thread pool in each image worker.
    from ctypes import CDLL, c_int

    if pyvips.API_mode:
        import _libvips

        library = _libvips.__file__
    else:
        library = pyvips.library_name("vips", 42)
    try:
        set_limit = CDLL(library).vips_concurrency_set
    except (AttributeError, OSError) as exc:
        from logging import getLogger

        getLogger(__name__).warning("Keeping the current libvips worker thread count: %s", exc)
        return
    set_limit.argtypes = [c_int]
    set_limit.restype = None
    set_limit(2)


class _LightImageReaders:
    def __init__(self, *, pin_directories: bool) -> None:
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.pins: dict[str, int] | None = {} if pin_directories else None

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.pins is not None:
            close_drvfs_directories(self.pins)


@cache
def _worker_image_readers() -> _LightImageReaders:
    # Created only inside a DataLoader worker, after fork or spawn.
    readers = _LightImageReaders(pin_directories=True)
    Finalize(None, readers.close, exitpriority=10)
    return readers


@DATASET_REGISTRY.register(suffix="traiNNer")
class RealESRGANDataset(BaseDataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        # Set by build_dataloader when its collation takes decoded RGB bytes.
        self.still_sources = False
        self.neosr_canvas_size = None
        if opt.otf_canvas != "trainer":
            if opt.gt_size is None or not opt.scale or min(opt.gt_size, opt.scale) <= 0:
                raise ValueError("NeoSR OTF geometry requires positive GT patch size and scale.")
            side = opt.gt_size + 32 if opt.otf_canvas == "neosr_tight" else max(512, opt.gt_size)
            self.neosr_canvas_size = ((side + opt.scale - 1) // opt.scale) * opt.scale
        self.zip_reader: ZipImageReader | None = None
        self.lmdb: PairedLmdbReader | None = None
        self.storage: list[str] = []
        backend = opt.io_backend["type"]
        if backend not in OTF_IMAGE_BACKENDS:
            raise ValueError(f"Unsupported OTF image storage {backend!r}.")
        self.gt_folders = opt.dataroot_gt
        if backend == "zip":
            from trainner_data.zip_images import validate_zip_dataset

            self.gt_folders = validate_zip_dataset(opt)

        assert isinstance(self.gt_folders, list), (
            f"dataroot_gt must be a list of folders for dataset {opt.name}"
        )
        self.target_gt_folders = opt.target_dataroot_gt
        self.target_paths: list[str] | None = None

        if self.target_gt_folders is not None:
            assert isinstance(self.target_gt_folders, list), (
                f"target_dataroot_gt must be a list of folders for dataset {opt.name}"
            )
            if len(self.target_gt_folders) != len(self.gt_folders):
                raise ValueError(
                    "target_dataroot_gt must have the same number of roots as dataroot_gt"
                )

        self.paths: list[str] = []
        if backend == "lmdb":
            if self.target_gt_folders is not None:
                raise NotImplementedError(
                    "target_dataroot_gt is not supported with lmdb datasets."
                )
            self.lmdb = PairedLmdbReader(opt.io_backend)
            for folder in self.gt_folders:
                keys = lmdb_keys(folder)
                self.paths.extend(keys)
                self.storage.extend([folder] * len(keys))
            return

        manifest_paths = None
        if opt.meta_info is not None:
            with open(opt.meta_info) as fin:
                manifest_paths = [
                    manifest_filename(line) for line in fin if line.strip()
                ]

        if backend == "zip":
            self.zip_reader = ZipImageReader(self.gt_folders, manifest_paths)
            self.paths = self.zip_reader.paths
            self.storage = self.zip_reader.storage
            # The libvips limit is process-wide. Mixed roots keep the native default.
            self._zip_on_drvfs = all(
                is_drvfs_directory(osp.dirname(osp.realpath(folder))) for folder in self.gt_folders
            )
            self.__getitems__ = self._zip_batch
            return

        if self.target_gt_folders is not None:
            self.target_paths = []
        for folder_idx, folder in enumerate(self.gt_folders):
            names = (
                manifest_paths
                if manifest_paths is not None
                else sorted(scandir(folder, full_path=False, recursive=True))
            )
            self.paths.extend(osp.join(folder, name) for name in names)
            if self.target_paths is not None:
                assert self.target_gt_folders is not None
                target_folder = self.target_gt_folders[folder_idx]
                self.target_paths.extend(
                    osp.join(target_folder, name) for name in names
                )

        if any(is_drvfs_directory(folder) for folder in self.gt_folders):
            self.__getitems__ = self._windows_disk_batch

    def _zip_batch(self, indices: list[int]) -> list[DataFeed]:
        assert self.zip_reader is not None
        in_worker = torch.utils.data.get_worker_info() is not None
        if in_worker and self._zip_on_drvfs:
            _configure_windows_reader_worker()
        with closing(self.zip_reader.read_ahead(indices, in_worker=in_worker)) as records:
            return [self.__getitem__(i, zip_bytes=b) for i, b in zip(indices, records, strict=True)]

    def _windows_disk_batch(self, indices: list[int]) -> list[DataFeed]:
        # Read ahead across DrvFS. Sample ordering and all random draws stay on
        # the DataLoader worker's main thread, including for repeated indices.
        in_worker = torch.utils.data.get_worker_info() is not None
        if in_worker:
            _configure_windows_reader_worker()
        if not self.opt.downsample_input:
            if in_worker:
                return self._light_disk_batch(indices, _worker_image_readers())
            # Keep the caller's pool local to this batch and pin only in workers.
            with closing(_LightImageReaders(pin_directories=False)) as readers:
                return self._light_disk_batch(indices, readers)

        pending = deque()
        samples = []
        read_workers = 2
        with ThreadPoolExecutor(max_workers=read_workers) as readers:
            for index in indices:
                pending.append((index, readers.submit(vipsimfrompath, self.paths[index])))
                if len(pending) > read_workers:
                    i, image = pending.popleft()
                    samples.append(self.__getitem__(i, source=image.result()))
            for i, image in pending:
                samples.append(self.__getitem__(i, source=image.result()))
        return samples

    def _light_disk_batch(self, indices: list[int], readers: _LightImageReaders) -> list[DataFeed]:
        # Budget the worker's pins before any of its read threads open files.
        if readers.pins is not None:
            pin_drvfs_directories(readers.pins, (self.paths[i] for i in indices))
        pending = deque()
        samples = []
        ahead = 8
        try:
            for i in indices[:ahead]:
                pending.append(readers.pool.submit(read_drvfs_image_bytes, self.paths[i]))
            for following, index in enumerate(indices, start=ahead):
                if following < len(indices):
                    path = self.paths[indices[following]]
                    pending.append(readers.pool.submit(read_drvfs_image_bytes, path))
                source = vipsimfromprefetched(self.paths[index], pending.popleft().result())
                samples.append(self.__getitem__(index, source=source))
        finally:
            # A failed sample can leave speculative reads queued for this batch.
            if pending:
                for future in pending:
                    future.cancel()
                wait(pending)
        return samples

    def __getitem__(
        self, index: int, *, source: pyvips.Image | None = None, zip_bytes: bytes | None = None
    ) -> DataFeed:
        gt_path = self.paths[index]

        assert self.opt.use_hflip is not None
        assert self.opt.use_rot is not None

        gt_db = None
        if self.zip_reader is not None:
            archive, name = self.storage[index], gt_path
            gt_path = f"{archive}!/{name}"
            if zip_bytes is None:
                if self._zip_on_drvfs and torch.utils.data.get_worker_info() is not None:
                    _configure_windows_reader_worker()
                zip_bytes = self.zip_reader.read(index)
            try:
                vips_img_gt = vipsimfrombuffer(zip_bytes)
            except pyvips.Error as exc:
                raise ValueError(f"Could not decode ZIP image {gt_path}: {exc}") from exc
        elif self.lmdb is None:
            vips_img_gt = vipsimfrompath(gt_path) if source is None else source
        else:
            gt_db = self.storage[index]
            gt_key = gt_path
            gt_path = f"{gt_db} [{gt_key!r}]"
            try:
                vips_img_gt = vipsimfrombuffer(self.lmdb.read(gt_db, gt_key))
            except pyvips.Error as exc:
                raise ValueError(f"Could not decode LMDB GT {gt_path}. {exc}") from exc
        target_gt_path = None
        vips_img_target = None
        if self.target_paths is not None:
            target_gt_path = self.target_paths[index]
            vips_img_target = vipsimfrompath(target_gt_path)

        h: int = vips_img_gt.height  # type: ignore
        w: int = vips_img_gt.width  # type: ignore
        assert self.opt.gt_size is not None

        if vips_img_target is not None:
            target_h: int = vips_img_target.height  # type: ignore
            target_w: int = vips_img_target.width  # type: ignore
            if target_h != h or target_w != w:
                raise ValueError(
                    f"target GT size mismatch for {target_gt_path}: "
                    f"expected ({h}, {w}), got ({target_h}, {target_w})"
                )

        # A source of the one canvas geometry the GPU makes goes as its decoded
        # bytes; the draws stay the CPU path's (see traiNNer.data.still_source).
        still_source = self.still_sources and is_still_source(vips_img_gt, self.opt)

        # Reduce the source itself, not just the synthesized LR. Both source and
        # supervision target must stay aligned before padding and shared cropping.
        if self.opt.downsample_input and not still_source:
            vips_img_gt = _downsample_input_vips(
                vips_img_gt, self.opt.downscale_algorithm
            )
            if vips_img_target is not None:
                vips_img_target = _downsample_input_vips(
                    vips_img_target, self.opt.downscale_algorithm
                )

        # The x2 DVD response needs the source in its canonical orientation. A
        # source that is exactly gt_size square becomes the centre of its canvas,
        # which the model reads back; any other keeps its canonical pixels here,
        # decoded once for the canvas and the DVD window.
        if still_source:
            canonical_width = canonical_height = self.opt.gt_size
        else:
            canonical_width, canonical_height = vips_img_gt.width, vips_img_gt.height
        dvd_window = self.opt.dvd_prob > 0 and not (
            canonical_width == canonical_height == self.opt.gt_size
        )
        # A halved source already wraps its numpy pixels; copy_memory would
        # return it in a wrapper without the array's reference, freeing them.
        if dvd_window and not self.opt.downsample_input:
            vips_img_gt = vips_img_gt.copy_memory()
            if vips_img_target is not None:
                vips_img_target = vips_img_target.copy_memory()
        canonical_gt, canonical_target = vips_img_gt, vips_img_target

        # Augment after reduction so flips/rotations touch fewer pixels when enabled.
        hflip, vflip, rot90 = draw_augment_vips(self.opt.use_hflip, self.opt.use_rot)
        if still_source:
            h = w = self.opt.gt_size
        else:
            vips_img_gt = augment_vips(
                vips_img_gt, hflip, vflip, rot90, randomize=False
            )
            if vips_img_target is not None:
                vips_img_target = augment_vips(
                    vips_img_target, hflip, vflip, rot90, randomize=False
                )
            h = vips_img_gt.height  # type: ignore
            w = vips_img_gt.width  # type: ignore

        crop_pad_size = self.neosr_canvas_size or (self.opt.gt_size + 32)
        padded_height = h < crop_pad_size
        padded_width = w < crop_pad_size
        # The canvas origin in the oriented image, negative on a padded axis.
        canvas_left = -(max(0, crop_pad_size - w) // 2)
        canvas_top = -(max(0, crop_pad_size - h) // 2)
        if self.neosr_canvas_size is not None:
            canvas_left = canvas_top = 0
        # pad
        if still_source:
            img_gt = source_bytes(vips_img_gt)
        elif padded_height or padded_width:
            pad_image = _neosr_pad_vips if self.neosr_canvas_size else _reflect_pad_vips
            vips_img_gt = pad_image(vips_img_gt, crop_pad_size)
            if vips_img_target is not None:
                vips_img_target = pad_image(vips_img_target, crop_pad_size)

            # Crop coordinates must use the padded dimensions. In particular, one
            # dimension may have been padded while the other was already oversized.
            h = vips_img_gt.height  # type: ignore
            w = vips_img_gt.width  # type: ignore
        # crop
        if w > crop_pad_size or h > crop_pad_size:
            y = random.randint(0, h - crop_pad_size)
            x = random.randint(0, w - crop_pad_size)
            canvas_left += x
            canvas_top += y
            img_gt = single_crop_vips(vips_img_gt, crop_pad_size, x, y, gt_path)
            if vips_img_target is not None:
                img_target = single_crop_vips(
                    vips_img_target, crop_pad_size, x, y, target_gt_path
                )
        elif not still_source:
            img_gt = img2rgb(vips_img_gt.numpy())
            if vips_img_target is not None:
                img_target = img2rgb(vips_img_target.numpy())

        kernel, kernel2, sinc_kernel = sample_degradation_kernels(self.opt)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if not still_source:
            img_gt = img2tensor(img_gt, from_bgr=False, float32=True)

        output: DataFeed = {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            "gt_path": gt_path,
            # These masks request a centered final crop on the native padded axes.
            # NeoSR draws the final crop uniformly, including its reflected region.
            "otf_padded_height": padded_height and self.neosr_canvas_size is None,
            "otf_padded_width": padded_width and self.neosr_canvas_size is None,
        }
        if gt_db is not None:
            output.update(gt_db=gt_db, gt_key=self.paths[index])
        if still_source:
            output["gt_orientation"] = orientation_flags(hflip, vflip, rot90)
        if vips_img_target is not None:
            output["target_gt"] = img2tensor(img_target, from_bgr=False, float32=True)
            output["target_gt_path"] = target_gt_path
        if self.opt.dvd_prob > 0:
            geometry = DvdGeometry(
                canonical_width,
                canonical_height,
                canvas_left,
                canvas_top,
                crop_pad_size,
                hflip,
                vflip,
                rot90,
            )
            window = geometry.window()
            output["dvd"] = DvdSample(
                geometry,
                _window_tensor(canonical_gt, window) if dvd_window else None,
                _window_tensor(canonical_target, window)
                if dvd_window and canonical_target is not None
                else None,
            )

        return output

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "images"
