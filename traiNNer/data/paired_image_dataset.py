import random
from collections import OrderedDict
from os import path as osp
from typing import Any

import numpy as np
import pyvips
from torchvision.transforms.functional import normalize

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from traiNNer.data.image_luma import check_luma_normalization, luma_tensor
from traiNNer.data.paired_lmdb import PairedLmdbReader
from traiNNer.data.transforms import (
    augment_vips_pair,
    get_vips_dtype,
    paired_random_crop_vips,
)
from traiNNer.utils import img2tensor
from traiNNer.utils.img_util import img2rgb, vipsimfrombuffer, vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class PairedImageDataset(BaseDataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        # file client (io backend)
        self.lmdb: PairedLmdbReader | None = None
        self.io_backend_opt = opt.io_backend
        if self.io_backend_opt["type"] not in ("disk", "lmdb"):
            raise ValueError(f"Unsupported paired image storage {self.io_backend_opt['type']!r}.")
        self._decoded: OrderedDict[str, tuple[pyvips.Image, int]] = OrderedDict()
        self._decoded_bytes = 0
        self._native_crop = opt.phase == "train" and opt.native_paired_crop
        if self._native_crop:
            from traiNNer.data.paired_crop import kernel
            kernel(luma=opt.color is not None)  # Fail before starting loader workers if the requested tool is missing.
            if type(self) is PairedImageDataset:
                self.__getitems__ = self._batch_samples
        self.mean = opt.mean
        self.std = opt.std
        self.color = opt.color
        if self.color is not None:
            check_luma_normalization(opt)

        assert isinstance(opt.dataroot_lq, list), (
            f"dataroot_lq must be defined for dataset {opt.name}"
        )
        assert isinstance(opt.dataroot_gt, list), (
            f"dataroot_gt must be defined for dataset {opt.name}"
        )

        self.filename_tmpl = opt.filename_tmpl
        self.gt_folder, self.lq_folder = opt.dataroot_gt, opt.dataroot_lq

        if self.io_backend_opt["type"] == "lmdb":
            self.lmdb = PairedLmdbReader(self.io_backend_opt)
            self.paths = paired_paths_from_lmdb(
                (self.lq_folder, self.gt_folder), ("lq", "gt"), include_storage=True
            )
        elif self.opt.meta_info is not None:
            self.paths = paired_paths_from_meta_info_file(
                (self.lq_folder, self.gt_folder),
                ("lq", "gt"),
                self.opt.meta_info,
                self.filename_tmpl,
            )
        else:
            self.paths = paired_paths_from_folder(
                (self.lq_folder, self.gt_folder), ("lq", "gt"), self.filename_tmpl
            )

    def __getstate__(self) -> dict[str, Any]:
        # libvips images belong to the worker that decoded them.
        return {**vars(self), "_decoded": OrderedDict(), "_decoded_bytes": 0}

    def _read_image(self, path: str) -> pyvips.Image:
        if not self.opt.decoded_cache_mib or self.opt.phase != "train":
            return vipsimfrompath(path)
        cached = self._decoded.get(path)
        if cached is not None:
            self._decoded.move_to_end(path)
            return cached[0]

        image = vipsimfrompath(path)
        limit = self.opt.decoded_cache_mib * 2**20
        size = (
            int(image.width)
            * int(image.height)
            * int(image.bands)
            * np.dtype(get_vips_dtype(image)).itemsize
        )
        if size <= limit:
            while self._decoded_bytes + size > limit:
                _, (_, evicted) = self._decoded.popitem(last=False)
                self._decoded_bytes -= evicted
            try:
                image = image.copy_memory()
            except pyvips.Error as exc:
                raise OSError(
                    f"Could not decode {path} into the image cache: {exc}"
                ) from exc
            self._decoded[path] = (image, size)
            self._decoded_bytes += size
        return image

    def _batch_samples(self, indices):
        from traiNNer.data.paired_crop import CropSamples
        high, scale = self.opt.gt_size, self.opt.scale
        if not high or not scale or high % scale:
            raise ValueError("Native paired batches require a GT patch divisible by the scale.")
        samples = CropSamples(len(indices), high, high // scale, channels=1 if self.color else 3)
        for offset,index in enumerate(indices):
            samples.append(self.__getitem__(index, destination=(samples.gt[offset], samples.lq[offset])))
        return samples

    def __getitem__(self, index: int | tuple[int, str], *, destination=None) -> DataFeed:
        if isinstance(index, tuple):
            index, seed = index
            rng = random.Random(seed)
        else:
            rng = random
        scale = self.opt.scale
        assert scale is not None

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        pair = self.paths[index]
        gt_path, lq_path = pair["gt_path"], pair["lq_path"]
        encoded_lq = None
        if self.lmdb is None:
            vips_img_gt = self._read_image(gt_path)
            vips_img_lq = self._read_image(lq_path)
        else:
            encoded_lq = self.lmdb.read(pair["lq_db"], lq_path)
            vips_img_lq = vipsimfrombuffer(encoded_lq)
            vips_img_gt = vipsimfrombuffer(self.lmdb.read(pair["gt_db"], gt_path))
            lq_path = osp.join(pair["lq_db"], lq_path + ".png")
            gt_path = osp.join(pair["gt_db"], gt_path + ".png")

        packed = None
        if self._native_crop:
            from traiNNer.data.paired_crop import try_crop_pair
            packed = try_crop_pair(vips_img_gt, vips_img_lq, self.opt, gt_path, lq_path, destination, rng=rng)
        if packed is not None:
            img_gt_tensor, img_lq_tensor = packed
        else:
            if self.opt.phase == "train":
                assert self.opt.gt_size is not None
                assert self.opt.use_hflip is not None
                assert self.opt.use_rot is not None

                # flip, rotation
                vips_img_gt, vips_img_lq = augment_vips_pair(
                    (vips_img_gt, vips_img_lq),
                    self.opt.use_hflip,
                    self.opt.use_rot,
                    self.opt.use_rot,
                    rng=rng,
                )

                # random crop
                img_gt, img_lq = paired_random_crop_vips(
                    vips_img_gt, vips_img_lq, self.opt.gt_size, scale, lq_path, gt_path, rng=rng
                )

                assert isinstance(img_gt, np.ndarray)
                assert isinstance(img_lq, np.ndarray)
            else:
                img_gt = img2rgb(vips_img_gt.numpy())
                img_lq = img2rgb(vips_img_lq.numpy())

            # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
            # TODO: It is better to update the datasets, rather than force to crop
            if self.opt.phase != "train":
                img_gt = img_gt[0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :]

            if self.color is None:
                img_gt_tensor = img2tensor(img_gt, float32=True, from_bgr=False)
                img_lq_tensor = img2tensor(img_lq, float32=True, from_bgr=False)
            else:
                img_gt_tensor = luma_tensor(img_gt, self.color)
                img_lq_tensor = luma_tensor(img_lq, self.color)
            if destination is not None:
                gt_out,lq_out = destination
                if gt_out.shape != img_gt_tensor.shape or lq_out.shape != img_lq_tensor.shape:
                    raise ValueError("Paired batch buffer shape must match the converted patch.")
                gt_out.copy_(img_gt_tensor)
                lq_out.copy_(img_lq_tensor)
                img_gt_tensor,img_lq_tensor = gt_out,lq_out

        # normalize
        if self.mean is not None and self.std is not None:
            normalize(img_lq_tensor, self.mean, self.std, inplace=True)
            normalize(img_gt_tensor, self.mean, self.std, inplace=True)

        result: DataFeed = {
            "lq": img_lq_tensor,
            "gt": img_gt_tensor,
            "lq_path": lq_path,
            "gt_path": gt_path,
        }

        if encoded_lq is not None and self.opt.phase != "train":
            result.update(lq_encoded=encoded_lq, lq_db=pair["lq_db"], gt_db=pair["gt_db"])
            if self.opt.phase != "train":
                result.update(lq_key=pair["lq_path"], gt_key=pair["gt_path"])
        return result

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "image pairs"
