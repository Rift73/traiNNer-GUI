from os import path as osp

from torch import Tensor
from torchvision.transforms.functional import normalize

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.image_luma import check_luma_normalization, luma_tensor
from trainner_data.lmdb import PairedLmdbReader
from trainner_data.manifests import lmdb_keys, manifest_filename
from traiNNer.utils import img2tensor, scandir
from traiNNer.utils.img_util import img2rgb, vipsimfrombuffer, vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


@DATASET_REGISTRY.register()
class SingleImageDataset(BaseDataset):
    """Read only lq images in the test phase.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There are two modes:
    1. 'meta_info_file': Use meta information file to generate paths.
    2. 'folder': Scan folders to generate paths.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        self.lmdb = None
        self.storage: list[str] = []
        self.mean = opt.mean
        self.std = opt.std
        if opt.color is not None:
            check_luma_normalization(opt)
        self.lq_folder = opt.dataroot_lq

        assert self.lq_folder is not None and isinstance(self.lq_folder, list), (
            f"dataroot_lq must be defined as a list of paths for dataset {opt.name}"
        )

        if opt.io_backend["type"] == "lmdb":
            self.lmdb = PairedLmdbReader(opt.io_backend)
            self.paths = []
            for folder in self.lq_folder:
                keys = lmdb_keys(folder)
                self.paths.extend(keys)
                self.storage.extend([folder] * len(keys))
        elif self.opt.meta_info is not None:
            self.paths = []
            with open(self.opt.meta_info) as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    filename = manifest_filename(line)
                    for folder in self.lq_folder:
                        self.paths.append(osp.join(folder, filename))

        else:
            self.paths = []
            for folder in self.lq_folder:
                self.paths.extend(
                    sorted(scandir(folder, recursive=True, full_path=True))
                )

    def __getitem__(self, index: int) -> DataFeed:
        lq_path = self.paths[index]
        if self.lmdb is None:
            encoded = None
            image = vipsimfrompath(lq_path)
        else:
            database = self.storage[index]
            encoded = self.lmdb.read(database, lq_path)
            image = vipsimfrombuffer(encoded)
            lq_path = osp.join(database, lq_path + ".png")
        img_lq = img2rgb(image.numpy())

        if self.opt.color is None:
            img_lq = img2tensor(img_lq, from_bgr=False, float32=True)
        else:
            img_lq = luma_tensor(img_lq, self.opt.color)
        assert isinstance(img_lq, Tensor)
        # normalize
        if self.mean is not None and self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        result: DataFeed = {"lq": img_lq, "lq_path": lq_path}
        if encoded is not None:
            result.update(lq_db=self.storage[index], lq_encoded=encoded)
            if self.opt.phase != "train":
                result["lq_key"] = self.paths[index]
        return result

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def label(self) -> str:
        return "images"
