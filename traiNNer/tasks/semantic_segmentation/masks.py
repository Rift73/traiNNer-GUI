"""Store predicted label IDs in a lossless grayscale PNG."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from torch import Tensor

from traiNNer.training.validation_images import ValidationImages


def mask_image(labels: Tensor, classes: int) -> Image.Image:
    if not 1 <= classes <= 65536:
        raise ValueError("PNG masks support up to 65536 classes.")
    dtype = np.uint8 if classes <= 256 else np.uint16
    return Image.fromarray(labels.detach().cpu().numpy().astype(dtype, copy=False))


class ValidationMasks:
    def __init__(self, root: Path, visualization: Path, dataset, step: int, classes: int):
        self.capture = uuid4().hex
        self.folder = visualization / "masks" / self.capture
        self.gallery = ValidationImages(root, dataset, step)
        self.classes = classes
        self.count = 0

    def add(self, predictions: Tensor, inputs, targets) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        # A batch makes one device transfer, only when the caller requested masks.
        for labels, source, target in zip(predictions.cpu(), inputs, targets, strict=True):
            image = mask_image(labels, self.classes)
            dest = self.folder / f"{self.count:08d}.png"
            image.save(dest)
            self.gallery.add({"lq_path": [source], "gt_path": [target]}, str(dest), *image.size)
            self.count += 1

    def publish(self) -> None:
        self.gallery.publish(self.capture)
