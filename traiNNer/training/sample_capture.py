"""Capture one sample actually installed on the model at an explicitly requested update."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import Tensor

from trainner_config.tasks import task_name


def _sample_tensors(model):
    task = task_name(model.opt)
    extra = {}
    if task == "object_detection":
        image, target = model.images[0], model.targets[0]
        keep = ~target["ignore"]
        values = {"input":image, "boxes":target["boxes"][keep], "labels":target["labels"][keep]}
        extra.update(classNames=list(model.class_names), categoryIds=list(model.category_ids),
                     boxFormat="xyxy, input pixels", ignoredAnnotations=int((~keep).sum()))
    elif task == "image_classification":
        values = {"input":model.image[0], "label":model.labels[0]}
        extra.update(classNames=list(model.class_names), classId=int(model.labels[0]))
    elif task == "semantic_segmentation":
        values = {"input":model.image[0], "mask":model.mask[0]}
        extra.update(classNames=list(model.class_names), ignoreLabel=model.ignore_label)
    elif task == "tabular_regression":
        values = {"input": model.features[0], "target": model.targets[0]}
        extra.update(featureColumns=model.columns.features, targetColumns=model.columns.targets)
    elif task in ("image_restoration","recurrent_restoration","autoencoder"):
        values = {name:value[0] for name,value in (("input",model.lq),("target",model.gt))
                  if isinstance(value,Tensor)}
        if task == "autoencoder":
            extra["reconstructionSource"] = "target"
        shot_ids = getattr(model, "shot_ids", None)
        if isinstance(shot_ids, Tensor):
            values["shotIds"] = shot_ids[0]
    else:
        raise ValueError(f"Training sample capture does not have a tensor contract for {task!r}.")
    return task, values, extra


class TrainingSampleRecorder:
    def __init__(self, opt):
        if task_name(opt) not in ("image_restoration", "recurrent_restoration", "autoencoder",
                                  "image_classification", "semantic_segmentation", "object_detection", "tabular_regression"):
            raise ValueError(f"Training sample capture does not have a tensor contract for {task_name(opt)!r}.")
        self.remaining = set(opt.logger.training_sample_steps or ())
        self.directory = Path(opt.path.experiments_root) / "training-samples"
        self.attempt = uuid4().hex
        managed = os.environ.get("TRAINNER_STUDIO_ATTEMPT")
        self.managed = Path(managed).name if managed else None

    def record(self, model, batch, update: int, epoch: int, batch_in_epoch: int):
        """Call only after the model has installed its prepared microbatch."""
        task, values, extra = _sample_tensors(model)
        if not values:
            raise ValueError(f"The {task} model has no prepared input to capture.")
        self.directory.mkdir(parents=True, exist_ok=True)
        key = f"{self.attempt}-{update}"
        with TemporaryDirectory(dir=self.directory,prefix=".capturing-") as tmp:
            stage = Path(tmp)
            host = {name: value.detach().to(device="cpu", copy=True).contiguous()
                    for name,value in values.items()}
            save_file(host,str(stage/"sample.safetensors"))
            details = {}
            for name, value in host.items():
                detail = {"shape":list(value.shape), "dtype":str(value.dtype).removeprefix("torch.")}
                details[name] = detail
                if name in ("input","target"):
                    if value.ndim == 4:
                        center = value.shape[0] // 2
                        detail["previewFrame"] = center
                        value = value[center]
                    if value.ndim == 3 and value.shape[0] in (1,3):
                        # PNG is a display view. Preserve model scale, dtype and
                        # out-of-range values in the downloadable tensor file.
                        image = value.float().clamp(0,1).mul(255).round().byte()
                        pixels = image[0] if image.shape[0]==1 else image.permute(1,2,0)
                        Image.fromarray(pixels.numpy()).save(stage/f"{name}.png")
                        detail["preview"] = f"{name}.png"
                elif name == "mask" and value.ndim == 2 and value.numel():
                    lo, hi = int(value.min()), int(value.max())
                    if 0 <= lo <= hi <= 65535:
                        dtype = np.uint8 if hi <= 255 else np.uint16
                        Image.fromarray(value.numpy().astype(dtype)).save(stage/"mask.png")
                        detail["preview"] = "mask.png"
            record = {"id":key, "attempt":self.managed, "candidateStep":update, "epoch":epoch,
                      "batchInEpoch":batch_in_epoch, "sampleIndex":0, "rank":0, "task":task,
                      "stage":"afterFeedBeforeOptimize", "tensors":details, **extra}
            if task == "tabular_regression":
                record.update(sourceCsv=str(Path(model.opt.datasets["train"].meta_info).expanduser().absolute()),
                              sourceRecord=int(batch["row"][0]) + 1)
            if not model.opt.high_order_degradation and not getattr(model,"batch_augment",None):
                for name in ("lq_path","gt_path"):
                    paths = batch.get(name)
                    if isinstance(paths,(tuple,list)) and paths and isinstance(paths[0],str):
                        record[name] = paths[0]
            (stage/"capture.json").write_text(json.dumps(record,allow_nan=False),encoding="utf-8")
            # Both previews and the raw values become visible in one rename.
            stage.rename(self.directory/key)
        self.remaining.remove(update)
