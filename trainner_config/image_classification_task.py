"""Configuration and catalog metadata for whole-image classification."""

from typing import Annotated

from msgspec import Meta, Struct

from trainner_config.objectives import validate_supervised_training
from trainner_config.options import ReduxOptions, StrictStruct
from trainner_config.tasks import TaskActionSpec, TrainingTaskSpec, register_task

DATASET_TYPE = "classfolderdataset"
MOBILENET_TYPE = "mobilenet_v3_small"


class ClassificationOptions(StrictStruct):
    class_names: Annotated[
        list[str],
        Meta(
            min_length=2,
            description="Class folder names in output ID order, starting at 0.",
        ),
    ]
    image_size: Annotated[
        int, Meta(ge=1, description="Resize the whole RGB image to this square size.")
    ] = 128
    top_k: Annotated[
        list[Annotated[int, Meta(ge=1)]] | None,
        Meta(
            description="Accuracy ranks to report alongside top1. Omit for top5 when there are at least five classes."
        ),
    ] = None

    def accuracy_ranks(self) -> tuple[int, ...]:
        if self.top_k is None:
            return (1, 5) if len(self.class_names) >= 5 else (1,)
        return tuple(sorted({1, *self.top_k}))


def _snapshot_metrics(_opt: ReduxOptions, settings: Struct | None) -> dict[str, str]:
    assert isinstance(settings, ClassificationOptions)
    return {f"top{k}": "higher" for k in settings.accuracy_ranks()}


def _validate(opt: ReduxOptions, architecture: str, settings: Struct | None) -> None:
    assert isinstance(settings, ClassificationOptions)
    names = settings.class_names
    if any(
        not name.strip() or name in (".", "..") or "/" in name or "\\" in name
        for name in names
    ) or len(set(names)) != len(names):
        raise ValueError(
            "task_options.class_names must contain distinct non-empty folder names."
        )
    if settings.accuracy_ranks()[-1] > len(names):
        raise ValueError("task_options.top_k cannot exceed the number of classes.")
    if (
        opt.scale != 1
        or opt.input_pixel_format != "rgb"
        or opt.output_pixel_format != "rgb"
    ):
        raise ValueError("Image classification uses scale 1 and RGB images.")
    if opt.dist or opt.num_gpu not in (0, 1, "auto"):
        raise ValueError(
            "Image classification supports one device. Set num_gpu to 0 or 1."
        )
    if opt.use_amp and not opt.amp_bf16:
        raise ValueError("Image classification supports BF16 autocast. Select amp_bf16 or disable AMP for FP32.")
    if opt.network_d is not None or opt.network_g_teacher is not None:
        raise ValueError("Image classification uses only network_g.")

    train_dataset = opt.datasets.get("train")
    if architecture == MOBILENET_TYPE:
        assert opt.network_g is not None
        classes = opt.network_g.get("num_classes")
        if type(classes) is not int or classes != len(names):
            raise ValueError(
                "network_g.num_classes must match task_options.class_names."
            )
        if (
            opt.is_train
            and train_dataset is not None
            and train_dataset.batch_size_per_gpu == 1
            and settings.image_size <= 32
        ):
            raise ValueError(
                "MobileNetV3 needs at least two images per batch when image_size is 32 or smaller."
            )

    train = opt.train
    if opt.is_train and (
        train is None or train.optim_g is None or train_dataset is None
    ):
        raise ValueError(
            "Image classification training requires datasets.train and train.optim_g."
        )
    if train is not None:
        validate_supervised_training(train, "cross-entropy")
        schedule = train.scheduler or (train.schedulers or {}).get("g")
        monitor = schedule.get("monitor") if schedule else None
        metrics = {
            f"top{k}" for k in settings.accuracy_ranks()
        }
        if train.early_stopping and train.early_stopping.monitor.metric not in metrics:
            raise ValueError(f"train.early_stopping.monitor.metric must be one of {sorted(metrics)}.")
        if isinstance(monitor, dict) and monitor.get("metric") not in metrics:
            raise ValueError(
                "The classification scheduler monitor must name an enabled top-k accuracy metric."
            )

    for phase, dataset in opt.datasets.items():
        if dataset.batch_size_per_gpu is not None and dataset.batch_size_per_gpu < 1:
            raise ValueError(f"datasets.{phase}.batch_size_per_gpu must be positive.")
        if dataset.type.lower() != DATASET_TYPE:
            raise ValueError(
                f"Image classification requires ClassFolderDataset for datasets.{phase}."
            )
        sizes = {
            size for size in (dataset.lq_size, dataset.gt_size) if size is not None
        }
        if sizes and sizes != {settings.image_size}:
            raise ValueError(
                f"datasets.{phase}.lq_size or gt_size must match task_options.image_size."
            )
        if (
            dataset.downsample_input
            or dataset.mean is not None
            or dataset.std is not None
            or dataset.color
        ):
            raise ValueError(
                "ClassFolderDataset reads [0, 1] RGB with a whole-image resize."
            )

    val = opt.val
    if (
        val is not None
        and val.val_enabled
        and (val.save_img or val.tile_size or val.metrics)
    ):
        raise ValueError(
            "Image classification has built-in top-k metrics. Disable tiled validation and validation image output, and omit val.metrics."
        )


register_task(
    "image_classification",
    TrainingTaskSpec(
        "network_g",
        "traiNNer.tasks.image_classification:ImageClassificationModel",
        dataset_types=(DATASET_TYPE,),
        dataset_geometry="whole_images",
        recommended_architectures=(MOBILENET_TYPE,),
        options_type=ClassificationOptions,
        validate=_validate,
        otf_degradation="unsupported",
        actions={
            "train": TaskActionSpec(("class_folders",)),
            "validate": TaskActionSpec(("class_folders",), snapshot_metrics=_snapshot_metrics),
            "holdout": TaskActionSpec(
                ("class_folders",), "traiNNer.evaluation:evaluate", optional_exports=("saveClassifications",)
            ),
            "predict": TaskActionSpec(
                ("images",), "traiNNer.tasks.predict:_predict_images", "json"
            ),
            "export": TaskActionSpec(
                ("images",), "traiNNer.tasks.predict:_export_image_onnx"
            ),
        },
    ),
)
