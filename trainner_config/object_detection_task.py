"""Lightweight configuration for COCO-box detection."""

from typing import Annotated

from msgspec import Meta, Struct

from trainner_config.objectives import validate_supervised_training
from trainner_config.options import ReduxOptions, StrictStruct
from trainner_config.tasks import TaskActionSpec, TrainingTaskSpec, register_task

DATASET_TYPE = "cocodetectiondataset"
FASTER_RCNN_TYPE = "fasterrcnn_mobilenet_v3_large_320_fpn"


class DetectionCategory(StrictStruct):
    id: Annotated[
        int, Meta(ge=0, description="Category ID in the COCO annotation JSON.")
    ]
    name: Annotated[str, Meta(min_length=1)]


class DetectionOptions(StrictStruct):
    categories: Annotated[
        list[DetectionCategory],
        Meta(
            min_length=1,
            description="COCO categories in foreground label order, starting at 1. Label 0 is background.",
        ),
    ]


def _snapshot_metrics(_opt: ReduxOptions, settings: Struct | None) -> dict[str, str]:
    assert isinstance(settings, DetectionOptions)
    return {
        "AP50": "higher", "recall50": "higher", "false_positives50": "lower",
        **{f"AP50/{category.name}": "higher" for category in settings.categories},
    }


def _validate(opt: ReduxOptions, architecture: str, settings: Struct | None) -> None:
    assert isinstance(settings, DetectionOptions)
    names = [category.name for category in settings.categories]
    ids = [category.id for category in settings.categories]
    if (
        any(not name.strip() for name in names)
        or len(set(names)) != len(names)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(
            "task_options.categories requires unique IDs and distinct non-empty names."
        )
    if (
        opt.scale != 1
        or opt.input_pixel_format != "rgb"
        or opt.output_pixel_format != "rgb"
    ):
        raise ValueError("Object detection takes whole RGB images at scale 1.")
    if opt.dist or opt.num_gpu not in (0, 1, "auto"):
        raise ValueError("Object detection supports one device. Set num_gpu to 0 or 1.")
    if opt.use_amp or opt.use_compile:
        raise ValueError(
            "Object detection currently uses eager FP32. Disable use_amp and use_compile."
        )
    if opt.network_d is not None or opt.network_g_teacher is not None:
        raise ValueError("Object detection uses only network_g.")

    if architecture == FASTER_RCNN_TYPE:
        assert opt.network_g is not None
        classes = opt.network_g.get("num_classes")
        if type(classes) is not int or classes != len(ids) + 1:
            raise ValueError(
                "network_g.num_classes must include all categories plus background."
            )
        minimum = opt.network_g.get("min_size", 320)
        maximum = opt.network_g.get("max_size", 640)
        if (
            type(minimum) is not int
            or type(maximum) is not int
            or not 1 <= minimum <= maximum
        ):
            raise ValueError(
                "Detection network_g requires positive min_size and max_size with min_size <= max_size."
            )
        score_threshold = opt.network_g.get("box_score_thresh", 0.05)
        if type(score_threshold) not in (int, float) or not 0 <= score_threshold <= 1:
            raise ValueError("network_g.box_score_thresh must be between 0 and 1.")

    train = opt.train
    if opt.is_train and (
        train is None or train.optim_g is None or "train" not in opt.datasets
    ):
        raise ValueError(
            "Object detection training requires datasets.train and train.optim_g."
        )
    if train is not None:
        validate_supervised_training(train, "the detection network losses")
        schedule = train.scheduler or (train.schedulers or {}).get("g")
        monitor = schedule.get("monitor") if schedule else None
        metrics = {
            "AP50",
            "recall50",
            "false_positives50",
            "targets",
            *(f"AP50/{name}" for name in names),
        }
        if train.early_stopping and train.early_stopping.monitor.metric not in metrics:
            raise ValueError(f"train.early_stopping.monitor.metric must be one of {sorted(metrics)}.")
        if isinstance(monitor, dict) and monitor.get("metric") not in metrics:
            raise ValueError(
                "The detection scheduler monitor must name AP50, recall50, false_positives50, targets, or AP50/<class>."
            )

    for phase, dataset in opt.datasets.items():
        if dataset.type.lower() != DATASET_TYPE:
            raise ValueError(
                f"Object detection requires CocoDetectionDataset for datasets.{phase}."
            )
        roots = dataset.dataroot_lq
        if not roots or (isinstance(roots, list) and (len(roots) != 1 or not roots[0])):
            raise ValueError(
                f"datasets.{phase}.dataroot_lq must name one image directory."
            )
        if not dataset.annotation_file:
            raise ValueError(
                f"datasets.{phase}.annotation_file must name the COCO JSON."
            )
        if (
            dataset.gt_size is not None
            or dataset.lq_size is not None
            or dataset.clip_size is not None
        ):
            raise ValueError(
                f"Remove crop and clip sizes from datasets.{phase}. Detection uses whole images."
            )
        if (
            dataset.downsample_input
            or dataset.mean is not None
            or dataset.std is not None
            or dataset.color
        ):
            raise ValueError(
                "CocoDetectionDataset uses [0, 1] RGB in original image geometry."
            )
        if dataset.prefetch_mode == "cuda":
            raise ValueError(
                "Object detection uses default or CPU prefetch. Its provider transfers variable targets."
            )

    val = opt.val
    if (
        val is not None
        and val.val_enabled
        and (val.save_img or val.tile_size or val.metrics)
    ):
        raise ValueError(
            "Object detection uses built-in AP50. Disable tiled validation and image output, and omit val.metrics."
        )


register_task(
    "object_detection",
    TrainingTaskSpec(
        "network_g",
        "traiNNer.tasks.object_detection:ObjectDetectionModel",
        dataset_types=(DATASET_TYPE,),
        dataset_geometry="whole_images",
        recommended_architectures=(FASTER_RCNN_TYPE,),
        options_type=DetectionOptions,
        validate=_validate,
        otf_degradation="unsupported",
        actions={
            "train": TaskActionSpec(("coco_boxes",)),
            "validate": TaskActionSpec(("coco_boxes",), snapshot_metrics=_snapshot_metrics),
            "holdout": TaskActionSpec(("coco_boxes",), "traiNNer.evaluation:evaluate", optional_exports=("saveDetections", "saveDetectionOperatingPoints")),
            "predict": TaskActionSpec(
                ("images",), "traiNNer.tasks.predict:_predict_images", "json"
            ),
            "export": TaskActionSpec(
                ("images",), "traiNNer.tasks.object_detection.onnx:export_detection"
            ),
        },
    ),
)
