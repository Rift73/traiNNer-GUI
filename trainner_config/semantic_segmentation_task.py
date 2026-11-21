"""Configuration and catalog metadata for training from image/mask pairs."""

from typing import Annotated

from msgspec import Meta, Struct

from trainner_config.objectives import validate_supervised_training
from trainner_config.options import ReduxOptions, StrictStruct
from trainner_config.tasks import TaskActionSpec, TrainingTaskSpec, register_task

DATASET_TYPE = "imagemaskdataset"
LRASPP_TYPE = "lraspp_mobilenet_v3_large"


class SegmentationOptions(StrictStruct):
    class_names: Annotated[
        list[str],
        Meta(min_length=2, description="Class names in mask ID order, starting at 0."),
    ]
    ignore_label: Annotated[
        int, Meta(description="Mask ID excluded from training loss and evaluation.")
    ] = 255


def _snapshot_metrics(_opt: ReduxOptions, settings: Struct | None) -> dict[str, str]:
    assert isinstance(settings, SegmentationOptions)
    return {"mIoU": "higher", **{f"IoU/{name}": "higher" for name in settings.class_names}}


def _validate(opt: ReduxOptions, architecture: str, settings: Struct | None) -> None:
    assert isinstance(settings, SegmentationOptions)
    names = settings.class_names
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError(
            "task_options.class_names must contain distinct non-empty names."
        )
    if 0 <= settings.ignore_label < len(names):
        raise ValueError(
            "task_options.ignore_label must be outside the class ID range."
        )
    if (
        opt.scale != 1
        or opt.input_pixel_format != "rgb"
        or opt.output_pixel_format != "rgb"
    ):
        raise ValueError(
            "Semantic segmentation uses scale 1 and RGB input and output settings."
        )
    if opt.dist or (isinstance(opt.num_gpu, int) and opt.num_gpu > 1):
        raise ValueError(
            "Semantic segmentation currently uses one device. Set num_gpu to 1 or 0."
        )
    if opt.use_amp and not opt.amp_bf16:
        raise ValueError("Semantic segmentation supports BF16 autocast. Select amp_bf16 or disable AMP for FP32.")
    if opt.network_d is not None or opt.network_g_teacher is not None:
        raise ValueError(
            "Semantic segmentation does not use network_d or network_g_teacher."
        )

    if architecture == LRASPP_TYPE:
        assert opt.network_g is not None
        classes = opt.network_g.get("num_classes")
        if type(classes) is not int or classes != len(names):
            raise ValueError(
                "network_g.num_classes must equal the number of task_options.class_names."
            )

    train = opt.train
    if opt.is_train and (train is None or train.optim_g is None):
        raise ValueError("Semantic segmentation training requires train.optim_g.")
    if train is not None:
        validate_supervised_training(train, "masked cross-entropy")

    for phase, dataset in opt.datasets.items():
        if dataset.type.lower() != DATASET_TYPE:
            raise ValueError(
                f"Semantic segmentation requires ImageMaskDataset for datasets.{phase}."
            )
        if phase == "train" and dataset.accum_iter < 1:
            raise ValueError(
                "datasets.train.accum_iter must be at least 1 for semantic segmentation."
            )
        if dataset.downsample_input:
            raise ValueError(
                "ImageMaskDataset requires pixel-aligned inputs. Disable downsample_input."
            )

    val = opt.val
    if (
        val is not None
        and val.val_enabled
        and (val.tile_size or val.metrics)
    ):
        raise ValueError(
            "Semantic segmentation evaluates full images with class IoU. "
            "Set val.tile_size to 0 and omit val.metrics."
        )
    if val and val.save_img and len(names) > 65536:
        raise ValueError("PNG mask output supports up to 65536 classes.")


register_task(
    "semantic_segmentation",
    TrainingTaskSpec(
        "network_g",
        "traiNNer.tasks.semantic_segmentation:SemanticSegmentationModel",
        dataset_types=(DATASET_TYPE,),
        recommended_architectures=(LRASPP_TYPE,),
        options_type=SegmentationOptions,
        validate=_validate,
        otf_degradation="unsupported",
        actions={
            "train": TaskActionSpec(("image_masks",)),
            "validate": TaskActionSpec(("image_masks",), snapshot_metrics=_snapshot_metrics),
            "holdout": TaskActionSpec(("image_masks",), "traiNNer.evaluation:evaluate", optional_exports=("saveImages",)),
            "predict": TaskActionSpec(
                ("images",), "traiNNer.tasks.predict:_predict_images"
            ),
            "export": TaskActionSpec(
                ("images",), "traiNNer.tasks.predict:_export_image_onnx"
            ),
        },
    ),
)
