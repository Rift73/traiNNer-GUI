"""Ordered numeric columns for supervised regression on CSV records."""

from typing import Annotated

from msgspec import Meta, Struct
from trainner_config.objectives import validate_supervised_training
from trainner_config.options import ReduxOptions, StrictStruct
from trainner_config.tasks import TaskActionSpec, TrainingTaskSpec, register_task


class RegressionOptions(StrictStruct):
    features: Annotated[
        list[str],
        Meta(min_length=1, description="Input column names in network order."),
    ]
    targets: Annotated[
        list[str],
        Meta(min_length=1, description="Numeric target column names in output order."),
    ]

    def __post_init__(self) -> None:
        columns = [*self.features, *self.targets]
        if any(not name.strip() for name in columns) or len(columns) != len(set(columns)):
            raise ValueError("Feature and target columns must have distinct nonempty names. A target cannot also be an input.")


def _metrics(_opt, _settings):
    return {"mae": "lower", "rmse": "lower"}


def _validate(opt: ReduxOptions, architecture: str, settings: Struct | None) -> None:
    assert isinstance(settings, RegressionOptions)
    if opt.scale != 1 or opt.dist or opt.num_gpu not in (0, 1, "auto"):
        raise ValueError("Tabular regression uses scale 1 and one training device.")
    if (
        opt.input_pixel_format != "rgb"
        or opt.output_pixel_format != "rgb"
        or opt.network_d
        or opt.network_g_teacher
    ):
        raise ValueError(
            "Tabular regression uses network_g and does not apply image pixel adapters."
        )
    if opt.use_amp and not opt.amp_bf16:
        raise ValueError("Tabular regression supports FP32 or BF16 autocast.")
    if architecture == "tabular_mlp":
        net = opt.network_g or {}
        if net.get("in_features") != len(settings.features) or net.get(
            "out_features"
        ) != len(settings.targets):
            raise ValueError(
                "Set network_g.in_features and out_features to the numbers of configured features and targets."
            )
    for key, ds in opt.datasets.items():
        if ds.batch_size_per_gpu is not None and ds.batch_size_per_gpu < 1:
            raise ValueError(f"datasets.{key}.batch_size_per_gpu must be positive.")
        if ds.type.casefold() != "numericcsvdataset" or not ds.meta_info:
            raise ValueError(
                f"datasets.{key} needs NumericCsvDataset and a CSV file in meta_info."
            )
        if (
            ds.io_backend.get("type", "disk") != "disk"
            or ds.dataroot_lq
            or ds.dataroot_gt
        ):
            raise ValueError(
                f"datasets.{key} reads its CSV from meta_info using the disk backend."
            )
        if (
            ds.gt_size
            or ds.lq_size
            or ds.color
            or ds.mean is not None
            or ds.std is not None
        ):
            raise ValueError(
                "NumericCsvDataset reads numeric columns in the units stored in each CSV."
            )
    if opt.is_train and (
        not opt.train or not opt.train.optim_g or "train" not in opt.datasets
    ):
        raise ValueError("Tabular training requires datasets.train and train.optim_g.")
    if opt.train:
        train = opt.train
        validate_supervised_training(train, "mean squared error")
        if train.grad_clip == "spike":
            raise ValueError(
                "Select fixed-norm clipping or disable clipping for tabular regression."
            )
        if train.early_stopping and train.early_stopping.monitor.metric not in _metrics(
            opt, settings
        ):
            raise ValueError("Use mae or rmse for tabular early stopping.")
    if opt.val and (opt.val.save_img or opt.val.tile_size or opt.val.metrics):
        raise ValueError(
            "Tabular validation reports mae and rmse. Disable image output and omit image metrics."
        )


register_task(
    "tabular_regression",
    TrainingTaskSpec(
        "network_g",
        "traiNNer.tasks.tabular:TabularModel",
        dataset_types=("numericcsvdataset",),
        dataset_geometry="records",
        recommended_architectures=("tabular_mlp",),
        options_type=RegressionOptions,
        validate=_validate,
        otf_degradation="unsupported",
        actions={
            "train": TaskActionSpec(("records",)),
            "validate": TaskActionSpec(("records",), snapshot_metrics=_metrics),
            "holdout": TaskActionSpec(
                ("records",), "traiNNer.evaluation:evaluate", "provider_scores"
            ),
            "predict": TaskActionSpec(
                ("records",), "traiNNer.tasks.tabular:predict_records", "json"
            ),
            "export": TaskActionSpec(
                ("records",), "traiNNer.tasks.tabular:export_records"
            ),
        },
    ),
)
