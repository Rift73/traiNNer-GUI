from collections.abc import Iterable
from typing import TYPE_CHECKING, NotRequired, TypedDict

from torch import Tensor
from torch.optim.optimizer import StateDict
from torch.utils.data import Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from trainner_config.options import PixelFormat as ConfigPixelFormat

if TYPE_CHECKING:
    from traiNNer.data.dvd_response import DvdSample


class DetectionTarget(TypedDict):
    boxes: Tensor
    labels: Tensor
    ignore: Tensor


class DataFeed(TypedDict):
    features: NotRequired[Tensor]
    targets: NotRequired[Tensor]
    row: NotRequired[int | Tensor]
    context_paths: NotRequired[list[str]]
    lq_encoded: NotRequired[bytes]
    lq_db: NotRequired[str]
    lq_key: NotRequired[str]
    gt_db: NotRequired[str]
    gt_key: NotRequired[str]
    lq: NotRequired[Tensor]
    lq_path: NotRequired[str]
    class_name: NotRequired[str | list[str]]
    detection_target: NotRequired[DetectionTarget]
    detection_image_id: NotRequired[int]
    detection_image_ids: NotRequired[list[int]]
    detection_images: NotRequired[list[Tensor]]
    detection_targets: NotRequired[list[DetectionTarget]]
    detection_paths: NotRequired[list[str]]
    gt: NotRequired[Tensor]
    gt_path: NotRequired[str]
    gt_y: NotRequired[Tensor]
    gt_u: NotRequired[Tensor]
    gt_v: NotRequired[Tensor]
    gt_geometry: NotRequired[Tensor]
    shot_ids: NotRequired[Tensor]
    kernel1: NotRequired[Tensor]
    kernel2: NotRequired[Tensor]
    sinc_kernel: NotRequired[Tensor]
    target_gt: NotRequired[Tensor]
    target_gt_path: NotRequired[str]
    otf_padded_height: NotRequired[bool | Tensor]
    otf_padded_width: NotRequired[bool | Tensor]
    # The flips and rotation of a sample sent as source bytes (still_source).
    gt_orientation: NotRequired[int | Tensor]
    # GT was normalized from decoded uint8 on CPU before the final spatial crop.
    normalized_byte_gt: NotRequired[bool | Tensor]
    # A sample's DVD geometry and window; a list of them once collated.
    dvd: NotRequired["DvdSample | list[DvdSample]"]


class OptimizerIdentity(TypedDict):
    type: str
    # Parameter names and shapes in optimizer group order.
    groups: list[list[tuple[str, tuple[int, ...]]]]


class TrainingState(TypedDict):
    epoch: int
    iter: int
    optimizers: list[StateDict]
    optimizer_identities: NotRequired[list[OptimizerIdentity]]
    schedule_free_training_parameters: NotRequired[list[dict[int, Tensor]]]
    resume_recipe: NotRequired[dict]
    schedulers: list[StateDict]
    scheduler_specs: NotRequired[list[dict]]
    early_stopping: NotRequired[dict]
    scheduler_pending: NotRequired[list[int]]
    scaler_g: NotRequired[StateDict]
    scaler_d: NotRequired[StateDict]
    scaler_ae: NotRequired[StateDict]
    random_streams: NotRequired[dict]
    ema_step: NotRequired[Tensor]
    adaptive_d_loss_ema: NotRequired[Tensor]


class DataLoaderArgs(TypedDict):
    dataset: Dataset
    batch_size: NotRequired[int | None]
    shuffle: NotRequired[bool | None]
    sampler: NotRequired[Sampler | Iterable | None]
    batch_sampler: NotRequired[Sampler[list] | Iterable[list]]
    num_workers: NotRequired[int]
    collate_fn: NotRequired[_collate_fn_t | None]
    pin_memory: NotRequired[bool]
    drop_last: NotRequired[bool]
    timeout: NotRequired[float]
    worker_init_fn: NotRequired[_worker_init_fn_t | None]
    prefetch_factor: NotRequired[int | None]
    persistent_workers: NotRequired[bool]
    pin_memory_device: NotRequired[str]


PixelFormat = ConfigPixelFormat
