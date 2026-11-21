"""Dataset validation, metric accumulation, and image export for SRModel."""

from __future__ import annotations

# Validation uses the model's existing metric hooks and shared result state.
# ruff: noqa: SLF001
import os
import shutil
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from os import path as osp
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.metrics import (
    LEARNED_TENSOR_METRICS, calculate_learned_metric_pt, calculate_metric,
    quantize_metric_tensor, requires_reference,
)
from traiNNer.metrics.psnr_ssim import calculate_psnr_pt, calculate_ssim_pt
from traiNNer.utils import get_root_logger, imwrite, tensor2img
from traiNNer.utils.logger import clickable_file_path

if TYPE_CHECKING:
    from torch.utils.tensorboard.writer import SummaryWriter

    from traiNNer.models.ae_model import AEModel
    from traiNNer.models.sr_model import SRModel
    from traiNNer.utils.redux_options import DatasetOptions, ReduxOptions


def _split_metric_configs(
    metrics: dict[str, dict[str, Any]],
) -> tuple[dict[str, tuple[str, int, bool]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Keep built-in tensor metrics separate from the numpy fallback path."""
    tensor_metrics: dict[str, tuple[str, int, bool]] = {}
    learned: dict[str, dict[str, Any]] = {}
    fallback_metrics: dict[str, dict[str, Any]] = {}
    for name, options in metrics.items():
        metric_type = options.get("type")
        if metric_type in ("calculate_psnr", "calculate_ssim", "calculate_neosr_psnr", "calculate_neosr_ssim"):
            tensor_metrics[name] = (
                metric_type.removeprefix("calculate_"),
                int(options.get("crop_border", 4 if "_neosr_" in metric_type else 0)),
                bool(options.get("test_y_channel", False)),
            )
        elif metric_type in LEARNED_TENSOR_METRICS:
            learned[name] = options
        else:
            fallback_metrics[name] = options
    return tensor_metrics, learned, fallback_metrics


def _tensor_metric_values(
    output: Tensor,
    gt: Tensor | None,
    metrics: dict[str, tuple[str, int, bool]],
    learned: dict[str, dict[str, Any]],
    *, streamed_cpu: bool = False,
) -> Iterator[tuple[str, Tensor]]:
    """Quantize like tensor2img, retaining results on-device until dataset end."""
    if gt is None or not metrics:
        if gt is None:
            learned = {name: opts for name, opts in learned.items() if opts["type"] == "calculate_topiq_nr"}
        if learned:
            source = quantize_metric_tensor(output)
            reference = quantize_metric_tensor(gt) if gt is not None and any(
                opts["type"] != "calculate_topiq_nr" for opts in learned.values()
            ) else None
            for name, options in learned.items():
                yield name, calculate_learned_metric_pt(source, reference, options)
        return
    grid = output.detach().float().clamp(0, 1).mul_(255).round_()
    target_grid = gt.detach().float().clamp(0, 1).mul_(255).round_()
    normalized = None
    keep_grid = bool(learned) or any(kind.startswith("neosr_") for kind, _, _ in metrics.values())
    for name, (kind, crop_border, test_y_channel) in metrics.items():
        if kind.startswith("neosr_"):
            from traiNNer.metrics.neosr_pixels import score_grid
            value = score_grid(kind.removeprefix("neosr_"), grid, target_grid, crop_border, test_y_channel,
                               streamed_cpu=streamed_cpu)
            yield name, value.mean().detach()
            continue
        if normalized is None:
            normalized = (grid / 255.0, target_grid / 255.0) if keep_grid else (grid.div_(255.0), target_grid.div_(255.0))
        output_metric_tensor, gt_metric_tensor = normalized
        if kind == "psnr":
            value = calculate_psnr_pt(
                output_metric_tensor,
                gt_metric_tensor,
                crop_border=crop_border,
                test_y_channel=test_y_channel,
                streamed_cpu=streamed_cpu,
            )
        else:
            value = calculate_ssim_pt(
                output_metric_tensor,
                gt_metric_tensor,
                crop_border=crop_border,
                test_y_channel=test_y_channel,
                streamed_cpu=streamed_cpu,
            )
        # Calling item() here would serialize each image with the GPU.
        yield name, value.mean().detach()
    if learned:
        # PNG conversion maps NaN to black. This conversion follows the pixel
        # reductions so those keep their original non-finite semantics.
        divisor = grid.new_full((), 255.0)
        source = grid.nan_to_num_().div_(divisor)
        reference = target_grid.nan_to_num_().div_(divisor)
        for name, options in learned.items():
            yield name, calculate_learned_metric_pt(source, reference, options)


def _validation_image_paths(
    opt: ReduxOptions,
    dataset_opt: DatasetOptions,
    relative: Path,
    current_iter: int,
    multi_val_datasets: bool,
    source_kind: str,
    input_files: set[Path],
) -> tuple[str, str | None]:
    """Return the output PNG and optional first-seen input preview PNG."""
    assert opt.val is not None
    assert opt.path.visualization is not None
    root = Path(opt.path.visualization)
    dataset = dataset_opt.name
    if not opt.is_train or multi_val_datasets:
        if not dataset or Path(dataset).name != dataset or dataset in (".", ".."):
            raise ValueError(f"Validation dataset name cannot be used as a folder: {dataset!r}.")
    stem = relative.name
    preview = None
    if opt.is_train:
        if multi_val_datasets:
            head = f"{dataset} - {relative.parts[0]}"
            # Keep a longer dataset name's namespace clear, e.g. "Val - Night".
            longer_names = (d.name for phase, d in opt.datasets.items()
                            if phase.split("_")[0] in ("val", "test") and d.name.startswith(dataset + " - "))
            if any(head == other or head.startswith(other + " - ") for other in longer_names):
                relative = Path(dataset) / relative
            else:
                relative = Path(head, *relative.parts[1:])
        output = relative / f"{stem}_{current_iter:06d}.png"
        preview = relative / f"{stem}_{'gt' if source_kind == 'gt' else 'lr'}.png"
    else:
        suffix = opt.val.suffix
        if suffix and (Path(suffix).name != suffix or suffix in (".", "..")):
            raise ValueError("The evaluation filename suffix must not contain a directory.")
        filename = f"{stem}_{suffix}.png" if suffix else f"{stem}.png"
        output = Path(dataset) / relative.parent / filename

    resolved_root = root.resolve()
    if root.is_symlink():
        raise ValueError(f"Validation visualization folder is a symlink: {root}.")
    for target in (output, preview):
        if target is None:
            continue
        destination = (root / target).resolve()
        if target.is_absolute() or ".." in target.parts or destination != resolved_root / target:
            raise ValueError(f"Validation export has an unsafe destination: {target}.")
        if destination in input_files:
            raise ValueError(f"Validation export would overwrite a selected input: {destination}.")
    return str(root / output), str(root / preview) if preview is not None else None


class _ImageWriter:
    """Overlap encoding with inference without retaining a dataset of decoded images."""

    def __init__(self, pool: ThreadPoolExecutor, slots: int):
        self.pool = pool
        self.slots = slots
        self.pending: deque[tuple[Future, int]] = deque()
        self.queued_bytes = 0

    def finish_one(self) -> None:
        future, size = self.pending.popleft()
        self.queued_bytes -= size
        future.result()

    def submit(self, size: int, action, *args) -> None:
        while self.pending and (len(self.pending) >= self.slots or self.queued_bytes + size > 128 * 1024**2):
            self.finish_one()
        self.pending.append((self.pool.submit(action, *args), size))
        self.queued_bytes += size


def run_validation(
    model: SRModel | AEModel,
    dataloader: DataLoader,
    current_iter: int,
    tb_logger: SummaryWriter | None,
    save_img: bool,
    multi_val_datasets: bool,
    *,
    reconstruction: bool = False,
) -> None:
    """Score restoration or autoencoder reconstruction against the observed reference."""
    from traiNNer.metrics.session import metric_session
    from traiNNer.tasks.temporal_results import dataset_sink

    args = (model, dataloader, current_iter, tb_logger, multi_val_datasets, reconstruction)
    assert model.opt.val is not None
    assert isinstance(dataloader.dataset, BaseDataset)
    cpu_stitch = not reconstruction and model.opt.val.tile_size > 0 and model.opt.val.tile_mode == "context"
    metric_device = torch.device("cpu") if cpu_stitch else model.device
    with metric_session(metric_device, cache_accelerator=model.opt.val.cache_metric_models_on_device), dataset_sink(dataloader.dataset.opt.name) as temporal:
        if not save_img:
            _evaluate(*args, writer=None, temporal=temporal, cpu_stitch=cpu_stitch)
            return
        workers = max(1, min(8, (os.cpu_count() or 1) // 2))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="valsave") as pool:
            _evaluate(*args, writer=_ImageWriter(pool, workers * 2), temporal=temporal, cpu_stitch=cpu_stitch)


def _evaluate(
    model: SRModel | AEModel,
    dataloader: DataLoader,
    current_iter: int,
    tb_logger: SummaryWriter | None,
    multi_val_datasets: bool,
    reconstruction: bool,
    *,
    writer: _ImageWriter | None,
    temporal,
    cpu_stitch: bool,
) -> None:
    save_img = writer is not None
    assert isinstance(dataloader.dataset, BaseDataset)
    assert model.opt.val is not None
    assert model.opt.path.visualization is not None

    dataset_name = dataloader.dataset.opt.name
    output_channels = set()

    # Reconstruction always scores the encoded image against itself. Paired
    # datasets supply the high-resolution target, single images supply LQ.
    source_kind = "gt" if reconstruction and dataloader.dataset.opt.dataroot_gt else "lq"
    source_key = source_kind + "_path"
    scores = dict.fromkeys(model.opt.val.metrics or {}, 0.0)
    counts = dict.fromkeys(scores, 0)
    if model.with_metrics:
        model._initialize_best_metric_results(dataset_name)
    pbar = None
    if model.use_pbar:
        pbar = tqdm(total=len(dataloader), unit="image")

    logger = get_root_logger()
    if save_img and len(dataloader) > 0:
        logger.info(
            "Saving %d validation images to %s.",
            len(dataloader),
            clickable_file_path(model.opt.path.visualization, "visualization folder"),
        )

    run_metrics = model.with_metrics
    metrics_cfg = model.opt.val.metrics if run_metrics else None
    image_manifest = None
    directory = model.opt.path.experiments_root if model.opt.is_train else model.opt.path.results_root
    if save_img and directory and (model.opt.is_train or model.opt.val.record_samples):
        from traiNNer.training.validation_images import ValidationImages
        image_manifest = ValidationImages(
            Path(directory), dataloader.dataset.opt, current_iter, source_kind
        )

    # Fast path for built-in PSNR/SSIM: compute directly on tensors to avoid
    # expensive tensor->numpy conversions per validation image.
    tensor_metric_cfg: dict[str, tuple[str, int, bool]] = {}
    learned_metric_cfg: dict[str, dict[str, Any]] = {}
    fallback_metric_cfg: dict[str, dict[str, Any]] = {}
    pending_tensor_metrics: list[tuple[str, Tensor]] = []
    if run_metrics and metrics_cfg is not None:
        tensor_metric_cfg, learned_metric_cfg, fallback_metric_cfg = _split_metric_configs(metrics_cfg)

        if tensor_metric_cfg or learned_metric_cfg:
            logger.info(
                "Validation metric fast path enabled for %s.",
                ", ".join((*tensor_metric_cfg, *learned_metric_cfg)),
            )

    reference_metrics = {
        name for name, options in fallback_metric_cfg.items() if requires_reference(options["type"])
    }

    records = None
    if model.opt.val.record_samples and run_metrics:
        from traiNNer.training.validation_records import ValidationRecords
        tile = 0 if reconstruction else model.opt.val.tile_size
        inference = {
            "device": model.device.type,
            "precision": str(model.amp_dtype).removeprefix("torch.") if getattr(model, "validation_amp", model.use_amp) else "float32",
            "tileSize": tile,
            "tileOverlap": model.opt.val.tile_overlap if tile else None,
        }
        if cpu_stitch:
            inference.update(tileMode="context", stitchDevice="cpu", metricDevice="cpu")
        records = ValidationRecords(model.opt, dataloader.dataset.opt, current_iter,
                                    set(tensor_metric_cfg), source_kind, inference=inference)

    names = None
    roots = ()
    input_files = set()
    if save_img:
        from traiNNer.training.validation_images import (
            relative_image, selected_files_in_output, selected_image_names, source_roots,
        )

        roots = source_roots(dataloader.dataset.opt, source_kind)
        input_files = selected_files_in_output(dataloader.dataset, Path(model.opt.path.visualization))
        names = selected_image_names(dataloader.dataset, source_kind, roots,
                                     flat=not model.opt.is_train or multi_val_datasets)
    export_names: set[str] = set()
    for val_data in dataloader:
        sample = records.record_sample(val_data) if records is not None else None
        if source_key not in val_data:
            raise ValueError(f"Validation dataset {dataset_name!r} must supply {source_key}.")
        img_name = osp.splitext(osp.basename(val_data[source_key][0]))[0]
        model.gt = None
        if cpu_stitch and "gt" in val_data:
            # Only the LQ input goes through the network. Keep the decoded HR
            # reference beside the stitched result for pixel adapters and metrics.
            feed = val_data.copy()
            reference = feed.pop("gt")
            model.feed_data(feed)
            model.gt = reference.cpu()
        else:
            model.feed_data(val_data)
        if reconstruction and source_kind == "lq":
            model.gt = model.lq
        model.test()

        output_tensor = model.output_gt if reconstruction else model.output
        assert output_tensor is not None
        if not model.opt.is_train:
            output_channels.add(output_tensor.shape[-3])
        gt_tensor = model.gt
        if temporal is not None:
            temporal(val_data, output_tensor, gt_tensor)
        sr_img = None

        # Fast tensor metrics (PSNR/SSIM).
        if run_metrics and (tensor_metric_cfg or learned_metric_cfg):
            observed = 0
            for name, value in _tensor_metric_values(
                output_tensor, gt_tensor, tensor_metric_cfg, learned_metric_cfg,
                streamed_cpu=(cpu_stitch or not model.opt.is_train) and output_tensor.device.type == "cpu",
            ):
                counts[name] += 1
                observed += 1
                pending_tensor_metrics.append((name, value))
            if records is not None:
                records.expect_tensor_scores(observed)

        # Fallback metrics keep existing numpy path.
        if run_metrics and fallback_metric_cfg:
            sr_img = tensor2img(output_tensor, to_bgr=False)
            metric_data = {"img": sr_img}
            if gt_tensor is not None:
                metric_data["img2"] = tensor2img(gt_tensor, to_bgr=False)
            for name, opt_ in fallback_metric_cfg.items():
                if gt_tensor is None and name in reference_metrics:
                    continue
                result = calculate_metric(metric_data, opt_, output_tensor.device)
                scores[name] += result
                counts[name] += 1
                if sample is not None:
                    sample.scores[name] = float(result)

        # Release per-image references; keep the CUDA allocator cache hot so
        # subsequent images can reuse those allocations without a device sync.
        source_tensor = None
        preview_path = None
        if save_img:
            source_path = val_data[source_key][0]
            relative = names[source_path] if names is not None else relative_image(source_path, roots).with_suffix("")
            save_img_path, preview_path = _validation_image_paths(
                model.opt, dataloader.dataset.opt, relative, current_iter, multi_val_datasets, source_kind, input_files,
            )
            if save_img_path in export_names:
                raise ValueError(f"Two validation inputs map to {save_img_path}. Select distinct source members.")
            export_names.add(save_img_path)
            if preview_path is not None and not os.path.exists(preview_path):
                source_tensor = gt_tensor if reconstruction else model.lq
        if source_tensor is not None and source_tensor.ndim == 5:
            source_tensor = source_tensor[:, source_tensor.size(1) // 2]
        model.lq = None
        model.gt = None
        if reconstruction:
            model.output_lq = model.output_gt = None
        else:
            model.output = None

        if save_img:
            if sr_img is None:
                sr_img = tensor2img(output_tensor, to_bgr=False)
            if image_manifest is not None:
                height, width = sr_img.shape[:2]
                image_manifest.add(val_data, save_img_path, width, height)
            if source_tensor is not None:
                Path(save_img_path).parent.mkdir(parents=True, exist_ok=True)
            sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR) if sr_img.ndim == 3 else sr_img
            assert writer is not None
            writer.submit(sr_img_bgr.nbytes, imwrite, sr_img_bgr, save_img_path)
            if source_tensor is not None:
                assert preview_path is not None
                original = val_data.get(source_kind + "_encoded")
                source = Path(val_data[source_key][0])
                preserve_source_bytes = not dataloader.dataset.opt.color
                if preserve_source_bytes and original and original[0].startswith(b"\x89PNG\r\n\x1a\n"):
                    save = Path(preview_path).write_bytes
                    arguments = (original[0],)
                elif (preserve_source_bytes and not val_data.get(source_kind + "_db")
                      and source.suffix.lower() == ".png" and source.is_file()):
                    save = shutil.copy
                    arguments = (str(source), preview_path)
                else:
                    save = imwrite
                    arguments = (tensor2img(source_tensor, to_bgr=source_tensor.shape[-3] != 1), preview_path)
                size = len(arguments[0]) if isinstance(arguments[0], bytes) else getattr(arguments[0], "nbytes", 0)
                writer.submit(size, save, *arguments)

        if pbar is not None:
            pbar.update(1)
            pbar.set_description(f"Test {img_name}")
    if pbar is not None:
        pbar.close()

    if pending_tensor_metrics:
        # Preserve the original per-image floating-point values and Python
        # summation order while paying for one GPU-to-CPU synchronization.
        host_metric_values = (
            torch.stack([value for _, value in pending_tensor_metrics]).cpu().tolist()
        )
        for (metric_name, _), metric_value in zip(
            pending_tensor_metrics, host_metric_values, strict=True
        ):
            scores[metric_name] += float(metric_value)
        if records is not None:
            for sample_index, (name, _), value in zip(
                records.tensor_sample_indices, pending_tensor_metrics, host_metric_values, strict=True
            ):
                records.samples[sample_index].scores[name] = float(value)

    if writer is not None:
        while writer.pending:
            writer.finish_one()
        if image_manifest is not None:
            image_manifest.publish(records.capture_id if records is not None else None)

    if run_metrics:
        model.metric_results = {name: float(value) / counts[name] for name, value in scores.items() if counts[name]}
        for name, value in model.metric_results.items():
            model._update_best_metric_result(dataset_name, name, value, current_iter)
        missing = scores.keys() - model.metric_results.keys()
        if missing:
            logger.info("Validation %s produced no observations for %s.", dataset_name, ", ".join(sorted(missing)))
        if records is not None:
            records.metrics = {name: definition for name, definition in records.metrics.items() if counts[name]}
            records.publish()
            model.validation_sample_record = records
        if model.metric_results:
            model._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    if not model.opt.is_train:
        model.validation_output_channels = output_channels
    model.first_val_completed = True
