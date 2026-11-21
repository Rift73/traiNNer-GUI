"""Registered prediction and single-file ONNX export.

Use a trainer YAML for the task, generator, scale, and pixel formats.
Supply an explicit .safetensors or weights-only .pth checkpoint.

For one input image, supply a .png output file. For a folder, supply a separate
output folder. Relative stems are preserved recursively. RGB results are
8-bit PNGs. Segmentation results are 8- or 16-bit single-channel PNGs containing
class IDs in task_options.class_names order. For classification, supply a .json
output file even for a directory of inputs. It contains one class ID, name, and
ordered raw logits per image. Detection also writes a .json file, with
score-ordered boxes in original image pixels and a COCO category ID per box.
Frame sequence prediction accepts a directory of numbered frames and returns
eligible center frames. Autoencoder prediction reconstructs still images.
Numeric regression takes a UTF-8 CSV with the named feature columns; its JSON
predictions follow the trained target order. Its ONNX input is [batch, features].
Each command emits one JSON result on stdout.

Still-image ONNX export uses a fixed float32 input named input with shape [1, 3, H, W]
and RGB samples in [0, 1] for RGB models. A one-channel still restoration graph
takes an already encoded BT.601 luma plane [1, 1, H, W] and returns luma
[1, 1, H*scale, W*scale] in the same encoding. File prediction encodes the source
image and writes an 8-bit grayscale PNG. RGB restoration returns float32 rgb in [0, 1] with
shape [1, 3, H*scale, W*scale]. Segmentation returns int64 mask of shape
[1, H, W]. Classification returns float32 logits of shape [1, num_classes].
Resize the whole input to task_options.image_size before invoking a classification
ONNX graph. PNG and JSON encoding are separate. Object detection supports
eager CPU prediction with a variable number of boxes per image.

Block-read rsr_compact exports one fixed window with --clip-size T. Its FP32
input is chronological RGB in [0, 1], NTCHW [1, T, 3, H, W]. The graph returns
the unclamped FP32 RGB centre frame at index T // 2, NCHW [1, 3, H*scale, W*scale].
"""

import argparse
import json
import pickle
import sys
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import numpy as np
import torch
from PIL import Image
from safetensors import SafetensorError
from torch import Tensor, nn
from trainner_config.image_classification_task import ClassificationOptions
from trainner_config.object_detection_task import DetectionOptions
from trainner_config.semantic_segmentation_task import SegmentationOptions
from trainner_config.tasks import resolve_task, still_restoration_luma, task_name

from traiNNer.archs import build_network
from traiNNer.archs.arch_info import TEMPORAL_ARCHS
from traiNNer.models.checkpoint_payload import load_matching_network_checkpoint
from traiNNer.models.onnx_export import prepare_onnx_network
from traiNNer.tasks import load_task_provider
from traiNNer.tasks.dispatch import load_task_action
from traiNNer.utils.color_util import (
    complete_output_from_input,
    dataset_luma_pt,
    dataset_luma_weights,
    pixelformat2rgb_pt,
    rgb2pixelformat_pt,
    rgb2ycbcr,
)
from traiNNer.utils.options import yaml_load
from traiNNer.utils.redux_options import ReduxOptions

if TYPE_CHECKING:
    from traiNNer.tasks.object_detection.inference import DetectionPredictor


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


class _ImagePredictor(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        options: ReduxOptions,
        architecture: str,
        settings: ClassificationOptions | SegmentationOptions | None,
        *,
        luma: str | None = None,
        class_order_verified: bool | None = None,
        tiles: tuple[int, int, str] | None = None,
        complete_planes: bool = False,
    ) -> None:
        super().__init__()
        self.network = network
        self.options = options
        self.architecture = architecture
        self.settings = settings
        self.class_order_verified = class_order_verified
        self.tiles = tiles
        self.complete_planes = complete_planes
        self.luma = luma
        self.file_luma = dataset_luma_weights(luma, device="cpu") if luma == "luma601" else None

    @property
    def output_kind(self) -> str:
        if isinstance(self.settings, ClassificationOptions):
            return "classification"
        if isinstance(self.settings, SegmentationOptions):
            return "mask"
        return "luma" if self.luma else "rgb"

    @property
    def output_name(self) -> str:
        return (
            "logits"
            if isinstance(self.settings, ClassificationOptions)
            else self.output_kind
        )

    def metadata(self) -> dict[str, Any]:
        result = {
            "task": self.options.task,
            "architecture": self.architecture,
            "output_kind": self.output_kind,
        }
        if self.luma:
            result["luma_encoding"] = self.luma
            result["uint8_y_rounding"] = self.luma == "y" and self.options.input_pixel_format == "rgb"
            result["output_color"] = "bt601_full" if self.luma == "luma601" else "bt601_studio"
        if self.complete_planes:
            result.update(predicted_planes=self.options.output_pixel_format, complementary_planes="input_bicubic")
        if self.tiles:
            size, overlap, mode = self.tiles
            result["tiles"] = {"size": size, "overlap": overlap, "units": "input_pixels", "mode": mode}
        if self.settings is not None:
            result["class_names"] = self.settings.class_names
        if isinstance(self.settings, SegmentationOptions):
            result["ignore_label"] = self.settings.ignore_label
            result["class_order_verified"] = self.class_order_verified
        if isinstance(self.settings, ClassificationOptions):
            result["image_size"] = self.settings.image_size
        return result

    def network_input(self, image: Tensor) -> Tensor:
        if self.luma or self.options.task == "autoencoder":
            return image
        return rgb2pixelformat_pt(image, self.options.input_pixel_format)

    def forward(self, image: Tensor) -> Tensor:
        if self.options.task == "autoencoder":
            output = self.network(image)
            if not torch.jit.is_tracing() and (
                output.shape != image.shape or not torch.isfinite(output).all()
            ):
                raise ValueError(
                    "The autoencoder returned an invalid RGB reconstruction."
                )
            return output
        model_input = self.network_input(image)
        if self.tiles is None:
            output = self.network(model_input)
        else:
            from traiNNer.models.sr_inference import infer_tiled
            size, overlap, mode = self.tiles
            output = infer_tiled(self.network, model_input, tile_size=size, tile_overlap=overlap,
                                 scale=self.options.scale, device=image.device, amp_dtype=image.dtype, use_amp=False,
                                 mode=mode)
        if (
            isinstance(self.settings, SegmentationOptions)
            and isinstance(output, dict)
            and output.keys() == {"out"}
        ):
            output = output["out"]
        if not isinstance(output, Tensor):
            raise ValueError("Headless prediction requires a single output tensor.")

        if not torch.jit.is_tracing():
            channels = (
                len(self.settings.class_names) if self.settings is not None else (1 if self.luma else 3)
            )
            if self.complete_planes:
                channels = 1 if self.options.output_pixel_format == "y" else 2
            scale = self.options.scale
            if isinstance(self.settings, ClassificationOptions):
                expected = (image.shape[0], channels)
            else:
                expected = (
                    image.shape[0],
                    channels,
                    image.shape[2] * scale,
                    image.shape[3] * scale,
                )
            if output.shape != expected or not output.is_floating_point():
                raise ValueError(
                    f"{self.architecture} returned {tuple(output.shape)} {output.dtype}. "
                    f"Expected a floating point tensor of shape {expected} for {self.options.task}."
                )
            if not torch.isfinite(output).all():
                raise ValueError(f"{self.architecture} returned nonfinite values.")

        if isinstance(self.settings, ClassificationOptions):
            return output.float()
        if isinstance(self.settings, SegmentationOptions):
            return output.argmax(dim=1)
        if self.luma:
            return output.float().clamp(0, 1)
        if self.complete_planes:
            return complete_output_from_input(output, image, self.options.output_pixel_format, self.options.scale).clamp(0, 1)
        return (
            pixelformat2rgb_pt(output, None, self.options.output_pixel_format)
            .float()
            .clamp(0, 1)
        )


def _load_predictor(
    opt: ReduxOptions,
    checkpoint: Path,
    *,
    score_threshold: float | None = None,
    top_limit: int | None = None,
    checkpoint_signature: Sequence[int] | None = None,
    honor_validation_tiles: bool = False,
) -> "_ImagePredictor | DetectionPredictor":
    if opt.task is None:
        opt.task = task_name(opt)
    if opt.task != "object_detection" and any(
        value is not None
        for value in (score_threshold, top_limit, checkpoint_signature)
    ):
        raise ValueError(
            "Score threshold, top limit, and checkpoint signature are detection options."
        )

    luma = still_restoration_luma(opt) if opt.task == "image_restoration" else None
    complete_planes = bool(opt.task == "image_restoration" and opt.val and opt.val.complementary_planes == "input_bicubic"
                           and opt.output_pixel_format in ("y", "uv"))
    dataset = opt.datasets.get("train")
    if (
        opt.task in ("image_restoration", "autoencoder")
        and dataset is not None
        and (dataset.mean is not None and dataset.std is not None)
    ):
        raise ValueError(
            "Headless prediction expects training images in [0, 1] RGB. "
            "Per-dataset mean/std normalization is unsupported."
        )

    tiles = None
    if honor_validation_tiles and opt.task == "image_restoration" and opt.val and opt.val.tile_size:
        if not 0 <= opt.val.tile_overlap < opt.val.tile_size:
            raise ValueError("Use a nonnegative val.tile_overlap smaller than val.tile_size.")
        tiles = (opt.val.tile_size, opt.val.tile_overlap, opt.val.tile_mode)

    # Deployment builds the registered network, without training or dataset setup.
    opt.is_train = False
    opt.num_gpu = 0
    opt.dist = False
    opt.use_amp = False
    opt.train = None
    opt.val = None
    opt.datasets = {}
    opt.network_g_teacher = None
    opt.network_d = None
    selection = resolve_task(opt)
    assert selection is not None
    spec, architecture, settings = selection
    net_options = getattr(opt, spec.network_field)
    assert net_options is not None
    if architecture in TEMPORAL_ARCHS:
        raise ValueError(
            f"{architecture} requires video clips. Supply a still-image architecture."
        )
    if opt.scale < 1:
        raise ValueError("A positive integer scale is required.")
    if luma is None and not complete_planes and opt.task != "autoencoder" and opt.output_pixel_format in ("y", "uv"):
        raise ValueError(
            "Luma- or chroma-only outputs need a reference image. Use RGB or YUV444 output."
        )
    load_task_provider(spec)
    if isinstance(settings, DetectionOptions):
        return _load_detection_predictor(
            opt,
            architecture,
            settings,
            checkpoint,
            score_threshold=score_threshold,
            top_limit=top_limit,
            checkpoint_signature=checkpoint_signature,
        )
    assert settings is None or isinstance(
        settings, (ClassificationOptions, SegmentationOptions)
    )
    if isinstance(settings, SegmentationOptions) and len(settings.class_names) > 65536:
        raise ValueError("Deployment supports up to 65536 class IDs in a PNG mask.")

    network = build_network({**net_options, "scale": opt.scale}).to(
        device="cpu", dtype=torch.float32
    )
    segmentation = isinstance(settings, SegmentationOptions)
    from trainner_config.tasks import AETHER_ARCHITECTURES

    class_order_verified = None
    if opt.task == "image_restoration" and architecture.lower() in AETHER_ARCHITECTURES:
        from .aether_deployment import load_deployment_weights

        load_deployment_weights(network, checkpoint, opt.path.param_key_g)
    else:
        class_order_verified = load_matching_network_checkpoint(
            network,
            checkpoint,
            opt.path.param_key_g,
            class_names=settings.class_names if settings is not None else None,
            expected_task=opt.task if settings is not None else None,
            allow_legacy_class_order=segmentation,
        )
    if opt.task == "autoencoder":
        from traiNNer.tasks.restoration_predict import Reconstruction

        network = Reconstruction(network, opt.scale)
    return _ImagePredictor(
        network, opt, architecture, settings, luma=luma, class_order_verified=class_order_verified,
        tiles=tiles, complete_planes=complete_planes
    ).eval()


def _load_detection_predictor(
    opt: ReduxOptions,
    architecture: str,
    settings: DetectionOptions,
    checkpoint: Path,
    *,
    score_threshold: float | None,
    top_limit: int | None,
    checkpoint_signature: Sequence[int] | None,
) -> "DetectionPredictor":
    from traiNNer.tasks.object_detection.inference import (
        DetectionPredictor,
        checkpoint_signature as read_checkpoint_signature,
    )

    assert opt.network_g is not None
    threshold = (
        opt.network_g.get("box_score_thresh", 0.05)
        if score_threshold is None
        else score_threshold
    )
    limit = 100 if top_limit is None else top_limit
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("Detection score threshold must be between 0 and 1.")
    if type(limit) is not int or limit < 1:
        raise ValueError("Detection top limit must be a positive integer.")

    checkpoint = checkpoint.resolve(strict=True)
    signature = read_checkpoint_signature(checkpoint)
    if checkpoint_signature is not None and tuple(checkpoint_signature) != signature:
        raise ValueError(
            f"Checkpoint {checkpoint} no longer matches the selected ledger identity."
        )

    network = build_network({**opt.network_g, "scale": opt.scale}).to(
        device="cpu", dtype=torch.float32
    )
    load_matching_network_checkpoint(
        network,
        checkpoint,
        opt.path.param_key_g,
        class_names=[category.name for category in settings.categories],
        category_ids=[category.id for category in settings.categories],
        expected_task="object_detection",
    )
    if read_checkpoint_signature(checkpoint) != signature:
        raise ValueError(f"Checkpoint {checkpoint} changed while loading.")
    return DetectionPredictor(
        network,
        architecture,
        settings,
        checkpoint,
        signature,
        score_threshold=float(threshold),
        top_limit=limit,
    )


def _input_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported still image extension: {source.suffix!r}.")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    inputs = [
        path
        for path in sorted(source.rglob("*"))
        if path.suffix.lower() in _IMAGE_EXTENSIONS and path.is_file()
    ]
    if not inputs:
        raise ValueError(f"No supported still images found in {source}.")
    return inputs


def _prediction_paths(
    source: Path, destination: Path, inputs: list[Path]
) -> list[tuple[Path, Path]]:
    output_root = destination.resolve()
    folder_input = source.is_dir()
    if folder_input:
        if output_root.is_relative_to(source):
            raise ValueError(
                "The output directory must be outside the input directory."
            )
        if destination.exists() and not destination.is_dir():
            raise ValueError("A directory input requires an output directory.")
        pairs = [
            (path, destination / path.relative_to(source).with_suffix(".png"))
            for path in inputs
        ]
    else:
        if destination.suffix.lower() != ".png":
            raise ValueError("A single input image requires a .png output file.")
        pairs = [(source, destination)]

    outputs: set[Path] = set()
    for _, path in pairs:
        resolved = path.resolve()
        if folder_input and not resolved.is_relative_to(output_root):
            raise ValueError(
                f"Output path {path} resolves outside its output directory."
            )
        if resolved in outputs:
            raise ValueError(f"Multiple images would write {path}.")
        if resolved.exists() or path.is_symlink():
            raise FileExistsError(f"Output already exists: {path}")
        outputs.add(resolved)
    return pairs


def _read_image(path: Path, *, segmentation: bool, native_y: bool = False) -> Tensor:
    import pyvips
    from traiNNer.utils.img_util import img2rgb, img2tensor, vipsimfrompath

    try:
        if segmentation:
            # Match ImageMaskDataset's RGB decoding.
            with Image.open(path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Expected a single-frame image.")
                pixels = np.array(image.convert("RGB"))
        else:
            # Match SingleImageDataset's sRGB and tensor conversion.
            decoded = vipsimfrompath(str(path))
            if decoded.get_typeof("n-pages") and decoded.get("n-pages") != 1:
                raise ValueError("Expected a single-frame image.")
            pixels = img2rgb(decoded.numpy())
            if native_y:
                # SingleImageDataset rounds uint8 Y codes before tensor normalization.
                pixels = rgb2ycbcr(pixels, y_only=True)[..., None]
        return img2tensor(pixels).unsqueeze(0).contiguous()
    except (OSError, ValueError, pyvips.Error) as exc:
        raise ValueError(f"Could not decode {path}. {exc}") from exc


def _write_new_file(path: Path, writer: Callable[[BinaryIO], Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("xb")
    try:
        with file:
            writer(file)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _predict_file(model: _ImagePredictor, source: Path) -> Tensor:
    if isinstance(model.settings, ClassificationOptions):
        from traiNNer.tasks.image_classification.dataset import read_classification_image

        image = read_classification_image(source, model.settings.image_size).unsqueeze(
            0
        )
    else:
        image = _read_image(
            source, segmentation=isinstance(model.settings, SegmentationOptions),
            native_y=model.luma == "y" and model.options.input_pixel_format == "rgb",
        )
        if model.file_luma is not None:
            image = dataset_luma_pt(image, *model.file_luma)
        elif model.luma == "y" and model.options.input_pixel_format == "y":
            image = rgb2pixelformat_pt(image, "y")
    try:
        return model(image)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Prediction failed for {source}. {exc}") from exc


@torch.inference_mode()
def predict(
    config: str | Path,
    checkpoint: str | Path,
    input_path: str | Path,
    output: str | Path,
    *,
    clip_size: int | None = None,
    score_threshold: float | None = None,
    top_limit: int | None = None,
    checkpoint_signature: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Run the recipe's registered prediction handler."""
    opt, _ = yaml_load(str(config))
    handler = load_task_action(opt, "predict")
    opt.is_train = False
    flags = {
        name: value
        for name, value in {
            "clip_size": clip_size,
            "score_threshold": score_threshold,
            "top_limit": top_limit,
            "checkpoint_signature": checkpoint_signature,
        }.items()
        if value is not None
    }
    result = handler(
        opt,
        Path(checkpoint),
        Path(input_path).resolve(),
        Path(output).absolute(),
        **flags,
    )
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("outputs"), list)
        or any(not isinstance(path, str) for path in result["outputs"])
    ):
        raise ValueError(
            "The prediction handler must return an object with an outputs list of paths."
        )
    return result


def _predict_frames(
    opt: ReduxOptions,
    checkpoint: Path,
    source: Path,
    destination: Path,
    *,
    clip_size: int | None = None,
    **detection_options: object,
) -> dict[str, Any]:
    if detection_options:
        raise ValueError("Detection options cannot be applied to a frame sequence.")
        from traiNNer.tasks.restoration_predict import predict_sequence

        return predict_sequence(opt, checkpoint, source, destination, clip_size)


def _predict_images(
    opt: ReduxOptions,
    checkpoint: Path,
    source: Path,
    destination: Path,
    *,
    clip_size: int | None = None,
    score_threshold: float | None = None,
    top_limit: int | None = None,
    checkpoint_signature: Sequence[int] | None = None,
) -> dict[str, Any]:
    if clip_size is not None:
        raise ValueError("clipSize applies only to frame sequence prediction.")
    inputs = _input_images(source)
    model = _load_predictor(
        opt,
        checkpoint,
        score_threshold=score_threshold,
        top_limit=top_limit,
        checkpoint_signature=checkpoint_signature,
        honor_validation_tiles=True,
    )
    if not isinstance(model, _ImagePredictor) or isinstance(
        model.settings, ClassificationOptions
    ):
        if destination.suffix.lower() != ".json":
            task = (
                "Detection"
                if not isinstance(model, _ImagePredictor)
                else "Classification"
            )
            raise ValueError(f"{task} prediction requires a .json destination.")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output already exists: {destination}")

        if not isinstance(model, _ImagePredictor):
            base = source if source.is_dir() else source.parent
            predictions = [
                model.predict(path, file_name=path.relative_to(base).as_posix())
                for path in inputs
            ]
        else:
            assert isinstance(model.settings, ClassificationOptions)
            predictions = []
            for path in inputs:
                scores = _predict_file(model, path)[0]
                label = int(scores.argmax().item())
                predictions.append(
                    {
                        "input": str(path),
                        "class_id": label,
                        "class_name": model.settings.class_names[label],
                        "logits": scores.tolist(),
                    }
                )
        document = {**model.metadata(), "predictions": predictions}
        encoded = json.dumps(document, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        _write_new_file(destination, lambda file: file.write(encoded))
        return {
            **model.metadata(),
            "outputs": [str(destination)],
            "count": len(predictions),
        }

    pairs = _prediction_paths(source, destination, inputs)
    for path, target in pairs:
        result = _predict_file(model, path)
        if isinstance(model.settings, SegmentationOptions):
            from .semantic_segmentation.masks import mask_image
            image = mask_image(result[0], len(model.settings.class_names))
        else:
            from traiNNer.utils.img_util import tensor2img

            image = Image.fromarray(tensor2img(result, to_bgr=False))
        _write_new_file(target, partial(image.save, format="PNG"))

    return {**model.metadata(), "outputs": [str(path) for _, path in pairs]}


def export_onnx(
    config: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    height: int | None = None,
    width: int | None = None,
    opset: int = 17,
    clip_size: int | None = None,
    dynamic_spatial: bool = False,
) -> dict[str, Any]:
    """Run the recipe's registered handler for portable ONNX export."""
    opt, _ = yaml_load(str(config))
    handler = load_task_action(opt, "export")
    opt.is_train = False
    destination = Path(output).absolute()
    if destination.suffix.lower() != ".onnx":
        raise ValueError("Export requires a .onnx output file.")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if not 13 <= opset <= 20:
        raise ValueError("Portable export supports ONNX opsets 13 through 20.")

    from trainner_config.tasks import require_task_action

    dimensions = {"opset": opset}
    capabilities = require_task_action(opt, "export")
    if capabilities.inputs != ("records",):
        if height is None or width is None or min(height, width) < 1:
            raise ValueError("Export height and width must be positive.")
        dimensions.update(height=height, width=width)
    if dynamic_spatial:
        from trainner_config.tasks import supports_dynamic_spatial_export

        if not supports_dynamic_spatial_export(task_name(opt), str((opt.network_g or {}).get("type", ""))):
            raise ValueError("Dynamic spatial export is supported for Compact and Aether image restoration recipes.")
        dimensions["dynamic_spatial"] = True
    if clip_size is not None:
        if "sequences" not in capabilities.inputs:
            raise ValueError("Clip length applies to sequence exports.")
        dimensions["clip_size"] = clip_size
    result = handler(opt, Path(checkpoint), destination, **dimensions)
    if not isinstance(result, dict) or result.get("output") != str(destination):
        raise ValueError(
            "The export handler must report the requested ONNX output path."
        )
    return result


def _export_image_onnx(
    opt: ReduxOptions,
    checkpoint: Path,
    destination: Path,
    *,
    height: int,
    width: int,
    opset: int,
    clip_size: int | None = None,
    dynamic_spatial: bool = False,
) -> dict[str, Any]:
    if clip_size is not None:
        raise ValueError("clip_size applies only to sequence export.")
    from .onnx_export import write_portable_onnx

    model = _load_predictor(opt, checkpoint)
    assert isinstance(model, _ImagePredictor)
    from trainner_config.tasks import AETHER_ARCHITECTURES

    aether_model = model.architecture.lower() in AETHER_ARCHITECTURES
    if aether_model:
        model.network.fuse_model()
    if isinstance(model.settings, ClassificationOptions):
        size = model.settings.image_size
        if (height, width) != (size, size):
            raise ValueError(
                f"Classification export requires height and width equal to task_options.image_size ({size})."
            )
    channels = 1 if model.luma else 3
    example = torch.zeros((1, channels, height, width), dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        network_input = model.network_input(example)
        model.network = prepare_onnx_network(
            model.network, tuple(network_input.shape), dynamic_spatial=dynamic_spatial
        ).eval()
        if getattr(model.network, "onnx_deployment_precision_handled", False):
            raise ValueError(
                "This architecture prepared a deployment-specific precision graph. Portable export requires FP32."
            )
        output_example = model(example)

    signal = model.metadata()
    metadata = {
        **signal,
        "input_name": "input",
        "input_shape": list(example.shape),
        "input_dtype": "float32",
        "input_color": signal.get("output_color", "rgb_0_1"),
        "output_name": model.output_name,
        "output_shape": list(output_example.shape),
        "output_dtype": "int64"
        if isinstance(model.settings, SegmentationOptions)
        else "float32",
        "opset": opset,
    }
    if aether_model:
        from traiNNer.models.sr_qat import aether_export_precision

        metadata["precision"] = aether_export_precision(model.network)
    if dynamic_spatial:
        metadata.update(
            fixed_shape=False,
            example_input_shape=list(example.shape),
            scale=opt.scale,
            input_shape=[1, channels, "height", "width"],
            output_shape=[1, channels, "output_height", "output_width"],
        )
    return write_portable_onnx(model, example, destination, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Predict from the task's declared inputs")
    export = subparsers.add_parser(
        "export", help="Export a portable ONNX model"
    )
    for command in (run, export):
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument(
        "--clip-size", type=int, help="Frames per temporal prediction window."
    )
    run.add_argument(
        "--score-threshold", type=float, help="Detection score cutoff, exclusive"
    )
    run.add_argument("--top-limit", type=int, help="Maximum detections per image")
    run.add_argument(
        "--checkpoint-signature",
        type=int,
        nargs=5,
        metavar=("DEV", "INO", "SIZE", "MTIME_NS", "CTIME_NS"),
        help="Expected detection checkpoint ledger identity",
    )
    export.add_argument("--height", type=int, help="Input height for spatial models.")
    export.add_argument("--width", type=int, help="Input width for spatial models.")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--dynamic-spatial", action="store_true", help="Variable height and width for Compact and Aether restoration models.")
    export.add_argument(
        "--clip-size",
        type=int,
        help="Frame count for fixed-window export. Required for RSR. TemporalSPAN uses its trained window.",
    )

    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            if args.command == "run":
                result = predict(
                    args.config,
                    args.checkpoint,
                    args.input,
                    args.output,
                    clip_size=args.clip_size,
                    score_threshold=args.score_threshold,
                    top_limit=args.top_limit,
                    checkpoint_signature=args.checkpoint_signature,
                )
            else:
                result = export_onnx(
                    args.config,
                    args.checkpoint,
                    args.output,
                    height=args.height,
                    width=args.width,
                    opset=args.opset,
                    clip_size=args.clip_size,
                    dynamic_spatial=args.dynamic_spatial,
                )
    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        pickle.UnpicklingError,
        SafetensorError,
    ) as exc:
        parser.exit(1, f"{args.command} failed. {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
