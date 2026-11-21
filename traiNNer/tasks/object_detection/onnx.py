"""Portable Faster R-CNN inference with fixed input geometry and variable detections."""

from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.ops import clip_boxes_to_image

from traiNNer.tasks.object_detection.inference import DetectionPredictor
from traiNNer.tasks.onnx_export import write_portable_onnx
from traiNNer.tasks.predict import _load_predictor
from trainner_config.options import ReduxOptions


class DetectionGraph(nn.Module):
    def __init__(self, predictor: DetectionPredictor) -> None:
        super().__init__()
        self.network = predictor.network
        self.register_buffer("category_ids", torch.tensor([0, *(c.id for c in predictor.categories)], dtype=torch.int64))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        result = self.network([image[0]])[0]
        # Torchvision has already applied the predictor's threshold, NMS, and
        # top limit in score order. Return boxes in original image coordinates.
        boxes = clip_boxes_to_image(result["boxes"], (image.shape[-2], image.shape[-1]))
        return boxes, result["scores"], self.category_ids[result["labels"]]


def export_detection(
    opt: ReduxOptions, checkpoint: Path, destination: Path, *, height: int, width: int,
    opset: int, clip_size: int | None = None,
) -> dict:
    if clip_size is not None:
        raise ValueError("Detection export takes one RGB image.")
    predictor = _load_predictor(opt, checkpoint)
    if not isinstance(predictor, DetectionPredictor):
        raise ValueError("The recipe must select an object detector.")
    graph = DetectionGraph(predictor).eval()
    example = torch.zeros((1, 3, height, width), dtype=torch.float32)
    metadata = {
        **predictor.metadata(),
        "backend": "onnx",
        "opset": opset,
        "input_name": "input",
        "input_shape": list(example.shape),
        "input_dtype": "float32",
        "input_color": "rgb_0_1",
        "fixed_input_shape": True,
        "outputs": [
            {"name": "boxes", "shape": ["detections", 4], "dtype": "float32"},
            {"name": "scores", "shape": ["detections"], "dtype": "float32"},
            {"name": "category_ids", "shape": ["detections"], "dtype": "int64"},
        ],
    }
    return write_portable_onnx(graph, example, destination, metadata)
