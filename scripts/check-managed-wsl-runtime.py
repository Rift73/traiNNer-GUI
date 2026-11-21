"""Check the managed imports, media libraries and optional CUDA device."""

import sys

import torch
from torchvision.ops import nms


def check_runtime() -> None:
    import av
    import chainner_ext
    import cv2
    import lmdb
    import pyvips
    import segmentation_models_pytorch
    from torchcodec.decoders import VideoDecoder
    from torchcodec.encoders import VideoEncoder

    from traiNNer.check.check_dependencies import check_dependencies

    check_dependencies()
    # Exercise the library bindings as well as the Python imports.
    image = torch.zeros((2, 3, 16, 16), dtype=torch.uint8)
    encoded = VideoEncoder(image, frame_rate=24).to_tensor(
        "mp4", codec="libx264", preset="ultrafast"
    )
    decoded = VideoDecoder(encoded, num_ffmpeg_threads=1)
    if decoded[0].shape != (3, 16, 16):
        raise RuntimeError("TorchCodec could not round-trip a test frame with FFmpeg.")
    if nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.ones(1), 0.5).tolist() != [0]:
        raise RuntimeError("The loaded torchvision native ops failed.")
    if pyvips.Image.black(1, 1).width != 1:
        raise RuntimeError("libvips could not create a test image.")

    imports = (av, chainner_ext, cv2, lmdb, segmentation_models_pytorch)
    print("Managed runtime loaded " + ", ".join(module.__name__ for module in imports))


def check_cuda() -> None:
    from traiNNer.data.ptx_cuda import PtxModule

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch could not initialize CUDA in WSL 2.")
    if torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError("The first NVIDIA GPU must be Ampere or newer (SM 8.0+).")
    with torch.cuda.device(0):
        if (torch.ones(1, device="cuda") + 1).item() != 2:
            raise RuntimeError("PyTorch could not run its CUDA smoke check.")
        # Loading the actual shipped PTX checks the Windows driver's PTX JIT.
        PtxModule("effect_cuda.ptx")._function("effect_shift_yuv")
        PtxModule("training_cuda.ptx")._function("dvd_horizontal")
    print(f"CUDA {torch.version.cuda} ready on {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--gpu"]:
        check_cuda()
    else:
        check_runtime()
