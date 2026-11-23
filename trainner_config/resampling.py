"""Kernel names accepted by the NeoSR on-the-fly resize policy."""

from typing import Literal, get_args

NeoSRAntialiasMode = Literal[
    "aa_box", "aa_bilinear", "aa_bicubic", "aa_mitchell", "aa_lanczos2", "aa_lanczos3",
]
NeoSRResizeMode = Literal["area", "bilinear", "bicubic", NeoSRAntialiasMode]
NEOSR_AA_MODES = get_args(NeoSRAntialiasMode)
NEOSR_RESIZE_MODES = get_args(NeoSRResizeMode)
