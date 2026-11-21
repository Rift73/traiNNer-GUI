"""Kernel names accepted by the NeoSR on-the-fly resize policy."""

NEOSR_AA_MODES = (
    "aa_box", "aa_bilinear", "aa_bicubic", "aa_mitchell", "aa_lanczos2", "aa_lanczos3",
)
NEOSR_RESIZE_MODES = ("area", "bilinear", "bicubic", *NEOSR_AA_MODES)
