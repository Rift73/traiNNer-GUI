"""Source-backed string examples for the architecture schema."""

_RESIDUAL_CONVS = {"resi_connection": ("1conv", "3conv")}
_RESIDUAL_CONV_OR_IDENTITY = {"resi_connection": ("1conv", "identity")}
_DAT_EXAMPLES = {
    **_RESIDUAL_CONVS,
    "upsampler": ("pixelshuffle", "pixelshuffledirect"),
}

# The empty string selects the same-size restoration branch. "nearest+conv"
# requires scale 4 in the architectures that implement it.
_RESTORATION_UPSAMPLERS = ("pixelshuffle", "pixelshuffledirect", "nearest+conv", "")
_RESTORATION_EXAMPLES = {
    **_RESIDUAL_CONVS,
    "upsampler": _RESTORATION_UPSAMPLERS,
}

_MODULE_EXAMPLES: dict[str, dict[str, tuple[str, ...]]] = {
    # The Spandrel wrappers use the options implemented by the wrapped models.
    "traiNNer.archs.atd_arch": _RESTORATION_EXAMPLES,
    "traiNNer.archs.craft_arch": _RESIDUAL_CONV_OR_IDENTITY,
    "traiNNer.archs.dat_arch": _DAT_EXAMPLES,
    "traiNNer.archs.dat_iln_arch": _DAT_EXAMPLES,
    "traiNNer.archs.drct_arch": {
        **_RESIDUAL_CONV_OR_IDENTITY,
        "upsampler": ("pixelshuffle",),
    },
    # DUAL's rib and none require context_na to be disabled.
    "traiNNer.archs.dual_arch": {"c_pos_bias": ("rfb", "rib", "none")},
    # Only pixelshuffle is wired into the DWT forward path.
    "traiNNer.archs.dwt_arch": {
        **_RESIDUAL_CONVS,
        "upsampler": ("pixelshuffle",),
    },
    "traiNNer.archs.grl_arch": {
        "anchor_proj_type": (
            "avgpool",
            "maxpool",
            "conv2d",
            "separable_conv",
            "patchmerging",
        ),
        "conv_type": ("1conv", "3conv", "1conv1x1", "linear"),
        "init_method": ("n", "r", "w", "l", "t1"),
        "qkv_proj_type": ("linear", "separable_conv"),
        "upsampler": _RESTORATION_UPSAMPLERS,
    },
    "traiNNer.archs.hat_arch": _RESIDUAL_CONV_OR_IDENTITY,
    "traiNNer.archs.hat_iln_arch": _RESIDUAL_CONV_OR_IDENTITY,
    "traiNNer.archs.hit_sir_arch": _RESTORATION_EXAMPLES,
    "traiNNer.archs.hit_sng_arch": _RESTORATION_EXAMPLES,
    "traiNNer.archs.hit_srf_arch": _RESTORATION_EXAMPLES,
    "traiNNer.archs.rcan_arch": {
        "act_mode": ("relu", "leaky_relu", "elu", "silu", "gelu", "none"),
    },
    "traiNNer.archs.rgt_arch": _RESIDUAL_CONVS,
    "traiNNer.archs.srformer_arch": _RESIDUAL_CONVS,
    "traiNNer.archs.srformerv2_arch": _RESTORATION_EXAMPLES,
    "traiNNer.archs.srvgg_arch": {"act_type": ("prelu", "relu", "leakyrelu")},
    "traiNNer.archs.swin2sr_arch": _RESIDUAL_CONVS,
    "traiNNer.archs.swinir_arch": _RESIDUAL_CONVS,
    # Stable names documented by the U-Net wrapper, which delegates to SMP.
    "traiNNer.archs.unetsegmentation_arch": {
        "encoder_name": ("resnet50", "resnet34", "efficientnet-b7"),
    },
}

# The DocRes discriminator and DiTn accept LayerNorm_type but do not use it.
_DOCRES_EXAMPLES = {"LayerNorm_type": ("WithBias", "BiasFree")}
_FACTORY_EXAMPLES = {
    "traiNNer.archs.docres_arch.docres_base": _DOCRES_EXAMPLES,
    "traiNNer.archs.docres_arch.docres_large": _DOCRES_EXAMPLES,
}


def architecture_choice_examples(factory: object, argument: str) -> tuple[str, ...]:
    """Return examples for a known factory argument without loading any models."""
    module = getattr(factory, "__module__", None)
    if not isinstance(module, str):
        return ()
    qualified_name = f"{module}.{getattr(factory, '__qualname__', '')}"
    examples = _FACTORY_EXAMPLES.get(qualified_name, _MODULE_EXAMPLES.get(module, {}))
    return examples.get(argument, ())
