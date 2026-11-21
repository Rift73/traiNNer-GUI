"""Architecture compile contracts and PyTorch's process-wide cache setup.

BaseModel owns device movement, eager preparation, and wrapping order. These
helpers only resolve architecture options and configure the compiler policies
that must remain active when torch.compile traces a network lazily.
"""

import json
from logging import Logger
from typing import Any

import torch
from torch import nn

from traiNNer.utils.misc import is_json_compatible
from traiNNer.utils.redux_options import ReduxOptions


def resolve_inductor_options(
    net: nn.Module, compile_mode: str, logger: Logger
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return architecture overrides and mode-expanded options, if needed.

    None preserves torch.compile's mode argument. Nonempty architecture
    overrides instead require options, because PyTorch rejects mode + options.
    """
    resolver = getattr(net, "get_compile_inductor_options", None)
    architecture_options: dict[str, Any] = {}
    if callable(resolver):
        resolved_options = resolver(compile_mode)
        if not isinstance(resolved_options, dict):
            raise ValueError("get_compile_inductor_options() must return a dict")
        if not is_json_compatible(resolved_options):
            raise ValueError(
                "get_compile_inductor_options() must return a "
                "JSON-compatible dict with string keys"
            )
        architecture_options = dict(resolved_options)

    effective_options: dict[str, Any] | None = None
    if architecture_options:
        effective_options = dict(
            torch._inductor.list_mode_options(compile_mode)  # noqa: SLF001
        )
        effective_options.update(architecture_options)
        logger.info(
            "Network %s adds architecture-specific Inductor options %s.",
            net.__class__.__name__,
            json.dumps(architecture_options, sort_keys=True),
        )
    return architecture_options, effective_options


def enable_precompile_cache(
    compile_mode: str, effective_options: dict[str, Any] | None
) -> None:
    """Validate the experimental API, then enable lazy artifact capture.

    The caller records that publication is enabled only after this succeeds.
    CUDA-Graph policy must be seeded before the package deserializer runs, not
    deferred until Inductor receives the eventual torch.compile options.
    """
    try:
        from torch._dynamo.precompile_context import PrecompileContext
    except ImportError as error:
        raise RuntimeError(
            "Fast compile-resume requires PyTorch's precompile-context "
            f"API, which is unavailable in torch {torch.__version__}."
        ) from error

    required_api = {
        "PrecompileContext._dynamo_cache_entries": hasattr(
            PrecompileContext, "_dynamo_cache_entries"
        ),
        "PrecompileContext._backend_artifacts_by_key": hasattr(
            PrecompileContext, "_backend_artifacts_by_key"
        ),
        "PrecompileContext.save_to_dynamo_cache": hasattr(
            PrecompileContext, "save_to_dynamo_cache"
        ),
        "torch._dynamo.config.caching_precompile": hasattr(
            torch._dynamo.config,  # noqa: SLF001
            "caching_precompile",
        ),
        "torch._inductor.list_mode_options": hasattr(
            torch._inductor,  # noqa: SLF001
            "list_mode_options",
        ),
    }
    missing_api = [name for name, available in required_api.items() if not available]
    if missing_api:
        raise RuntimeError(
            "Fast compile-resume is not compatible with torch "
            f"{torch.__version__}; missing "
            + ", ".join(missing_api)
            + ". Disable torch.compile or use the validated PyTorch build."
        )

    # PyTorch 2.14's experimental serializer records the public SDPBackend
    # enum under its pybind class name. Supply its missing module alias before
    # Dynamo captures artifacts.
    import torch.nn.attention as attention_module
    from torch.nn.attention import SDPBackend

    if not hasattr(attention_module, "_SDPBackend"):
        attention_module._SDPBackend = SDPBackend  # noqa: SLF001
    torch._dynamo.config.caching_precompile = True  # noqa: SLF001

    # A restored max-autotune package must retain its original CUDA Graph
    # Trees wrapper; otherwise it restores quickly but runs outside the trees.
    compile_mode_options = effective_options
    if compile_mode_options is None:
        compile_mode_options = torch._inductor.list_mode_options(compile_mode)  # noqa: SLF001
    torch._inductor.config.triton.cudagraphs = bool(  # noqa: SLF001
        compile_mode_options.get("triton.cudagraphs", False)
    )


def apply_compile_cache_tag(
    net: nn.Module,
    *,
    opt: ReduxOptions,
    use_amp: bool,
    amp_dtype: torch.dtype,
    memory_format: torch.memory_format,
    architecture_options: dict[str, Any],
    logger: Logger,
) -> None:
    """Append this architecture's graph ABI to the process-wide cache identity.

    Tags include the input/compiler contract and retain insertion order. A
    student + teacher run intentionally differs from a solo-student run: this
    coupling may cause a safe cold compile, never an unsafe stale cache hit.
    """
    architecture_cache_tag = getattr(net, "compile_cache_key_tag", None)
    if architecture_cache_tag is None:
        return
    if not isinstance(architecture_cache_tag, str) or not architecture_cache_tag:
        raise ValueError(
            "compile_cache_key_tag must be a non-empty string when defined"
        )

    train_dataset = opt.datasets.get("train")
    compile_batch = (
        train_dataset.batch_size_per_gpu if train_dataset is not None else None
    )
    compile_crop = train_dataset.lq_size if train_dataset is not None else None
    architecture_cache_tag = (
        f"{architecture_cache_tag}:mode={opt.compile_mode}"
        f":input={compile_batch}x{compile_crop}x{compile_crop}"
        f":amp={use_amp}-{amp_dtype}"
        f":layout={memory_format}"
        ":autotune-at-compile-time="
        f"{torch._inductor.config.triton.autotune_at_compile_time}"  # noqa: SLF001
        ":static-triton-launcher="
        f"{torch._inductor.config.use_static_triton_launcher}"  # noqa: SLF001
    )
    if architecture_options:
        architecture_cache_tag += ":inductor-options=" + json.dumps(
            architecture_options, sort_keys=True, separators=(",", ":")
        )
    current_cache_tag = torch.compiler.config.cache_key_tag
    cache_tags = current_cache_tag.split("|") if current_cache_tag else []
    if architecture_cache_tag not in cache_tags:
        cache_tags.append(architecture_cache_tag)
        torch.compiler.config.cache_key_tag = "|".join(cache_tags)
    logger.info(
        "Network %s uses compile cache ABI tag %s.",
        net.__class__.__name__,
        architecture_cache_tag,
    )
