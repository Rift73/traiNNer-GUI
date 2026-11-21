"""Translate NeoSR's two-stage OTF recipe into trainer and dataset options."""

from __future__ import annotations

from trainner_config.resampling import NEOSR_RESIZE_MODES

_KERNELS = {"iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"}
_RESIZE_MODES = ("area", "bilinear", "bicubic")
_STAGE_FIELDS = ("kernel_list", "kernel_prob", "blur_sigma", "blur_sigmay", "betag_range",
                 "betap_range", "sinc_prob", "resize_prob", "resize_range", "resize_mode",
                 "resize_mode_prob", "noise_prob", "gaussian_noise_prob", "noise_range",
                 "poisson_scale_range", "gray_noise_prob", "jpeg_prob", "jpeg_range")
_KEYS = {key + suffix for key in _STAGE_FIELDS for suffix in ("", "2")} | {
    "rotation", "final_sinc_prob", "final_resize_mode", "final_resize_mode_prob", "blur_prob",
    "clean_prob", "clean_mode", "clean_mode_prob", "sharpen_prob", "sharpen_sigma", "sharpen_amount",
    "second_blur_prob", "jpeg_compress",
}


def import_otf(group, raw_dataset, dataset, options, notes, unresolved):
    # The source loader overlays [degradations] onto [datasets.train].
    values = {}
    origins = {}
    for table, prefix in ((raw_dataset, "datasets.train."), (group, "degradations.")):
        for key in list(table):
            if key in _KEYS:
                value = table.pop(key)
                if key in values and value != values[key]:
                    notes.append(f"{prefix}{key} overrides the value in {origins[key]}.")
                values[key], origins[key] = value, prefix + key

    def review(key, reason):
        unresolved(origins.get(key, "degradations." + key), values.get(key), reason)

    def transfer(key, destination, name=None, *, required=True):
        if key in values:
            destination[name or key] = values[key]
        elif required:
            review(key, "NeoSR requires this value for the selected OTF pipeline.")

    options["otf_resize_policy"] = "neosr"
    sample = {}
    first_blur = values.get("blur_prob", 1.0)
    if type(first_blur) not in (int, float) or not 0 <= first_blur <= 1:
        review("blur_prob", "Choose a probability between zero and one.")
    else:
        options["blur_prob"] = float(first_blur > 0)
        if 0 < first_blur < 1:
            dataset["sample_blur_prob"] = first_blur
    transfer("second_blur_prob", options, "blur_prob2")
    transfer("final_sinc_prob", dataset)
    dataset.update(kernel_range=[7, 21], kernel_range2=[7, 21], final_kernel_range=[7, 21])
    if "rotation" in values:
        dataset["blur_rotation"] = values["rotation"]
    jpeg = values.get("jpeg_compress", True)
    if type(jpeg) is not bool:
        review("jpeg_compress", "Select whether to apply the JPEG stages.")
    for suffix in ("", "2"):
        kinds = values.get("kernel_list" + suffix)
        if not isinstance(kinds, list) or not kinds or any(
            not isinstance(kind, str) or kind not in _KERNELS for kind in kinds
        ):
            review("kernel_list" + suffix, "Select supported Gaussian, generalized Gaussian or plateau kernels.")
        else:
            transfer("kernel_list" + suffix, dataset)
        for key in ("kernel_prob", "sinc_prob", "blur_sigma", "betag_range", "betap_range"):
            required = key not in ("betag_range", "betap_range") or isinstance(kinds, list) and any(
                isinstance(k, str) and k.startswith("generalized" if key == "betag_range" else "plateau") for k in kinds
            )
            transfer(key + suffix, dataset, required=required)
        other_sigma = "blur_sigmay" + suffix
        if other_sigma in values:
            dataset[other_sigma] = values[other_sigma]
        for key in ("resize_prob", "resize_range", "gaussian_noise_prob", "noise_range",
                    "poisson_scale_range", "gray_noise_prob"):
            transfer(key + suffix, options)
        options["compress_prob" + suffix] = int(bool(jpeg))
        transfer("jpeg_range" + suffix, options, "compress_jpeg_range" + suffix, required=bool(jpeg))
        for key in ("noise_prob" + suffix, "jpeg_prob" + suffix):
            gate = values.get(key, 1)
            if type(gate) not in (int, float) or not 0 <= gate <= 1:
                review(key, "Choose a per-image probability between zero and one.")
            elif key.startswith("jpeg") and gate == 0:
                options["compress_prob" + suffix] = 0
            elif gate != 1:
                sample[key] = gate

    for source, target, weight_key in (("resize_mode", "", "resize_mode_prob"),
                                       ("resize_mode2", "2", "resize_mode_prob2"),
                                       ("final_resize_mode", "3", "final_resize_mode_prob")):
        modes = values.get(source, list(_RESIZE_MODES))
        if not isinstance(modes, list) or not modes or any(m not in NEOSR_RESIZE_MODES for m in modes):
            review(source, "Choose supported NeoSR interpolation modes.")
            if weight_key in values:
                review(weight_key, "Resolve the corresponding resize mode list first.")
            continue
        weights = values.get(weight_key, [1.0 / len(modes)] * len(modes))
        if (weight_key in values and source not in values or not isinstance(weights, list)
                or len(weights) != len(modes) or any(type(w) not in (int, float) or w < 0 for w in weights)
                or not any(weights)):
            review(weight_key, "Select one non-negative weight per declared resize mode with a positive total.")
            continue
        options["resize_mode_list" + target] = modes
        options["resize_mode_prob" + target] = weights
    for key in ("clean_prob", "clean_mode", "clean_mode_prob", "sharpen_prob", "sharpen_sigma", "sharpen_amount"):
        if key in values:
            if key == "clean_mode_prob" and "clean_mode" not in values:
                review(key, "Declare the clean_mode list that these weights apply to.")
                continue
            sample[key] = values[key]
    if sample:
        import msgspec
        from trainner_config.options import NeoSRSampleOptions

        try:
            msgspec.convert(sample, type=NeoSRSampleOptions, strict=True)
        except msgspec.ValidationError as exc:
            for key in sample:
                review(key, str(exc))
        else:
            options["neosr_samples"] = sample
    batch = dataset.get("batch_size_per_gpu")
    size = raw_dataset.pop("queue_size", 180)
    if type(batch) is int and type(size) is int and size >= batch > 0:
        options["queue_size"] = size // batch * batch
    else:
        unresolved("datasets.train.queue_size", size, "Choose a pool at least as large as one positive batch.")
    tight = raw_dataset.pop("tight_canvas", False)
    if type(tight) is bool:
        dataset["otf_canvas"] = "neosr_tight" if tight else "neosr"
    else:
        unresolved("datasets.train.tight_canvas", tight, "Choose true or false for the NeoSR degradation canvas.")
    notes.append("Blur, noise and JPEG use the local implementations. Training pairs and seeded draws can differ.")
    notes.append("The replay pool keeps full-precision targets. Set queue_lq_uint8: true to pack exactly reconstructable synthetic LQ batches. This trades a value check for lower persistent memory.")
