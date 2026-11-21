"""Inspect paired image registration independently of training or augmentation."""

from __future__ import annotations

import json
import math
import subprocess
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .native import executable
from .paths import local_path


def _origins(length: int, patch: int):
    count = min(3,length//patch)
    if count<=1:
        return [(length-patch)//2]
    return [round(i*(length-patch)/(count-1)) for i in range(count)]


def _assess(peak, patches):
    if peak is None:
        return {"assessment":"insufficientTexture","informativePatches":0}
    if peak["atLimit"]:
        return {"assessment":"searchBoundary","informativePatches":peak["patchCount"]}
    reliable = [p for p in patches if p and not p["atLimit"] and p["peak"]>=.5 and p["peakMargin"]>=.04]
    support = sum(math.hypot(p["dx"]-peak["dx"],p["dy"]-peak["dy"]) <= 1 for p in reliable)
    if not reliable or peak["peak"] < .5 or peak["peakMargin"] < .04:
        label = "ambiguous"
    elif len(reliable)>=2 and support*3 < len(reliable)*2:
        label = "inconsistentAcrossPatches"
    elif math.hypot(peak["dx"],peak["dy"]) >= .65 and peak["peak"]-peak["zero"] >= .025:
        label = "possibleShift"
    else:
        label = "noClearShift"
    return {"assessment":label,"informativePatches":peak["patchCount"],
            "distinctPeaks":len(reliable),"supportingPatches":support}


def _color_bands(image):
    if image.interpretation in ("b-w", "grey16") and image.bands in (1, 2):
        count = 1
    elif image.interpretation in ("srgb", "rgb", "rgb16") and image.bands in (3, 4):
        count = 3
    else:
        raise ValueError(
            "16-bit registration needs grayscale or RGB with optional alpha. "
            f"Decoded {image.interpretation} with {image.bands} bands."
        )
    has_alpha = image.bands != count
    return (image.extract_band(0, n=count) if has_alpha else image), has_alpha


def _float_patch(image, x: int, y: int, width: int, height: int, scale: int) -> bytes:
    """Keep fractional GT box means and put both roles on the Sobel 0..255 scale."""
    import numpy as np

    patch = image.crop(x * scale, y * scale, width * scale, height * scale).cast("double")
    if scale > 1:
        patch = patch.shrink(scale, scale)
    samples = patch.numpy().reshape(height, width, image.bands)
    if image.format == "ushort":
        samples = samples / 257.0
    rgb = np.broadcast_to(samples, (height, width, 3))
    return np.ascontiguousarray(rgb, dtype=np.float32).tobytes()


def inspect_pair(request: dict, root: Path, *, reader=None):
    gt = local_path(request.get("gt"),root)
    lq = local_path(request.get("lq"),root)
    radius = request.get("radius",6)
    if type(radius) is not int or not 2 <= radius <= 12:
        raise ValueError("Use a translation search radius from 2 to 12 low-resolution pixels.")
    given_scale = request.get("scale")
    if given_scale is not None and (type(given_scale) is not int or given_scale<1):
        raise ValueError("Scale must be a positive integer.")
    payload = bytearray()
    regions = []
    import pyvips

    try:
        with ExitStack() as files:
            sources = []
            decoded = []
            for role, path in (("gt", gt), ("lq", lq)):
                key = request.get(role + "Key")
                if key is None:
                    sources.append(path)
                    decoded.append(pyvips.Image.new_from_file(str(path), fail=True))
                else:
                    if not isinstance(key, str) or not key or not key.isascii() or "\0" in key:
                        raise ValueError("Select an existing ASCII LMDB image key.")
                    if reader is None:
                        from trainner_data.lmdb import PairedLmdbReader
                        reader = PairedLmdbReader({"type": "lmdb"})
                    encoded = reader.read(str(path), key)
                    sources.append(files.enter_context(BytesIO(encoded)))
                    decoded.append(pyvips.Image.new_from_buffer(encoded, "", fail=True))

            depths = {"uchar": 8, "ushort": 16}
            if any(image.format not in depths for image in decoded):
                raise ValueError(
                    "Registration supports unsigned 8- or 16-bit samples. "
                    f"Decoded GT {decoded[0].format}, LQ {decoded[1].format}."
                )
            bits = {role: depths[image.format] for role, image in zip(("gt", "lq"), decoded, strict=True)}
            wide = 16 in bits.values()
            if wide:
                (hi, hi_alpha), (lo, lo_alpha) = map(_color_bands, decoded)
                alpha = hi_alpha or lo_alpha
            else:
                # Retain the established Pillow color conversion and byte-rounded
                # BOX reference for ordinary 8-bit registration results.
                hi, lo = (files.enter_context(Image.open(source)) for source in sources)
                alpha = any("A" in im.getbands() or "transparency" in im.info for im in (hi, lo))
            w, h = lo.width, lo.height
            scale, rx = divmod(hi.width, w)
            if rx or scale == 0 or hi.height != h*scale or given_scale not in (None,scale):
                raise ValueError(f"Registration needs an exact equal integer scale on both axes. GT is {(hi.width,hi.height)}, LQ is {(w,h)}.")
            pw,ph = min(w,80),min(h,80)
            if min(pw,ph) < max(16,2*(radius+1)+8):
                raise ValueError("The image is too small for this translation search radius.")
            for y in _origins(h,ph):
                for x in _origins(w,pw):
                    regions.append({"x":x,"y":y,"width":pw,"height":ph})
                    if wide:
                        payload.extend(_float_patch(hi, x, y, pw, ph, scale))
                        payload.extend(_float_patch(lo, x, y, pw, ph, 1))
                    else:
                        box = (x,y,x+pw,y+ph)
                        reference = hi.crop(tuple(scale*n for n in box)).convert("RGB")
                        if scale>1:
                            reference = reference.resize((pw,ph),Image.Resampling.BOX)
                        payload.extend(reference.tobytes())
                        payload.extend(lo.crop(box).convert("RGB").tobytes())
    except (pyvips.Error, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError) as e:
        raise ValueError(f"Unable to inspect this image pair. {e}") from e
    args = [str(executable("alignment","align")),str(pw),str(ph),str(radius),str(len(regions))]
    if wide:
        args.append("rgb-f32")
    run = subprocess.run(args, input=bytes(payload), capture_output=True, timeout=30)
    if run.returncode:
        raise ValueError(run.stderr.decode(errors="replace").strip() or "Native alignment inspection failed.")
    result = json.loads(run.stdout)
    for region,patch in zip(regions,result["patches"],strict=True):
        patch["region"] = region
    evidence = {name:_assess(peak,[p[name] for p in result["patches"]]) for name,peak in result["estimate"].items()}
    return {"gt":str(gt),"lq":str(lq),"scale":scale,"radius":radius,"units":"LQ pixels",
            **{key:request[key] for key in ("gtKey","lqKey") if key in request},
            "direction":"positive dx/dy means LQ content appears right/down of downsampled GT",
            "decodedSampleBits":bits,
            "method":("Decoded RGB samples without ICC conversion, fractional GT box means, float32 on a 0..255 scale, "
                      "normalized Sobel correlation over sampled patches" if wide else
                      "8-bit RGB, area-downsampled GT, normalized Sobel correlation over sampled patches"),
            "alphaIgnored":alpha,"evidence":evidence,**result}
