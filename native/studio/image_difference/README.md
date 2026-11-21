# Saved validation PNG difference

`studio.image_difference.compare_pngs(left, right, output_dir, *, versions,
reference=None, region=None, progress=None, cancelled=None)` runs on CPU.
Supply the recorded `artifact_version(stat_result)` for each supplied input
in `versions`, keyed by `left`, `right` and optionally `reference`. The
worker opens regular files once, reads those descriptors, and rechecks both
descriptor and path versions before publishing.

All inputs must have identical dimensions, channel layout and bit depth.
Supported PNGs are single-frame, noninterlaced, opaque grayscale or RGB,
8 or 16 bits per channel. Samples retain PNG channel order and integer
values. Color management, orientation and resampling are outside this
measurement. The optional rectangle `{x, y, width, height}` uses integer
source pixels, top-left origin, with the right and bottom edges excluded.

The result includes full `width` and `height`, measured `region`, `pixels`,
`bitDepth`, `channels`, `units`, and input paths and versions. `leftVsRight`
contains `changedPixels` (any channel differs) and `channels[name]` with
`mae`, `rmse` and `max` in stored sample levels. Grayscale uses `Y`.
With a reference, `leftVsReference` and `rightVsReference` have the same
structure. `closerPixels.left`, `.right` and `.tie` compare the sums of
squared channel errors to the reference at each pixel.

`outputs.absoluteDifference` is the absolute path of the ROI-sized PNG.
Each output channel stores the original absolute difference directly,
with the input bit depth. One level of difference is one output level.
The worker creates the output directory if needed. It publishes
`absolute-difference.png` atomically without overwriting an existing file.
Use a separate output directory per job on a filesystem supporting hard
links. Ordinary failures and cancellation remove the staged file.

`progress({stage: "comparing", completed, total})` counts decoded source
rows. The decoder reads through the full PNG even when measuring a crop.
A true `cancelled()` raises `ImageDifferenceCancelled`. The native loop
checks callbacks at row boundaries and the worker checks before publication.

`make -C native/studio/image_difference` builds `build/clang/difference.so`
using installed LLVM Clang and libpng. Studio can prepare it with
`executable("image_difference", "difference.so", source="difference.c")`.
