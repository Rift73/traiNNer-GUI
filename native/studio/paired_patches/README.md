# Paired PNG patch extraction

`studio.dataset_patches.extract_paired_patches(body, destination, *, root,
progress=None, cancelled=None)` takes a decoded disk `PairedImageDataset` in
`body.dataset`, positive integers `scale`, `cropSize` and `stride`, and an
optional `task` of `image_restoration`. `root` defaults to the checkout.
`validate_patch_options(body)` returns the three integers in that order.
Use a new destination under an existing private parent outside the source roots.

Source rows follow the trainer's folder selector or `meta_info` reader,
including nested names, filename templates, root order and repeated entries.
One manifest applies to every declared root pair according to trainer rules.
Each selected occurrence receives its own output directory under both roots.
For a train/validation workflow, extract each already-selected split separately
so overlapping crops from the same source stay together.

## Coordinates and formats

Crop size and stride use **LQ pixels**. Both axes start at zero, advance by
the stride, and include the last valid origin exactly once. For axis length
`N`, crop `C` and stride `S`, the count is `1 + ceil((N - C) / S)`.
A 128-pixel axis with a 53-pixel crop and a 45-pixel stride starts at
0, 45 and 75. A pair requires `1 <= S <= C <= min(LQ width, LQ height)`
and exact integer-scale GT dimensions. The GT rectangle is the LQ rectangle
multiplied by scale. These rules preserve an existing registration. They
cannot detect or correct a shifted source. There is no padding or resampling.

The kernel copies the original 8- or 16-bit grayscale, RGB or RGBA sample
bytes into losslessly encoded PNGs. Each side retains its own depth and layout.
It preserves the original encoded ICC, sRGB, gamma, chromaticity, significant
bits, pixel density and HDR color chunks when present. Text and other ancillary
metadata are omitted. Palette, packed low-bit-depth, grayscale-alpha,
color-key transparency, interlacing, APNG, EXIF, stereo layout and embedded
image offsets produce an error. The ordinary RGB trainer may discard alpha
when consuming the saved files.

## Artifacts and limits

- `gt/` and `lq/` contain matching keys. `meta_info.txt` records trainer order.
- Each line in `patches.jsonl` gives the output `key`, source row/root IDs,
  relative GT/LQ filenames, observed file versions, dimensions, channel counts
  and bit depths. `lqBox` and `gtBox` are top-left `[x,y,width,height]` in the
  corresponding stored source grid. Root IDs are one-based.
- `dataset.json` contains the scale, grid settings, counts, original root paths
  for provenance, and a portable `dataset` selection with paths relative to the
  artifact directory. Resolve those paths against the moved directory when
  placing the selection in another recipe. Keep its training crop within the
  exported patch and use the recorded recipe scale.
- The result returns `sourcePairs`, `patchPairs`, `gtPath`, `lqPath`,
  `manifestPath` (coordinate JSONL), `metaInfoPath`, `reportPath` and the
  immediately usable `dataset` selection with absolute paths.

Extraction reads each role once per source occurrence, using one decoded band
of source width by crop height, capped at 256 MiB, plus libpng buffers.
Source paths and version stamps occupy memory proportional to source selection.
Each coordinate produces two PNG files and overlap can substantially increase
output size. The kernel processes one output file at a time. Progress events
include `stage`, `completed`, `total`, `sourceRow` and `completedPatchPairs`.
Within `lq` or `gt`, completed/total count patches for the current role.
Cooperative cancellation raises `DatasetPatchesCancelled`. The worker checks
open-file and path versions, closes the staged outputs, then publishes the
directory using a no-overwrite rename. Cancellation and ordinary failures
clean up staging.

`make -C native/studio/paired_patches` builds `build/clang/patches.so` with
installed Clang and libpng. The prepare entry is
`executable("paired_patches", "patches.so", source="patches.c")`.
