# Studio dataset inspector

The first `studio.data.scan()` call builds the C++17 inspector with LLVM
Clang, `make`, `pkg-config`, GdkPixbuf, libtiff, and libwebp. To build
explicitly, run `make -C native/studio/data` from the checkout root in WSL.
The ignored `native/studio/data/build/clang/inspect` executable is reused
while it is current. The build records the compiler version and target in
`build/clang/toolchain.txt`.
`studio.data.preview()` uses Pillow from the existing trainer Python environment.

## Scan request

Call `studio.data.scan_request(body)` with `gt` and optional `lq`, or with
`gtRoots` and optional `lqRoots`. The root arrays pair by position. `scale`,
`gtPatchSize`, `filenameTemplate`, and `phase` are optional. `phase` defaults
to `train` and also accepts `val` or `test`.

- `source: folder` is the default. The recursive inventory follows symlinked
  directories, skips ancestor cycles and dot-prefixed entries, and includes
  the visible regular files the trainer enumerates. It does not filter by
  image extension before matching.
- `source: meta_info` takes a `metaInfo` path and two sides. It uses the
  trainer's explicit records, including its nested-to-flat LQ lookup.
- `source: lmdb` takes two sides with each database's own `meta_info.txt`.
  It lists the native reader's selected keys and opens actual encoded values
  for reported samples. `headerRowsChecked` is the number of manifest rows
  whose headers were probed. For preview, call `preview_lmdb(path, key, max_size)`
  with the sample's database path and `gtKey` or `lqKey`.

The folder matcher applies `filenameTemplate` to the GT basename without its
final extension. It expects that LQ stem in the same relative directory and
permits a different final extension. The default `{}` requires both stem sets
to match. Custom templates can select a subset and report unmatched files as
warnings. Duplicate stems and a template selecting one LQ for several GTs
are ambiguous. Templates with Python formatting options use Python's exact
formatting to supply expected names to the native matcher.

Train geometry requires exact GT = LQ x scale for pairs with known dimensions.
Validation can crop excess GT at the top-left and reports insufficient GT
dimensions as errors. A supplied GT patch size enables minimum size checks
for training. `validationRoots` on folder requests compares physical file
identity across the explicit validation roots and selected dataset files.
This restoration scan treats segmentation separately. Use the label-aware CPU
inspector below for image/mask pairs.

Each `roots[]` entry has an ordinal string `id` with its own counts. Samples
and root-specific issues carry `rootId`. A single folder keeps its relative
filename as the sample ID. Multi-root IDs include the root ID. `counts.paired`
describes unambiguous selections, and `ok` reports whether any error was found.
`geometry` on a paired sample is `exact`, `crop`, `mismatch`, or `unknown`.
Manifest counts describe the trainer's selected records independently of
header errors. Native folder probes read common image headers and leave
unsupported dimensions unknown. Header checks cover dimensions and selected
paths, and pixel decoding is exercised when a preview is requested.

`limits` reports capacity for 100 samples and 200 issues. `totals` reports the
full row counts, and `truncated` flags a capped response. Previews use Pillow
and fit within 1,536 pixels on the longest edge.

## Label-aware CPU inspection

Call `studio.task_data.inspect_task_data(body, root=checkout, progress=None,
cancelled=None)` in a job worker. Pass the recipe-shaped `task`,
`task_options`, and `datasets` mapping. Each split uses its trainer path
fields. Paths are normalized in WSL relative to the supplied root and
inputs are opened read-only.

| `task` | Split fields | Label settings |
| --- | --- | --- |
| `semantic_segmentation` | `dataroot_lq` images, `dataroot_gt` masks, and `gt_size` or `lq_size` for training crops. Root arrays match by position. | `class_names` in ID order and `ignore_label` (default 255). Without names, the observed non-ignored IDs are reported as unmapped. |
| `image_classification` | `dataroot_lq` with immediate class subdirectories. Also accepts a root array. | `class_names` in output order. Omit to discover folder names with null IDs. |
| `object_detection` | One image root in `dataroot_lq` and an `annotation_file` in COCO JSON format. | `categories` in foreground label order, with `id` and `name`. Omit to inventory declared categories without assigning output labels. |

Optional `auditLimit` defaults to 100 per split. Zero inventories paths
and metadata only. Null audits every eligible row. A finite budget uses
a reproducible reservoir from the whole split. `sampleLimit` accepts
0 through 100 and defaults to 25 audited rows per split. Sample and issue
caps affect displayed evidence. Progress events carry `stage`
(`inventory` or `auditing`), `splitId`, `completed`, and `total`
(null while discovering rows). A true `cancelled()` raises
`TaskDataCancelled` at the next cooperative checkpoint.

The result contains `ok`, overall coverage `complete`, `splits[]`,
`issues[]`, `totals`, and `limits`. Each split provides `counts`,
`inventoryComplete`, `audit` with `method`, `eligible`, `attempted`,
`completed`, and `complete`, plus capped `samples`. A geometry mismatch
can be fully audited and still produce an error. When a tree is
unreadable, its counts describe the visible records and
`inventoryComplete` flags the partial inventory.

- Segmentation counts only unique same-relative-stem disk pairs as audit
  candidates. Geometry tracks header-size equality and train-crop fit.
  `maskPixels` measures the first frame of successfully audited masks,
  preserving integer IDs, including paletted PNG indices and 16-bit
  labels. Class, ignored, and unknown pixel totals are audited totals.
  `unknown` or `unmapped` includes exact pixel counts for up to 256
  observed IDs and the exact remainder in `unlistedPixels`.
- Classification counts supported paths per class and per directory
  across each whole split, flags empty directories and unlabeled files,
  and compares class directories with the `train` split when supplied.
  Header audits verify that a file has a readable still-image header.
- Detection counts the full COCO manifest, verifies resolved image
  paths, and clips and checks referenced boxes against declared sizes.
  `emptyImages` have zero annotation rows. `ignoredOnlyImages` have
  accepted boxes and every box is flagged with `iscrowd` or `ignore`.
  `imagesWithoutUsableBoxes` also includes empty images and images whose
  boxes are invalid. Duplicate IDs are reported and the first record is
  used for the corresponding inventory. Header audits compare file
  geometry with COCO declarations. Sample boxes are clipped pixel
  `xyxy`, with up to 20 boxes per audited sample.

The mask worker builds `make -C native/studio/data mask-histogram` once
on first eligible PNG use. LLVM Clang links the installed libpng. The
kernel counts noninterlaced grayscale or paletted PNGs by row, preserving
the trainer's decoded values. The existing Pillow decoder covers TIFF,
interlaced PNG, and PNGs that the native path cannot read. The per-split
`maskPixels.engines` counts record which path supplied the histogram.

## Optional native paired-folder selection

`make -C native/studio/data paired-folders` enables the native path-only
selection for the trainer's default `{}` folder matching in this checkout.
It keeps root order and sorts relative stems within each group. It uses the
ordinary trainer path for custom templates, unavailable or incompatible
binaries, undecodable filenames, or folder errors. Training does not build
the inspector or open images for header checks during path selection.

## Paired-video training

`make -C native/studio/data paired-video` enables native LQ clip assembly for
`PairedVideoDataset` training. It uses the same Clang toolchain and requires
only the C++ standard library. For supported RGB byte patches it writes
directly to the final contiguous float32 TCHW tensor, using normalization
values from the existing PyTorch conversion. Decode, crop, augmentation,
validation, and worker collation keep their existing paths. An absent or
incompatible library, or a clip with other dtypes or pixel layouts, uses the
existing conversion and stack.

## Registered native paired crops

For PairedImageDataset training, `native_paired_crop: true` uses the existing libvips decoder and its ICC handling. The cropper draws the same flip, rotation and crop coordinates, fetches the corresponding square on the original source canvas, and packs the oriented bytes directly into the final float32 RGB tensors with the Clang kernel. Coordinates are multiplied by the scale for the GT. It handles 8-bit grayscale, RGB and RGBA. Other decoded formats take the ordinary transform path before any crop randomness is consumed. Run `./trainner-gui prepare` once when enabling this option. The default retains the established loader pending end-to-end performance measurements.

With `datasets.train.native_paired_crop: true`, a paired still-image loader decodes its usual color-managed libvips input, selects the registered LQ/GT windows, and fills planar float RGB directly into the outgoing batch. Loader workers write the shared Torch buffers used for transfer. Zero-worker and external per-sample callers also work. This option requires `./trainner-gui prepare`. Other source formats retain the existing conversion.

## Ordered paired training identity

The checkpoint integration can call
`trainner_data.paired_identity.capture_paired_training_identity(train_loader.dataset)`
once at launch when `checkpoint_rng` is enabled with the built-in training
`PairedImageDataset`. Save the returned plain dictionary with the random
streams and sampler cursor. A strict resume can pass the saved dictionary
and the new capture to `require_paired_training_identity(saved, current)`
before restoring the streams. Missing or incompatible identities, including
legacy checkpoints, raise `PairedTrainingIdentityError`. Resuming with
`checkpoint_rng: false` uses fresh streams. Preflight can call
`paired_replay_settings(dataset_options, scale=recipe.scale)` for the same
settings checks without touching data.

Version 1 returns `version`, `provider`, `storage`, `pairs`,
`selection_sha256`, `source_metadata_sha256`, and `settings`.
The first digest covers the actual ordered GT/LQ paths or database/key
pairs, including duplicate rows. The settings include the resolved scale
and crop sizes, flip and rotation flags, native crop selection, color
option, and normalization. Loader geometry, random streams, and model
settings have their own checkpoint fields.

For disk folders and `meta_info`, the second digest covers each selected
file's device, inode, encoded length, modification time, and change time.
The installed Clang adapter batches the metadata calls into bounded
buffers. Its Python fallback serializes the same fields. LMDB uses the
corresponding `data.mdb` metadata and the committed transaction ID of each
selected database. The LMDB metadata probe opens a separate read-only
process and does not read record payloads or touch the lock file.
A committed change elsewhere in the same database can also invalidate
replay.

These are metadata identities. Path changes, relocation, or re-encoding
can invalidate replay even when the decoded pixels match. Changes that
preserve all captured metadata may evade detection. The capture reflects
startup observations on a live filesystem; it does not freeze the sources.
The caller should retain the startup result and reuse it at checkpoint time.
The capture opens no images and needs no per-update scan.

The optional disk adapter builds on first capture, or explicitly with
`make --no-print-directory -C native/studio/data paired-identity`.

## Native RGB sample packing for built-in tasks

`make -C native/studio/data task-sample-pack` builds
`build/clang/task_sample_pack.so` with LLVM Clang. Classification, indexed
semantic segmentation, and COCO detection use it when available. The
packer reads decoded RGB views, including signed strides left by crop and
augmentation, and writes normalized float32 CHW directly. Detection uses
planar storage. Classification and segmentation retain their interleaved
channel layout. The existing image transforms, mask conversion, and box
handling run normally. A missing or incompatible library uses the original
Torch conversions. Pillow still supplies the decoded byte buffer.
