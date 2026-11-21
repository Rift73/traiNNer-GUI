# Image and prepared video split leakage audit

Call `studio.split_audit.audit_split_leakage(body, *, root, scratch_dir=None,
progress=None, cancelled=None)` in the analysis worker. `root` defaults to
the checkout. Pass a decoded recipe with a `datasets.train` entry and at
least one `datasets.val` or `datasets.val_*` entry. The audit covers
explicitly declared validation splits independent of the training validation
toggle. Paths accept Linux and Windows input through Studio's WSL resolver.

- `PairedImageDataset` uses trainer folder pairing, `filename_tmpl`,
  `meta_info` and its nested-to-flat LQ fallback, or LMDB `meta_info.txt`
  and its selected keys. `dataroot_gt` and `dataroot_lq` accept one path or
  corresponding root arrays.
- `PairedVideoDataset` reads prepared disk frame roots in trainer order with
  the configured positive `clip_size`. The loader's `FrameWindowIndex`
  selects scenes, LQ contexts and exact-name GT centers. An optional
  `video_scene_manifest` selects scenes by ordered root slot and checks
  their membership digests. Only frames covered by eligible windows and
  their matching targets enter the comparison. Legacy `meta_info` has no
  selection effect for this provider. Prepared paired training applies to
  windowed image restoration. Recurrent recipes can use this provider for
  evaluation. A `VideoClipDataset` training split has different evidence,
  described below.
- `SingleImageDataset` uses its LQ folder, manifest or selected LMDB
  keys. Only LQ occurrences appear in its report.
- `RealESRGANDataset` uses the actual constructor's selected GT files
  or LMDB database and key. Disk sources can have aligned
  `target_dataroot_gt` files. `RealESRGANPairedDataset` also inventories its
  reachable paired rows through `dataroot_lq` and the paired GT roots.
  The report records both branches' provider rows. Sampler enlargement
  and modulo reuse do not inflate membership. Use arrays for
  `target_dataroot_gt` and `paired_dataroot_gt` as required by the providers.
- With `task: image_classification`, `ClassFolderDataset` selects supported
  files under each first-level class directory of each `dataroot_lq`. The
  class directory name appears as `labelContext.className` in group examples.
- With `task: semantic_segmentation`, `ImageMaskDataset` selects images in
  `dataroot_lq` paired by relative stem with masks under the corresponding
  `dataroot_gt`. Ambiguous, missing or extra stems invalidate the split.
  Examples give the matched `labelContext.maskRelativePath` under the mask
  root with the same `rootId`. Mask bytes are outside the comparison.
- With `task: object_detection`, `CocoDetectionDataset` requires one
  `dataroot_lq` and its `annotation_file`. Its COCO image records select
  files in document order, including empty and ignored-only images.
  Provide recipe `task_options.categories` to validate the configured
  category mapping. Examples include `labelContext.cocoImageId` and
  `labelContext.fileName`. Annotation JSON is checked for changes.
- The three supervised providers use the disk backend. Each selected row
  contributes exactly one `image` occurrence. Class names, masks, boxes
  and IDs describe the selection and never supply image digests. An
  image re-encoded with different bytes has a different encoded identity.
- Other temporal providers, including `SingleVideoDataset`, produce an
  incomplete split with a selection issue.

The optional `groupLimit` defaults to 100 and accepts 0 through 200.
`scratch_dir` may be an existing private job folder. The worker removes
its SQLite scratch directory on completion, failure, or cooperative
cancellation. `progress(event)` receives `stage`, `completed`, `total`
(when known), and `splitId` (during selection). A true `cancelled()`
raises `SplitAuditCancelled` at a cooperative checkpoint. Native file
reads and digest updates check between chunks. Supervised
folder traversal and COCO indexing also check for cancellation. A single
COCO JSON read/decode and directory sorting can finish before the next check.

## Report

`complete` covers resolved selection and measured encoded comparison.
`hasTrainValidationOverlap` is true when confirmed, false after a
complete audit with zero matches, and null when an incomplete audit
found zero confirmed matches. `splits[]` carries each split's stable
`selectionId`, normalized root paths, selection coverage, and counts of
provider rows (`samples`), role occurrences (`inputs`) and distinct source
locators (`sources`). Two COCO IDs can select one file. Per-split
`trainValidationSamples` and `trainValidationSources` count distinct rows
and source locators in confirmed train/validation groups across the entire
audit, independent of the group display limit. `counts.sourceStates` distinguishes
`ready`, `missing`, `changed`, and `unreadable` backing sources. Changed
split selections and changed or unreadable inputs are excluded from
confirmed groups. Missing keys and files have explicit issues.

Each group records `sha256`, `encodedBytes`, occurrence and sample
counts, `bySplit` role totals, and up to eight `examples` with
split/sample/root IDs, role, and file path or LMDB path/key.
`trainValidation` flags content in train and any validation split,
`crossValidation` flags content in multiple validation splits,
`withinSplit` flags reuse across distinct provider rows in one split, and
`sameSampleRoles` flags equal content in different roles of one row.
A group may satisfy several flags. `counts` totals cover all confirmed
groups, with separate `groupsTruncated` and `issuesTruncated` indicators.

`reportId` is available for complete audits. It hashes the ordered
relative locators and keys, observed sizes, and candidate digests using
the same ordering across relocated roots. Absolute paths, physical
inodes and display limits are excluded. The ID describes the audit's
measured evidence. Only candidate sources have content digests.
Identities are measured over encoded bytes and have no perceptual
similarity meaning.

### Prepared video evidence

For a paired video split, `sampleUnit` is `window` and `inputUnit` is
`selectedFrameRole`. `counts.samples` counts eligible windows.
`counts.inputs` counts each selected LQ frame and GT center once per
scene, role and root slot. `counts.windowInputUses` includes every loader
use across overlapping windows. The `lqContext` and `gtCenter` roles can
match across splits even when names, root slots and roles differ.

Examples identify the actual split ID, scene key, numbered status, root
slot, relative path, and inclusive one-based `windowRange` within that
split. `sampleId` identifies the first window in the range. For reports
containing video, each group's `bySplit[].samples` counts the union of
affected samples in that split. Per-split `trainValidationSamples` takes
the union across all confirmed train/evaluation groups, including those
beyond the display limit. Reusing a single LQ frame in neighboring
windows creates one member and cannot generate a duplicate group by
itself. `sameSampleRoles` includes matching LQ and GT bytes used in the
same window, even when those frames have different names.

The fast scene split groups by physical device and inode. This separate
audit compares the size and SHA-256 of selected encoded bytes. A copy
with a different inode can match. Pixel similarity, re-encodes with
different bytes, and neighboring footage remain outside that measurement.

### Recurrent on-the-fly training

RSR and RDUAL train with on-the-fly degradation. `VideoClipDataset` visits
the immediate `.mkv`, `.mp4`, `.m2ts` and `.mov` entries in each configured
`dataroot_gt` in root order and uses `index/<video stem>.json` for each.
One length from `clip_lengths` is drawn per batch. Each item draws a new
start independent of its sampler index. Within-shot draws use clean shots
with positive motion-adjusted weight and room for the requested length.
When eligible cuts exist, `cut_clip_prob` controls draws crossing cuts
inside consecutive clean shots. The loader can reverse a selected span.

A training item carries decoded YUV planes, crop geometry and shot IDs.
The model constructs GT from those planes. The displayed
`<video basename>:<first frame>` identifies a draw only within its source
context and can collide across roots. Prepared evaluation gives frame
files and no source-container/PTS mapping. Container bytes, packet bytes,
filenames and adjacent PNGs cannot establish encoded GT image identity.

For this combination the report lists the configured training roots and
returns a training selection issue, `counts: null`, `complete: false`
and `hasTrainValidationOverlap: null`. Exact encoded coverage counts and
confirmed groups for supported evaluation splits remain available. The
audit does not open the training video trees or synthesize crops. The
trainer's nominal dataset length, eligible starts and actual sampled
clips are different quantities.

## Native and storage boundary

`make --no-print-directory -C native/studio/split_audit` builds
`build/clang/sha256.so` with LLVM Clang and the installed GLib SHA-256
primitive used by the checkout fingerprint tool. The worker builds it
on the first file candidate. It uses a fixed-size native buffer and
falls back to the existing native `hashlib` implementation when the
adapter is unavailable. LMDB comparisons use mapped values with a
read-only transaction and native `hashlib`. The audit leaves the LMDB
lock file untouched.

The private SQLite index keeps the comparison catalog on disk with a
fixed page cache. Provider selectors still materialize the active root
group, so peak selector memory depends on that group. For supervised
tasks the largest working selection is a class folder, an image/mask root
pair or one decoded COCO document and index. Prepared video holds the
trainer's frame-name index and sparse eligible starts for one split at a
time. Its lazy source stream represents window membership with integer
intervals in SQLite and reads each candidate file through the same disk
hash path as still images. The audit does not decode
training images, masks or Torch targets. Only sizes that
occur at least twice are hashed. A disk device/inode/stamp alias or an
identical database/key can reuse a verified digest. Singletons are
reported as size-only. File stamps, manifest stamps, disk re-selection,
and LMDB transaction IDs are checked again before reporting. This
checks a live read-only source over the audit interval and does not
freeze the filesystem.
