# Paired restoration train/validation manifests

Call `studio.dataset_split.split_dataset(body, destination, *, root, progress=None, cancelled=None)`
from the job worker. The default root is the checkout. Give the worker a **new**
destination under an existing private parent outside both source directories.

```json
{
  "dataset": {
    "name": "Restoration images",
    "type": "PairedImageDataset",
    "dataroot_gt": ["datasets/train/dataset1/hr"],
    "dataroot_lq": ["datasets/train/dataset1/lr"],
    "filename_tmpl": "{}"
  },
  "seed": 42,
  "validationFraction": 0.2,
  "groupBy": "pair"
}
```

The optional task is `image_restoration`. Each disk root accepts one path or a
one-element array. The worker uses the loader's folder selection or the source
dataset's existing `meta_info`, preserving selected order and repetitions
within each output. Unmatched files omitted by the trainer's custom-template
selector remain outside the population. Membership inspection reads filenames
and file metadata. Image decoding, crop suitability and comparisons of copied
or re-encoded images are separate diagnostics.

## Assignment

- `pair` (default) groups by the full GT-relative filename, including extension.
- `parent` keeps all rows with the same complete immediate GT parent together.
- `scene` keeps an entire top-level GT directory together across its descendants.
  In both directory modes, directly rooted files form one group named `.`.
  Scene names embedded in flat filenames require an explicit directory layout.

The fraction must lie strictly between zero and one. The worker rounds the
requested validation row count half-up and clamps it to leave both sides
nonempty. At least two independently assignable groups are required.
Repeated rows retain their weight. SHA-256 of the algorithm tag, integer seed
and relative group name determines group priority independently of absolute
root paths and filesystem enumeration. The smaller side is filled in that order,
with whole groups. A closer single group or one oversized group can improve the
result. Grouped allocation is approximate. The result includes the target,
achieved counts and any adjustment notes.

## Outputs and reuse

`manifests.train` and `manifests.val` point to UTF-8 GT-name files.
`datasets.train` and `datasets.val` contain the input dataset options with
absolute roots, distinct names and the corresponding `meta_info`. The workbench also
writes a recipe with the new partitions and preserves the existing evaluation sets.
`recipeDatasetKeys` identifies the generated entries. Their validation key is
`val` when available, followed by `val_split`, `val_split2` and so on.
`split.json` records this result, including `manifestSha256`, algorithm, seed,
grouping and per-side group and row counts. Moving the roots or manifest
directory requires updating the recipe paths. Manifest entries stay relative.

The trainer applies a manifest to every root pair and ignores external manifests
in LMDB mode. Both layouts are rejected here. The generated entry must resolve
to the exact LQ using the template and GT extension without depending on a
shadowable legacy flat fallback. Distinct GT entries sharing a physical source
file are rejected. Repeated occurrences of one pair stay together. Keep names
and pair layout stable when reusing a manifest. Its SHA-256 identifies the ordered manifest text.

Before publication the worker rereads both manifests with the trainer's reader
and verifies each selected GT/LQ path and observed file metadata. Files added
after initial selection are outside the saved population. The three output files
are published together with a no-overwrite rename. `progress(event)` receives
`stage`, `completed`, and `total` (null during initial discovery). A true
`cancelled()` raises `DatasetSplitCancelled` before publication. The provider's
initial scan and the group sort finish before the next cancellation check.
