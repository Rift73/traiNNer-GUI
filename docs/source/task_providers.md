# Task recipes and evaluation

Copy a complete recipe from `options/_templates/train/` to `options/train/` before training. Each recipe declares a task and a generator independently. The same task can use another architecture that satisfies its model contract.

To evaluate a checkpoint, use `python test.py -opt evaluation.yml`. Set `num_gpu: 0` for CPU evaluation. Supply one or more `val` or `test` datasets, set `val.val_enabled: true`, and enable metrics or image output as supported by the task. For restoration, segmentation, classification, and detection, supply `path.pretrain_network_g`. For an autoencoder, supply the complete model under `path.pretrain_network_ae` alongside the corresponding `network_ae` recipe. Use a dataset with ground-truth images for reconstruction. The evaluator reuses the existing results directory.

Recurrent evaluation uses full frame sequences. Use `val.tile_size: 0` and the generator's required clip window in `datasets.*.clip_size`.

## One-channel still-image restoration

The ordinary prediction and export commands also accept a generator trained on one luma plane. For Compact, set `network_g.num_in_ch: 1` and `network_g.num_out_ch: 1`. Keep the default `rgb` pixel-format fields when the dataset already supplies luma. Prediction uses `datasets.train.color` if a training split is present. A validation- or test-only recipe must specify the same encoding across its splits.

| Dataset color | RGB file input encoding | Identity model PNG codes for black and white |
|---|---|---|
| `luma601` | Full-range BT.601, computed in float as `0.299 R + 0.587 G + 0.114 B` from normalized sRGB | 0 and 255 |
| `y` | Studio-range BT.601. For 8-bit files, round `16 + 65.481 R + 128.553 G + 24.966 B` to an integer code before dividing by 255, matching the native single-image loader. | 16 and 235 |

Use `python -m traiNNer.tasks.predict run --config recipe.yml --checkpoint net_g.safetensors --input image.png --output result.png`. A folder input takes a separate output folder. Each result is an 8-bit, single-channel PNG. Saving clamps the prediction to [0, 1] and rounds to the nearest byte. Studio-range output retains its code values. The metadata reports `output_kind: luma`, `luma_encoding` and whether the file adapter rounds 8-bit Y before inference.

Use `python -m traiNNer.tasks.predict export --config recipe.yml --checkpoint net_g.safetensors --height 32 --width 48 --output model.onnx`. Supply already encoded FP32 luma to `input` as NCHW `[1, 1, H, W]`. The output `luma` is FP32 `[1, 1, H*scale, W*scale]` in the same signal domain, clamped to [0, 1]. Compact also accepts `--dynamic-spatial`. The file decoder and PNG encoder run outside the graph. Grayscale files passed to the file predictor are decoded as source images and receive the selected conversion, including its offset for studio-range Y.

A recipe that selects both `input_pixel_format: y` and `output_pixel_format: y` can omit dataset color. This path computes float studio-range Y directly. File prediction normally writes the resulting Y plane. Per-dataset mean/std normalization needs an explicit deployment adapter.

For an RGB result from a Y-only or UV-only still-image generator, set `val.complementary_planes: input_bicubic`. During validation, prediction and ONNX deployment the model supplies its learned planes. The other planes come from the RGB input, upscaled with bicubic interpolation, clamped to [0,1] and converted to studio-range BT.601. Prediction and export both take RGB and return RGB. This permits evaluation on images without ground truth and RGB metrics on held-out pairs using only the information available at deployment. With the default `target` policy, paired validation obtains the missing planes directly from the target. Its RGB scores measure a different reconstruction protocol. Dataset-level `color: y` and `luma601` remain single-channel workflows.

## Fixed-window RSR export

Use a recurrent recipe with `network_g.type: rsr_compact`, `network_g.temporal_read` set to `block` (the default) or `aligned`, RGB input and output, and full-frame evaluation (`val.tile_size: 0`). Legacy recipes that select RSR by architecture also work. Supply the trained generator checkpoint and the low-resolution frame dimensions.

```bash
python -m traiNNer.tasks.predict export \
  --config rsr.yml --checkpoint net_g.safetensors --output center.onnx \
  --clip-size 3 --height 32 --width 40 --opset 17
```

`clip_size` is required. The graph takes one complete window from a single scene, with identically sized frames in chronological order. Feed `input` as FP32 RGB in [0, 1], layout **NTCHW** (`[1, T, 3, H, W]`). The output `rgb` is the raw FP32 centre frame in **NCHW** (`[1, 3, H*scale, W*scale]`). The zero-based centre index is `T // 2`, including even windows. Clamp to [0, 1] and quantize when encoding an 8-bit image. To reproduce directory prediction, slide complete windows within each scene and omit boundary frames without a complete window.

Batch, clip length and spatial dimensions are fixed at export. Each invocation starts with a fresh recurrent state. The export result and the ONNX model's `trainner.*` properties record the read mode, layouts, shapes, precision, and centre index.

For `aligned`, supply a generator checkpoint containing the loaded frozen NeuFlow v2 weights. For windows with adjacent frames, the graph computes bidirectional flow and consistency from the supplied RGB frames. A one-frame window has no temporal neighbour. Aligned RSR requires opset 16 or later. When a window has two or more frames and its shorter input side is at least 512 pixels, use `--opset 18` or later. The graph downsamples only the flow inputs, using the eager model’s rounded dimensions, half-pixel sampling coordinates and antialiased bilinear edge weights. The restoration trunk still sees the full frame. RDUAL supports eager sequence prediction.

For example, a fixed 720p input uses `--height 720 --width 1280 --opset 18` and a 360×640 flow grid. A 1080p input uses `--height 1080 --width 1920 --opset 18` and a 270×480 flow grid. Supply the clip length for each export. Full-frame memory requirements still depend on the shape and window length.

CPU ONNX Runtime parity was checked at ×2 and opset 17. The block-read check used three 32×40 frames. The aligned check used three translated 40×48 frames, the real frozen NeuFlow weights and nonzero temporal blends. The maximum absolute difference from the eager centre was `1.52e-5` or less. Zeroing the eager flow displacements changed the centre by up to `0.718`.

At the antialias switch, the ordinary opset 18 export was also checked on two 512×513 frames with a known (6, 4) pixel translation. Eager NeuFlow measured a median (3.013, 1.993) pixel displacement inside the rounded 256×256 flow grid. CPU ONNX Runtime’s 1024×1026 centre frame matched the eager centre within `9.30e-6` maximum absolute error.

## Semantic segmentation

Start from `options/_templates/train/SemanticSegmentation/LRASPP.yml`. Each RGB file has a matching mask with the same relative stem. Masks store integer IDs. The list order in `class_names` defines those IDs.

```yaml
task: semantic_segmentation
task_options:
  class_names: [background, foreground]
  ignore_label: 255
datasets:
  val:
    name: segmentation validation
    type: imagemaskdataset
    dataroot_lq: datasets/segmentation/val/images
    dataroot_gt: datasets/segmentation/val/masks
```

Validation reports per-class IoU and mean IoU over the labeled pixels. Headless prediction produces single-channel PNG masks containing class IDs.

New segmentation checkpoints record the ordered `class_names` in both safetensors and PyTorch formats. A declared class order must match the recipe in training, fine-tuning, evaluation, prediction, and ONNX export. `path.strict_load_g: false` permits partial tensor loading for fine-tuning with the same declared class order. Resume and headless deployment require a matching tensor layout.

Legacy checkpoints without class names still load with a warning. Check the recipe's class order against the source model before using them. Prediction and ONNX export report `class_order_verified` alongside `class_names`. It is false for legacy weights and true when the saved list matches.

## Supervised training weights

For classification, segmentation and detection, positive `train.ema_decay` keeps an averaged network for validation. The numbered `net_g_ema` file is the evaluated network. Resume also needs the corresponding raw `net_g` file and training state. With `train.ema_mode: neosr`, the first applied update initializes the average and later updates copy the current network buffers. Use `train.optim_include_params_g` and `train.optim_exclude_params_g` to fine-tune selected parameter names. These filters control optimizer parameters; BatchNorm running statistics still follow the forward's training mode.

## Whole-image classification

Start from `options/_templates/train/ImageClassification/MobileNetV3Small.yml`. Put images under folders named by class. In this excerpt, a path such as `datasets/classification/val/dog/frame.jpg` has class ID 0.

```yaml
task: image_classification
task_options:
  class_names: [dog, cat]
  image_size: 128
datasets:
  val:
    name: classification validation
    type: classfolderdataset
    dataroot_lq: datasets/classification/val
```

Training, validation, and file prediction use a whole-image resize to `image_size`. Validation reports top-1 and any enabled valid top-k ranks. Headless prediction writes a JSON file containing one class ID, class name, and ordered raw-logit vector per input.

## Object detection

Start from `options/_templates/train/ObjectDetection/FasterRCNNMobileNet320.yml`. Per split, specify an image directory and a COCO annotation file.

```yaml
task: object_detection
task_options:
  categories:
    - {id: 7, name: vehicle}
    - {id: 12, name: person}
network_g:
  type: fasterrcnn_mobilenet_v3_large_320_fpn
  num_classes: 3
datasets:
  val:
    name: detection validation
    type: cocodetectiondataset
    dataroot_lq: datasets/detection/val/images
    annotation_file: datasets/detection/val/annotations.json
```

An `annotations.json` matching the recipe above can contain this single labeled image.

```json
{
  "images": [{"id": 1, "file_name": "street.png", "width": 640, "height": 480}],
  "categories": [{"id": 7, "name": "vehicle"}, {"id": 12, "name": "person"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 7, "bbox": [40, 120, 160, 100]}
  ]
}
```

COCO `images[].file_name` is relative to the image directory. The declared width and height match the stored file. COCO boxes use pixel `[x, y, width, height]` from the top left. The loader converts them to `[x_min, y_min, x_max, y_max]`, clips partial boxes to the image, and keeps empty targets for images with no annotations. Training flips and transposes move pixels and boxes together. The detector owns its whole-image resizing.

The ordered category map assigns foreground labels 1 and 2 to COCO IDs 7 and 12. The model includes background label 0. Its checkpoint records the COCO IDs and class names in that order.

Validation reports `AP50` over categories with evaluable targets, `AP50/<class>`, `recall50`, `false_positives50`, and `targets`. AP50 uses confidence ranking and one-to-one matching within image and category at IoU 0.50. Recall and false-positive counts pool detections across categories at that same IoU. Crowd and ignored boxes are excluded from training targets. An otherwise unmatched detection overlapping an ignored same-category box at IoU 0.50 or higher is excluded from scoring. A split with no evaluable targets reports the counts and has no AP50 or recall value.

```bash
python -m traiNNer.tasks.predict run \
  --config detection.yml --checkpoint net_g_1000.safetensors \
  --input images --output detections.json --score-threshold 0.25 --top-limit 100
```

The input can be one still image or a directory. The JSON contains `predictions` in input path order. Each entry records the input path in `input`, plus `file_name`, `width`, `height`, and `detections`. Directory filenames are relative to the input root. Each detection has `box_xyxy` in original image pixels, `category_id` from COCO, contiguous `class_id`, `class_name`, and `score`. The score cutoff is exclusive and detections are ordered by decreasing score. The cutoff defaults to `network_g.box_score_thresh` or 0.05 when omitted. The default limit is 100 per image.

For a ledger-selected checkpoint, pass `--checkpoint-signature DEV INO SIZE MTIME_NS CTIME_NS`. The command checks the signature before and after loading, verifies the recipe's ordered COCO IDs and class names against checkpoint metadata, and records the loaded signature and checkpoint path in JSON. The `traiNNer.tasks.predict.predict` function accepts the same sequence in `checkpoint_signature` and takes `score_threshold` and `top_limit` as keyword arguments. Headless detection supports TorchVision Faster R-CNN in eager FP32 on CPU. The export command rejects detection until a portable ONNX graph with a dynamic detection count is verified.

## Task actions

The Studio holdout action can save a detector's raw results using `saveDetections: true` or `trainner-gui evaluate RECIPE WEIGHTS --save-detections`. Each split receives a COCO bbox result file containing original image IDs, category IDs, pixel xywh boxes, and confidence. External evaluators can consume it alongside the original annotation file. The Clang evaluator prepared by `trainner-gui prepare` streams each image into the trainer's AP50 calculation. Without this library the trainer uses its Python reference scorer.

The optional `output` field on a task action describes what a client should request or write. Prediction handlers accept `image`, `json`, or `sequence`, with `image` as the default. Holdout handlers return `provider_scores` by default. Set `reference_scores` if the recipe supplies restoration metrics and supports saving reconstructed images. ONNX export always uses `onnx`.

`trainner_config.tasks.TRAINING_TASKS[task].actions` exposes the default actions. Use `task_actions(opt)` to select for the architecture in a typed recipe, including legacy recipes that omit `task`. `require_task_action(opt, action)` returns the selected action or raises `ValueError` before the runtime handler is loaded. These lookups import neither the model factory nor the action handlers.

An action's `inputs` lists alternative input kinds. A particular recipe still needs to meet its provider's dataset, checkpoint, and architecture requirements.

| Task or architecture family | Train | Validate and holdout | Predict | ONNX export |
| --- | --- | --- | --- | --- |
| Still-image restoration | `image_pairs`, `images` | `image_pairs`, `images` | `images` | `images` |
| Windowed restoration | `sequence_pairs`, `sequences` | `sequence_pairs`, `sequences` | `sequences` | Unavailable |
| Recurrent restoration | `sequences` | `sequence_pairs`, `sequences` | `sequences` | `sequences` for RGB `rsr_compact` (block or aligned, subject to the fixed-window limits above) |
| Autoencoder | `image_pairs` | `images`, `image_pairs` | `images` | `images` |
| Semantic segmentation | `image_masks` | `image_masks` | `images` | `images` |
| Image classification | `class_folders` | `class_folders` | `images` | `images` |
| Object detection | `coco_boxes` | `coco_boxes` | `images` | Unavailable |

`holdout` evaluates the configured `val` or `test` splits against an explicit checkpoint. Training `validate` uses the model's existing validation hooks. Sequence prediction takes numbered still frames in a directory and emits the center frames for complete windows.

## Provider extension

Call `trainner_config.tasks.register_task` with a task ID and `TrainingTaskSpec` from a lightweight `trainner_config/*_task.py` declaration or from the callback exposed by a `trainner.training_tasks` entry point. The declaration can define typed task settings and must name its runtime factory as `module:callable`. The trainer invokes the factory with `ReduxOptions` and the decoded settings.

A provider declares `dataset_geometry` as `aligned_patches`, `whole_images` or `records`. Whole-image and record datasets receive decoded settings as a second constructor argument during training and evaluation. A dataset that declares `collate_fn` can preserve variable-size tensors and annotations. Providers register their datasets and architectures when their runtime module is imported. Training and holdout evaluation load the selected provider before constructing datasets. A caller performing its own registry lookup can call `traiNNer.tasks.load_task_provider(TRAINING_TASKS[task_id])` first. Tools enumerating the whole catalog call `load_task_providers()`. Schema generation includes typed task settings and usable dataset names.

Use the existing model hooks for optimizer setup, training updates, checkpoint saves, and validation. Publish the current dataset’s scalar observations in `metric_results`. The trainer clears it before each dataset and the workbench records scores emitted by the provider even when the recipe omits `val.metrics`. Leave it empty when a split has no measurable target. Keep task-specific semantics, including label order, with the checkpoint.

Declare actions with `TaskActionSpec(inputs, handler=None)` in the provider's `TrainingTaskSpec.actions`. Optional `architecture_actions` maps lowercased architecture names to complete replacement action mappings. Older declarations that omit actions retain the model's training and validation hooks with unspecified input kinds. Declare each standalone action with a named `module:callable` handler.

- **Holdout** receives `(opt, *, on_dataset=None)`. The caller prepares the datasets, checkpoint path, device, and results paths in `ReduxOptions`. The handler returns a list of dataset reports with `dataset`, `samples`, `metrics`, and `nonfinite`, and can call `on_dataset` with progress events. Use `traiNNer.evaluation:evaluate` to opt into the generic evaluator.
- **Prediction** receives `(opt, checkpoint, source, destination, **supplied_flags)`. File arguments are `Path` objects. The prediction entry point passes optional flags only when supplied. Return a JSON-compatible dictionary with an `outputs` list of file paths and the task's output metadata.
- **Export** receives `(opt, checkpoint, destination, *, opset)`, plus `height` and `width` for spatial models. A caller specifying a window length also supplies the keyword argument `clip_size`. Write a portable single-file ONNX graph at `destination` and return a JSON-compatible dictionary with `output` equal to `str(destination)` and the graph's input and output contract.

The dispatcher supplies `opt.is_train=False` to standalone handlers. Call `resolve_task(opt)` to decode and validate provider-specific settings. Handler modules can load their runtime dependencies when called. Preserve the checkpoint's label IDs and class order during deployment and evaluation. The matching checkpoint loader can verify ordered `class_names` and COCO category IDs.

### Numeric regression

The built-in `tabular_regression` provider accepts named numeric records. The example at `options/_templates/train/TabularRegression/Tabular_MLP.yml` runs on CPU using the included synthetic rectangle data. For your data, point each split's `meta_info` at a UTF-8 CSV with a header. List the feature and target columns in network order in `task_options`. Extra CSV columns are ignored. Values keep their original units and must be finite after conversion to float32. Training loads selected columns into host RAM once per dataset and optimizes mean squared error.

The built-in MLP accepts a list of hidden widths (`[]` gives linear regression). A local architecture may take its place if its float input is `[batch, features]` and its output is `[batch, targets]`. Saved weights carry the column order and loading checks it. Validation reports mean absolute error and root mean squared error across every target value. Keep targets in comparable units when pooling them. Recorded comparisons check the actual selected row values. Prediction only requires the feature columns. ONNX export takes a dynamic batch of float32 records, with the ordered column names recorded in model metadata.

For numeric CSV evaluation, `datasets.val.batch_size_per_gpu` controls how many records share a forward pass. Its default is one. The example uses 128. Metrics pool target values across every batch, including a short final batch. Classification also accepts an explicit evaluation batch size because its dataset resizes each image to a fixed shape. Both providers record the batch size with their evaluation evidence. A custom dataset can set `supports_batched_evaluation = True` when its samples can be collated together and its evaluator accounts for every item, including a short final batch.

### Monitoring a validation score

Both `ReduceLROnPlateau` and `train.early_stopping` require a `monitor` with the exact validation dataset **name** and score key. Set `mode: min` for a declared lower-is-better score such as tabular `mae` or `rmse`, and `mode: max` for a declared higher-is-better score such as classification `top1`. Provider-generated scores can be monitored with `val.metrics` omitted or empty. A provider without a snapshot declaration can still publish scores through `metric_results` and use the explicit monitor mode. Publish a fresh mapping for each validation split, including an empty mapping when the split has no measurable score.

Each consumer observes only its selected dataset after scheduled validation on an accepted accumulation boundary. A second validation split, a rejected update, an absent or nonfinite score, and the final unscheduled validation add no observations. A plateau scheduler also requires its optimizer to have taken a step since the previous validation of that dataset. Early stopping counts consecutive finite observations that fail its absolute `min_delta` threshold and stops when the count reaches `patience`. PyTorch's plateau scheduler reduces LR when its bad-observation count exceeds its own `patience`.

### Retaining validation weights in a provider

A training provider can offer `val.save_best` if its validator can snapshot the network that produced a dataset score. Opt in on the effective `validate` action, including any architecture override that implements the same hook. The default is off. The catalog and recipe preflight expose the capability as `actions.validate.saveBest`.

```python
def snapshot_metrics(opt, settings):
    return {"mean_absolute_error": "lower"}

actions = {
    "train": TaskActionSpec(("image_pairs",)),
    "validate": TaskActionSpec(("image_pairs",), snapshot_metrics=snapshot_metrics),
}
```

The declaration callable receives the typed recipe and the decoded task settings (`None` for a provider without settings). Return a mapping from the exact scalar names this recipe can produce to `"higher"` or `"lower"`. Keep the callable free of model imports, dataset access and device work. Providers that evaluate the entries in `val.metrics` can pass `trainner_config.metrics.configured_snapshot_metrics` directly. Call `validation_snapshot_metrics(opt)` from the same module to read the resolved mapping. It returns `None` for an evaluator without snapshot support.

In the provider's validation method, finalize the dataset scores and call the existing tracker **inside the active inference context**, before restoring any EMA, schedule-free, or evaluation-specific weights:

```python
self.metric_results = dataset_scores
if self.best_weight_tracker is not None:
    self.best_weight_tracker.observe(self, inference_net, dataset_name, current_iter)
```

Pass the actual module used for inference. Its portable tensor state must belong to exactly one of `net_g`, `net_g_ema.ema_model`, `net_ae`, or `net_ae_ema.ema_model`. The tracker unwraps the module, writes on rank zero and keeps the first strictly best finite score per dataset and attempt. An absent or nonfinite metric produces no winner. A provider using `BaseModel.validation` gets an error if it reports a finite selected score without observing that score in this validation pass. The summary records the selected metric's direction for later checkpoint ranking, so the reader can rank the observations after the provider is removed or changed.
