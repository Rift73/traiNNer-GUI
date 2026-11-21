# traiNNer-GUI local API

Loopback JSON API served together with ui/ by the parent in WSL, same origin. GET /api/status => {name,version,trainerRoot}; GET /api/templates and GET /api/recipes => arrays of {path,name,task}; GET /api/recipe?path=... => {path,name,yaml,revision}; POST /api/recipes {name,yaml,revision?} => {path,name,yaml,revision}; POST /api/preflight {yaml} => {ok,issues:[{level:'error'|'warning',path,message}],summary:{name,task,architecture,scale},config?}; POST /api/render {yaml, edits:[{path:[string|number,...],value}]} => {yaml}; GET /api/catalog => {tasks:[{id,label}],architectures:[{id,task?}],losses:[{id}], schemaUrl:'/api/schema'}; GET /api/schema => full JSON Schema. POST /api/data/scan {gt,lq?,scale?} => {counts:{gt,lq,paired},issues:[{level,path,message}],samples:[{id,name,gt,lq?,gtWidth?,gtHeight?,lqWidth?,lqHeight?}],truncated}; POST /api/data/preview {path} => real image binary. GET /api/runs => array of {id,name,status,iteration?,total?,startedAt?,updatedAt?,task?}; POST /api/runs {yaml} => run object; GET /api/run?id=... => {id,name,status,iteration?,total?,startedAt?,updatedAt?,yaml?,metrics?,log?:string[],artifacts?:[{name,path,url?}]}; POST /api/runs/stop and .../resume {id} => run object. All errors JSON {error:string,issues?}. Unknown data/metrics are absent/null; no estimates invented.

When saving changes to an open draft, send the revision returned by the last read or save. A concurrent edit returns HTTP 409. The revision covers the exact UTF-8 bytes on disk, including a byte-order mark and line endings. Omit the revision to create a named draft or explicitly overwrite it.

## Windows project workspace

Create a project folder on a Windows drive and put a UTF-8 file named
`.trainner-gui-project.json` at its root. Generate a UUID once and keep it with
the project.

```json
{"version": 1, "project_id": "3e2a3e36-c480-4b83-8deb-90456f58655b"}
```

Pass the folder's absolute WSL mount path as `TRAINNER_GUI_PROJECT_DIR`. The
Windows launcher should also pass the UUID in `TRAINNER_GUI_PROJECT_ID` to pin
the expected project when a drive letter or mount point is reassigned. The
backend requires an existing DrvFS folder and marker. It checks both again for
project data access and launch. If the expected ID is omitted, the backend
takes the marker on the selected mount as its identity for that session.

```sh
export TRAINNER_GUI_PROJECT_DIR='/mnt/f/Training/My project'
export TRAINNER_GUI_PROJECT_ID='3e2a3e36-c480-4b83-8deb-90456f58655b'
./trainner-gui url
```

- `recipes/` contains saved drafts. `runs/<id>/` contains the managed run
  record, submitted recipe, and the attempt directories with launched YAML,
  console logs, and receipts.
- `runs/<id>/experiments/<name>/` contains trainer outputs, including the
  engine log, model weights, and full training states. TensorBoard writes
  under `runs/<id>/tb_logger/` when enabled.
- The Studio state directory must be on ext4 in project mode. Studio keeps
  service and attempt locks, queue state, analysis jobs and caches under
  `<state-dir>/projects/<project_id>/`. Use the same state base for a
  project's active services and trainers. The unselected CLI and its default
  state layout still work with the environment variables unset.

Desktop launch reuses only a service for the selected workspace. The desktop
and phone use the existing routes. `GET /api/status` adds `projectDir` and
`projectId` in project mode, and managed run detail adds `launchedYaml` for the
current attempt alongside the submitted `yaml`.

After replacing WSL, select the same marked folder with an ext4 state base.
Studio reads managed runs directly from the project, including their previous
attempt snapshots. Resume a stopped or failed run from its newest complete
numbered checkpoint using the usual Resume action or `trainner-gui resume ID`.
When the project path changes, Studio rebases launch input paths within the
project for the new attempt. External datasets and weights, the trainer
runtime, and any compatible recipe dependencies must be available separately.
Queue entries and other local operational state need their old ext4 state
directory.

## Runs and observations

`GET /api/run/timing?id=RUN` summarizes every recorded rate, data-wait percentage and memory observation in the selected trainer log. Use `source`, `from` and `to` to select a log and an inclusive update interval. `against` and `againstSource` add another run. The CLI equivalent is `./trainner-gui timing RUN --against OTHER --from-step 100 --to-step 1000`. A comparison reports the median of the second run's recent rate divided by the first run's rate at matching positive-rate steps, with at least two shared observations. The service reads the original logged measurements before chart decimation. Indexing advances in bounded chunks and the response marks partial sources.

The recent timer uses the trainer's batch window. The elapsed wall rate includes intervening validation and saves since the timer started. Data wait measures the share of the batch timer elapsed when the training loop receives its next batch. Worker computation that overlaps training may contribute zero wait. Allocated and reserved VRAM figures come from the PyTorch allocator. Comparing rates across runs requires checking the recipes, hardware and concurrent load.

Managed runs accept a requestId on POST /api/runs to deduplicate repeated launch submissions. Download links from run.artifacts use GET /api/run/artifact?id=...&path=.... GET /api/run/evaluations?id=... returns [{dataset,step,sampleCount,metric,better}] for completed per-image validation reports. POST /api/compare {left,right,dataset,metric,leftStep?,rightStep?} pairs those two run IDs on matching source files, dataset protocol and metric settings. Both steps may be omitted to select the most recent common step. A comparable result includes counts, summary {meanLeft,meanRight,meanDelta,minDelta,maxDelta,improved,tied,regressed}, improvements and regressions arrays [{sample,left,right,delta}]. Non-comparable data include differences or unmatched counts. Recording is opt-in through val.record_samples on SR validation.

GET /api/run/history?id=...&metric=...&source=...&budget=1000&from=...&to=... returns actual engine history for one attempt. Optional source defaults to the latest log. Omit metric to discover series. Response {sources:string[],source,series:[{key,count,first,last}],points:[[iteration,number|null]],more,indexedBytes,sourceBytes,matchedPoints?,downsampled?}. more=true requests another page of incremental indexing; polls reuse the persisted cursor. Downsampling retains bucket extrema and endpoints. These measurements come from existing trainer logs.

GET /api/queue returns {paused,activeId,entries}. POST /api/queue {yaml,recipePath?,requestId?} saves a held recipe snapshot. Only POST /api/queue/release {id} authorizes the queued training. POST /api/queue/{hold,cancel,stop} {id} controls that entry. POST /api/queue/pause {paused:true|false} controls further dispatch. GET /api/queue/entry?id=... includes YAML, GET /api/queue/group?id=... returns lineage; GET /api/run/replay?id=... inspects a prior run, POST /api/queue/replay {runId,requestId?} saves a held replay. The service picks pending entries in order as the local resource becomes available.

**Validation gallery.** GET /api/run/images?id=RUN with optional dataset, iteration, image (exact sample identifier), limit and after cursor. Returns {images,next,provenance}. Each row has kind=validation, name, image, dataset, iteration, url, path and where known width, height, lqUrl, gtUrl. Image URL returns original output bytes; append &maxSize=256 etc. for thumbnails. GT and LQ URLs return verified source previews (up to 1536), support file and LMDB origins, and return 409 if backing data changed. Older experiments only supply verifiable filename/archived LQ evidence; dataset names and GT can be absent. Select a recorded sample/dataset to query its actual exports across iterations. Query this endpoint when the person opens validation results.

**Incremental run monitoring.** Full GET /api/run?id=ID returns logCursor. GET /api/run/monitor?id=ID&cursor=... sends status, log, logReset, new logCursor and reported trainer telemetry (iterationsPerSecond, eta, vram, telemetryIteration where available). Append log unless logReset; no artifact list or recipe round-trip with each update.
**Checkpoint inspection.** POST /api/weights {path,limit?,filter?,after?} reads a real safetensors header, returning path,bytes,tensorCount,elements,dtypeElements,layoutSha256,metadata,tensors[{name,shape,dtype,elements}],next. POST /api/weights/compare {left,right,limit?,filter?,after?} identifies absent keys, changed shapes/dtypes, and sameLayout. Only structure is compared; comparing two snapshots with matching layouts does not compare learned values.

**Archive roots.** GET /api/archives lists registered external experiments folders; POST /api/archives {path} registers one, POST /api/archives/remove {id} forgets it from this checkout. The original experiment folders are read-only. Their runs have independent IDs, do not own a process and may expose logs, gallery and weights.
**Spatial registration.** POST /api/data/alignment {gt,lq,scale?,radius?} is an explicit still-image diagnostic (CPU/LLVM). It compares downsampled ground truth with observed LQ at up to nine locations. Returns actual patch regions, dx/dy in LQ pixels (positive is right/down in LQ), luma and separate blue/red differences, peak correlation, zero-offset correlation and next-peak margin. Textureless/repetitive/weak evidence can be inconclusive. These are local estimates without automatic correction/filtering; alpha is ignored and reported. No work is added to training.

**Run provenance.** GET /api/environment returns the actual WSL interpreter, installed package versions, platform, CPU capacity and disk space without initializing CUDA. GET /api/run/provenance?id=RUN returns one object per recorded attempt, including exact submitted recipe digest, a content fingerprint of the checkout's training sources and native data kernels, package versions, observed seed, real device after model construction, data-loader setup, optimizer and scheduler classes. Input weight size/mtime describe observed files; they are not content hashes. A source capture error remains visible.

**Analysis jobs.** User-requested CPU-only loss diagnostics, prediction and portable ONNX export run as isolated background jobs. POST /api/analysis {kind:"predict"|"export"|"loss-report",yaml,requestId?,weights?,input?,height?,width?,opset?,step?,batches?,parameter?,discriminator?}. Predict needs weights and a source still or folder. Export needs weights and explicit input height and width. Loss reports measure gradients of the actual configured still-image loss terms on up to 32 batches; optional weights, discriminator and parameter select the observation. GET /api/analysis lists jobs. GET /api/analysis/job?id=ID returns status and error, GET /api/analysis/log?id=ID returns the bounded log, POST /api/analysis/cancel {id} cancels the selected worker. GET /api/analysis/result?id=ID&offset=0&limit=50 returns {result,artifacts:[{path,url}],next}. Completed images, masks and ONNX graphs can be downloaded. All output is in private job folders. Existing train runs and recipes are read-only inputs to these actions.

**Measured validation summaries.** GET /api/run/validation?id=RUN&dataset=...&attempt=...&limit=50&before=ID returns {evaluations,next}, newest published first. Only id is required. Rows carry evaluation id, attempt, dataset, step, task, unit, hasClassCounts, metrics and explicit nonfinite values. Each completed managed validation captures the real evaluator output. Segmentation and classification can opt in to retaining their confusion matrix with val.record_samples. GET /api/run/classes?id=RUN&evaluation=E&limit=50&offset=0 returns class IDs, labels, support, predicted count, correct count, precision, recall, IoU, total and unit. Provide target=classId for the ranked observed predictions within that target class. Unobserved denominators return null. This preserves the distinction between image counts and annotated pixel counts. Earlier imported runs can lack these records.

The dataset scan includes multi-root folder pairs, filename templates, options for explicit manifests/LMDB, and a training-vs-validation split leakage check. Payload and evidence limits are in native/studio/data/README.md. For an LMDB sample pass the database path and key to POST /api/data/preview {path,key,maxSize?}. Classification predictions use a JSON artifact with the class ID, label and raw logits. For classification ONNX, choose width and height equal to task_options.image_size.

## Controlled studies

POST /api/studies, GET /api/studies, GET /api/study?id=… and POST /api/study/compare use the request and response shapes in STUDIES.md. Creation holds every entry. Existing queue controls release entries.

## Portable dataset preparation

POST /api/analysis with `{"kind":"pack-dataset","dataset":{"gtRoots":["..."],"lqRoots":["..."],"scale":4,"filenameTemplate":"{}"},"requestId":"..."}`. Optional `destination` is a new directory under an existing parent. Omit it to store the complete dataset under this job. GET /api/analysis/job provides progress. The result contains the source inventory, registered GT and LQ LMDB paths and a recipe fragment. POST /api/analysis/cancel cooperatively stops before publication. Classification and detection use whole-image geometry in /api/catalog.tasks[].datasetGeometry. COCO datasets use annotation_file alongside dataroot_lq.

## Pixel comparison

Gallery entries expose `versionOrigin`. A captured version pins the export at validation time (`atCapture`). For an older gallery, `onRead` pins the file observed when that page was opened. It can be used in a reproducible comparison of the current files. Historical byte identity for that output is unknown.

`POST /api/analysis` accepts `kind: "image-difference"`, `left`, `right`, an optional `reference` and optional `region: {x,y,width,height}` in the original PNG pixel coordinates. Each selection may be a local file path, a validation screenshot `{run: ID, path, version}` or `{job: ID, path, version}`, or a completed job artifact `{analysis: ID, path, version}`. The worker checks the selected file versions, reports per-channel errors in integer sample levels, and publishes a lossless absolute-difference PNG for the region. With a reference it also reports how many pixels each candidate is closer to under squared channel error. The CLI equivalent is `./trainner-gui image-difference left.png right.png --reference gt.png --region X Y WIDTH HEIGHT`.

## NeoSR import

Paired image recipes can import weighted single or multi-mode batch augmentation. Per-loss update schedules and the upsampled LQ color reference are supported for translated image losses. Aether INT8 QAT recipes use the nested `train.qat` options in the native schema. Prediction restores the checkpoint’s quantization state. ONNX exports identify QAT graphs with `precision=int8_qdq` and retain floating-point image inputs and outputs.

NeoSR dataset `color = "y"` maps to full-range BT.601 luma, `color: luma601`. One-plane still-image networks accept the converted training pair and retain a one-channel PNG during validation. Native `color: y` selects the existing studio-range convention. On-the-fly training converts both sides after RGB degradations, rounding the LQ plane to the 8-bit grid.

The importer maps Adam, AdamW, NAdam, Adan and supported schedule-free Adan/AdamW settings. Ordinary optimizer warmup follows its own step clock and composes with a supported MultiStep or Cosine schedule. Generator and discriminator can have different clocks. Unsupported settings are included in the review table.

A positive NeoSR `val.tile` selects `val.tile_mode: context` with a default halo of 16 input pixels per interior side. The network sees windows inside the image. Their centers form a disjoint output on the CPU, where holdout metrics run. The halo must be less than half the tile size. Whole-frame agreement depends on the network's receptive field and its stride or attention-window grid. Native recipes default to `blend`, which combines overlapping predictions. A zero tile size runs the network on the whole image.

For two-stage OTF recipes the importer routes the kernel settings to the data loader and noise, resampling and compression settings to training. The NeoSR resize policy uses the sampled first-stage scale factor and PyTorch's unfiltered area, bilinear or bicubic interpolation. Other recipes retain the trainer's original interpolation. The result lists differences in sampling and canvas geometry and preserves unsupported settings in the review section.

POST /api/recipes/import-neosr with `{"toml":"..."}` translates a source document into an editable native YAML draft. Returns `{yaml,sourceSha256,needsReview,review,notes}`. Unmapped entries preserve the original source keys and values in a `neosr_review` YAML section. Resolve that section and then remove it to enable training. This endpoint writes no files. The ordinary draft save handles persistence.

NeoSR imports preserve explicit `datasets.<phase>.num_worker_per_gpu` integers and `auto`. Omitted NeoSR training counts import as 4. Validation and holdout default to zero workers and keep batch size one and dataset order. Zero disables worker persistence and worker prefetch. An integer training count is per GPU under DataParallel and per rank under DDP. An integer evaluation count is per loader process. For `auto`, each process uses `max(1, affinity_cpus // world_size - 1)` workers. A single DataParallel process shares one such budget across its GPUs. On Linux, evaluation workers use spawn when the default start method is fork so validation can follow a decode in the parent. Python callers that launch these workers need a guarded main entry point.

Studies keep the assigned manual seed. Admission checks that each active loader's largest worker offset fits the NumPy seed range, using the current affinity for `auto`. The trainer rechecks with its actual worker count at launch. Fixed counts keep the existing rank and worker offsets. Distributed `auto` interleaves rank offsets so different affinity masks still give distinct seeds. Ordinary worker augmentations can depend on concurrency. The paired checkpoint replay mode keeps training crops keyed to sampler position. Use direct evaluation for replay of stochastic validation providers whose worker-local state is private.

## Measured weight drift

POST /api/analysis with `{"kind":"weight-drift","left":"baseline.safetensors","right":"candidate.safetensors"}` measures matching, same-shaped FP32, BF16 and FP16 tensors on CPU in bounded chunks. The result includes finite-pair change counts, nonfinite counts, mean and maximum absolute delta, RMS delta, relative L2 to the baseline and cosine. It lists missing, different-shaped and unsupported tensors separately. The summary returns up to 24 tensors ranked by RMS change and a downloadable JSONL row for every tensor. Values measure weight movement only. Whole-checkpoint quality requires evaluation.

## Reproducible single-process runs

The optional top-level `checkpoint_rng: true` captures Python, NumPy (legacy and project/codec generators), Torch CPU and the selected GPU's Torch random state at numbered checkpoints. Resume requires the original seed, device type, augmentation mode, sampler geometry and prepared data, with prefetch_mode null. Zero workers keep the sequential augmentation stream. For PairedImageDataset with workers, every crop and flip is keyed to its seed, epoch, rank and position in the sampler; changing the number of workers or discarding prefetched worker results preserves the sequence. ClassFolderDataset, ImageMaskDataset and CocoDetectionDataset also support position-seeded workers. With zero or more workers, their checkpoints record the ordered inputs, disk file versions, label settings and targets. Managed preflight checks the saved identity and recipe settings. The trainer compares the selected data once at startup. Earlier zero-worker image checkpoints omitted this identity and require `checkpoint_rng: false` to resume with fresh random streams. Existing version 1 identities remain compatible in their original augmentation mode, including a change between positive worker counts. Other providers use zero workers for replay. An imported full state carries its replay seed when managed YAML omits manual_seed. Select deterministic algorithms when arithmetic determinism matters. Dataset contents, code, library versions and third-party private generators must also match. With ordinary throughput settings the sampler cursor resumes the input order and fresh worker processes initialize their augmentation streams. Replayable runs wait for an accepted update before honoring a manual stop. Automatic failure recovery and epoch restart require explicitly opting out of RNG replay.

## Sequential training and task data

GET /api/curricula lists sequential plans. POST /api/curricula accepts `{name,baseYaml,phases,requestId?}`. GET /api/curriculum?id=... returns the phases, revisions and queue entries. POST /api/curriculum/retry accepts `{id,phaseId,edits?,requestId?}`. The phase schema, seed and checkpoint handoff, release conditions and alternatives are in CURRICULA.md. Every phase starts held. Existing queue controls release eligible phases.

POST /api/analysis with `{kind:"task-data",yaml,auditLimit:100,sampleLimit:25}` inspects each configured classification, segmentation, COCO detection or numeric CSV split. The default audits up to 100 selected samples per split. Set `auditLimit: null` to check all samples. Results distinguish inventory counts, observations, errors, warnings and examples. Numeric holdouts report how many inspected records share all selected input features with an inspected training record, regardless of targets, row order or unused columns. The overlap counts cover the rows actually read. The job runs on CPU and supports cooperative cancellation.

Object detection predictions use the ordinary prediction job with a detection recipe and weights. Optional scoreThreshold (0 to 1, exclusive cutoff) and topLimit (positive integer) apply per image. The downloadable predictions.json contains original-image coordinates and project category IDs. Faster R-CNN ONNX export takes one fixed-size FP32 RGB image. Its outputs are pixel-space `boxes[N,4]` in xyxy order, `scores[N]` and int64 project `category_ids[N]`. N depends on the image.

## Offline holdout evaluation

The lightweight task catalog exposes `tasks[].actions` and complete `architectureActions` overrides. Resolved recipe actions are in `POST /api/preflight` under `summary.actions`; each action names its accepted input kinds and output. Prediction, evaluation and export jobs check the declaration before launch. Detection holdouts accept `saveDetections:true` to export COCO bbox result JSON per split. The job's `cocoResults` maps dataset names to artifact paths.

`POST /api/reproduction/export` takes `{id, attemptId?, mode?, weightPath?, weightsVersion?}` and returns a JSON run-recipe bundle. `POST /api/reproduction/prepare` takes `{bundle, mappings?, sourceRoot?, name?, seed?, edits?}` and returns candidate YAML, preflight issues and compatibility evidence without submitting a job. Each mapping has a `from` prefix and a destination `to`. See `REPRODUCTION.md` for full-state replay and weights-only initialization.

Image-score observations include `evaluation`, `attempt` and `step` when available. Compare requests accept `leftEvaluation` and `rightEvaluation` to distinguish multiple attempts at the same iteration. ID 0 selects a legacy per-step ledger. A historical observation retains its scores when later PNGs replace its exports. Image URLs are bound to recorded file versions.

An analysis job `{kind:"evaluate",yaml,weights,split?,device?:"cpu"|"cuda",saveImages?:false}` runs the normal trainer evaluator on a selected validation/test section, or all such sections. It loads the indicated checkpoint strictly, uses FP32 on the selected device with compilation off, and records per-dataset counts, measured metrics, nonfinite observations and the effective recipe. Restoration and reconstruction support saved PNGs. Classification, segmentation and detection report task-specific metrics. The output directory contains the recorded SQLite summaries and applicable per-sample scores and images. Unpaired autoencoder evaluation encodes the observed input and scores reconstruction against that same image. CPU is the default. An explicit CUDA evaluation reserves the same checkout-wide execution slot as the managed queue and checks for an ordinary WSL trainer before launch. Its CUDA worker inherits the service’s device visibility and records the selected device. The CLI accepts `evaluate RECIPE WEIGHTS --device cuda`. Device and inference precision accompany the result for comparisons.

Run history sources also include TensorBoard event files in the experiment itself and in the matching sibling tb_logger/<experiment name>. The source key distinguishes each writer. Requests page a CRC-checked native TFRecord index and report sourceFormat=trainer-log or tensorboard. Scalar summaries only.

### Detection confidence operating points

For a detection holdout, `saveDetectionOperatingPoints: true` requests a per-split JSONL curve and a summary. It can accompany the COCO box export. The job result maps each dataset name to its JSONL artifact path in `detectionOperatingPoints` and to its summary in `detectionOperatingSummary`. The worker context `traiNNer.tasks.object_detection.operating_points.record(directory)` exposes the same splits in `outputs` and `summaries`. Each summary contains `iouThreshold`, `scoreComparison`, `detectorLimits`, `categories` and `global`.

Each JSONL row represents one distinct scored confidence and includes all scored detections at or above that `threshold`. Rows have `scope`, `threshold`, `support` (ordinary ground-truth boxes), `tp`, `fp`, `fn`, `precision`, `recall` and `f1`. Category rows also carry the 1-based model `classId`, original `categoryId` and `name`. The global rows pool counts across classes, including false positives in categories with zero support. Each summary group includes final counts, `scoredDetections`, `pointCount` and the measured `bestF1` point. Equal F1 selects the highest threshold. With zero support, recall, F1 and the best point are null. With targets but no scored detections, all targets are missed and the group has zero observed thresholds.

Matching uses descending confidence within each image and category, preserving prediction order on equal scores. A prediction takes the highest-IoU available ordinary box at IoU ≥ 0.50, breaking an IoU tie by target order. If no ordinary match remains, IoU ≥ 0.50 with any same-category box flagged `ignore` or `iscrowd` excludes that prediction. An ignored box can exclude multiple predictions. Every remaining unmatched prediction counts as a false positive. AP50 keeps its stable per-detection ordering and precision envelope. The operating points group equal confidences and report ordinary precision, recall and F1.

The Faster R-CNN head first applies its strict `score > scoreThresholdExclusive` filter, removes small boxes, performs per-class NMS at `nmsIou`, and keeps at most `maxDetectionsPerImage` across foreground classes. The report records these effective head settings under `detectorLimits`. Unknown limits are null. The curve uses only emitted detections. A best threshold describes this holdout, checkpoint and detector configuration. Evaluate it on independent data before using it as a deployment threshold.

## Split leakage and checkpoint selection

Set `val.save_best` to the name of one enabled validation metric to keep its winning inference weights. Each dataset and training attempt has its own safetensors file. A strictly better finite score replaces the file and a tie keeps the earlier step. Selection takes place during validation, so the saved artifact can come from a step between numbered checkpoints. The metric, direction, value, step, attempt and inference role are embedded in its metadata. Checkpoint selection reports `retainedValidationWeights` when the artifact matches the recorded scalar. Normal state files remain the entry point for full optimizer resume.

POST /api/analysis with `{kind:"split-leakage",yaml,groupLimit?:100}` checks the recipe's selected training and validation inputs for identical encoded content. The group limit bounds returned examples. Coverage, unreadable or changing input and match counts accompany the results. Run this explicitly for large datasets.

GET /api/run/checkpoint-selection?id=... lists recorded validation series. Adding dataset and metric ranks recorded observations and identifies currently available weights when their association is supported. The optional direction is higher or lower for historical metrics that lack a recorded direction. A missing or ambiguous checkpoint stays distinct from the observed best score.

## Comparing measured holdout results

POST /api/compare accepts either a managed run ID or `{"job":"completed-evaluation-id"}` for each of left and right, plus dataset, metric and optional leftStep/rightStep. Two training runs default to their latest common measured iteration. A standalone evaluation has the single observation -1; mixing it with a training run defaults to that run's latest measurement for the selected series. The response names both iterations, records inference precision/device/tile settings when available, and separates changed inputs, nonfinite values and missing observations from matching scores. Individual improvement/regression rows may include leftImageUrl and rightImageUrl for recorded PNGs. Input identity is based on path and file metadata. GET /api/analysis/evaluations and /api/analysis/images use the same query/response shapes as their /api/run counterparts on a completed evaluation job. The CLI accepts job:ID for a standalone observation.

## Restoration prediction contracts

For a windowed restoration model, CPU holdout accepts `saveTemporalErrors: true` (`evaluate --save-temporal-errors`). Select a PairedVideoDataset with numbered frame names and aligned references. For consecutive frame pairs it reports the mean absolute difference between the change in the prediction and the change in the reference, in RGB normalized to 0–1 after 8-bit rounding. Zero means the predicted change matched the reference at every scored pixel. A constant spatial error across frames also scores zero, so use spatial metrics alongside it. Each split has a JSONL of genuine transitions plus a pixel-weighted summary and its largest-error pair. The report counts skipped transitions across scene prefixes, missing numbers, discontinuous input windows or size changes. It uses the same forward pass as the other holdout outputs.

Portable ONNX export accepts `dynamicSpatial: true` for the Compact and Aether restoration families. The preflight response advertises support in `summary.actions.export.dynamicSpatial`. Height and width supply the tracing example. The exported graph accepts other positive spatial dimensions at batch size one and records the integer output scale. Aether QAT checkpoints preserve their INT8 quantize/dequantize operators and floating-point RGB interface. The headless predictor exposes the same option as `--dynamic-spatial`.

CPU validation reuses learned metric networks within the validation dataset. Set `val.cache_metric_models_on_device: true` to retain them across validation images on an accelerator. References are released when the validation pass ends. The default accelerator path loads one learned metric at a time.

The standard predict job also reconstructs a single image or folder with the trained autoencoder. Its output dimensions equal the input; dimensions must divide evenly by the autoencoder scale. Fixed-shape autoencoder ONNX export has the same RGB-in/RGB-out contract. Temporal generators accept a folder of PNG/JPEG frames and an optional positive `clipSize`. Fixed-window architectures supply their own size. Variable-window models take the unambiguous validation/test window from the recipe, then the training clip size if no evaluation size is set. Frame folders use natural numeric frame order within each complete scene prefix. Only centers with a full observed window are emitted; different scenes/roots are kept distinct. The result reports inputFrames, centerFrames, boundaryFramesOmitted, clipSize and eligible scene count. Use full RGB frames for temporal deployment. Fixed-window RGB ONNX export includes TemporalSPAN, TemporalSPANv2 and `rsr_compact`. Both SPAN families use the checkpoint architecture’s frame count. RSR requires an explicit clip size and reports `temporal_read` (`block` or `aligned`). Aligned RSR requires a checkpoint containing loaded NeuFlow v2 weights and opset 16 or later. For `clipSize` greater than one and `min(height,width) >= 512`, set `opset` to 18–20 to enable the native antialiased flow resize. The graph computes flow for adjacent frames. Each graph takes one same-scene RGB window and returns its centre frame. RDUAL supports eager sequence prediction.

## What the model received

Set `logger.training_sample_steps: [1, 500]` to capture the first sample fed for each requested candidate update, on rank zero. GET /api/run/training-samples?id=... returns `{samples,next}` with optional `limit` and `after`. Each capture has the epoch, 1-based batch in epoch, candidate update, zero-based sample index, task and tensor shape/dtype. `rawUrl` serves the tensors in safetensors format. RGB previews clip to [0,1] and round to 8 bits; temporal previews show the labeled center, while raw tensors retain the installed window. Masks keep their class IDs. Classification reports the actual target class, and detection includes the boxes and labels retained for training. This is post-feed and pre-optimizer: the recorded update may subsequently be rejected. File provenance is omitted for pooled/mixed examples. Capture is off by default; an explicit capture synchronizes and writes samples to disk.

Pooled scores for classification, segmentation and detection also use /api/compare when both observations have `val.record_samples: true`. Select by dataset and metric, optionally disambiguating repeated training steps with leftEvaluation and rightEvaluation IDs from /api/run/evaluations or /api/analysis/evaluations. A compatible result reports `aggregation:"pooledDataset"` and `pooled:{left,right,delta}`. It compares the evaluator's full-population results only after checking class definitions, metric protocol and every recorded input/target identity, including detection tie order. /api/analysis/validation and /api/analysis/classes browse evaluation jobs in the same form as their /api/run counterparts.

Full-state import preflight includes summary.resume with the selected numbered step, model files, optimizer and schedule counts, cursor, replay availability and file version, when present. Existing recipes without path.resume_state skip this inspection. Rejection reasons are normal preflight issues with recipe paths; a weights-only fine-tune uses a pretrain_network_* weight and clears resume_state.
Study comparisons include seedSummary.cohorts, aggregating one observation per matched training seed only at identical iterations, metric protocol and scored inputs. Each cohort includes the baseline, treatment and paired delta mean/median/min/max and sampleSd (null with a single seed). Partial populations and unavailable pairs are listed under excluded.

A CPU analysis job with kind pair-registration accepts yaml, split (default train), sampleLimit (1–64), radius (2–12 LQ pixels) and seed. It uses the actual PairedImageDataset selection, including manifest order, filename templates and LMDB roots and keys. Results report population, sampled, counts and pairs, with distinct luminance and two chroma displacement estimates and the evidence for each. The exposure is diagnostic; source compression and intentional channel shifts can also produce a displacement. Single-pair /api/data/alignment now accepts optional gtKey/lqKey alongside the database paths.

## Local registries, paired patches and checkpoint files

Use `./trainner-gui network-parameters recipe.yml` or submit an analysis of kind `network-parameters` with `yaml` to see the architecture's named parameters, buffers, shapes and sizes before choosing training data. The worker constructs the network on PyTorch's meta device. The result distinguishes all parameters, default trainable parameters and those matching the generator optimizer regular expressions. The JSONL lists every named tensor. The counts describe the bare architecture at construction. Runtime wrappers, QAT and optimizer state can change the training footprint.

The network parameter job accepts `inspectTransfer: true` (CLI `network-parameters --inspect-transfer`) for a fresh training recipe with `path.transfer_g` and a selected generator checkpoint. This runs the trainer's tensor transfer on a temporary CPU network before optimizer construction or QAT preparation. The result lists counts of selected elements copied or freshly initialized; `freshExcludedElements` counts initially trainable elements left at constructor values outside the fine-tuning filter. The full `transfer-receipt.json` lists accepted and skipped keys. A same-shaped head is eligible for copying by name. Exclude its keys explicitly when changing the meaning of class rows. The default parameter inventory uses only the metadata device.

Classification, segmentation and detection honor `train.optim_include_params_g` and `train.optim_exclude_params_g` on the bare network names. Exclusion wins. Unselected parameters stay frozen during the update, while modules in training mode continue their normal buffer updates (including BatchNorm running statistics). Classification and segmentation also support `use_amp: true` with `amp_bf16: true`. Their forwards use BF16 autocast and their cross-entropy reductions use FP32. Holdout comparison displays recorded inference settings separately from the scored population and metric definition.

`POST /api/catalog/refresh` loads this checkout’s installed factories in an isolated CPU process and atomically replaces its local schema snapshot. The equivalent CLI is `./trainner-gui catalog --refresh`. Catalog responses include the optimizer and image-metric constructor fields, parameter defaults, registry origins and any unsupported signatures. The local service and recipe preflight share the snapshot. A changed packaged schema supersedes a snapshot built against an older version.

`POST /api/analysis` accepts these CPU jobs. All paths in a completed result refer to that job’s retained output.

- `split-dataset` takes `yaml`, `validationFraction`, optional integer `seed` and `groupBy` of `pair`, `parent` or `scene`. It writes train/validation membership manifests and a recipe that keeps the other evaluation datasets. The result includes the dataset settings, `recipePath` and `recipeDatasetKeys`. Group related images or frames before extracting patches.
- `extract-patches` takes `yaml`, `split`, `cropSize` and optional `stride`. Sizes use LQ pixels. Each GT rectangle uses the same coordinates multiplied by the recipe scale. The output includes paired PNG folders, an exact-coordinate JSONL, trainer membership and dataset settings. The native reader retains supported 8- and 16-bit samples. Its format coverage is documented in [paired patch extraction](../native/studio/paired_patches/README.md).
- `convert-weights` takes a local `.pth` in `weights` and an optional `payload`. When the file has several matching tensor dictionaries, select one explicitly. The job reads it on CPU using restricted PyTorch loading and produces a separate safetensors file with the original tensor names, dtypes and JSON network metadata.
- Classification holdout with `saveClassifications: true` writes a JSONL with the observed target and ordered predicted probabilities for each input. Other optional export fields are listed by `summary.actions.holdout.optionalExports`.

`GET /api/run/checkpoint-frontier` takes `id`, `dataset`, `x` and `y` for two distinct metrics. Optional directions are `xDirection` and `yDirection`; use `higher` or `lower`. Both values must come from the same evaluation. Each cohort uses a single attempt and evaluator configuration. A populated `populationId` includes recorded sample identities and the evaluation protocol. With aggregate evidence alone, the response reports `populationVerified: false`. Setting `availableOnly=true` derives the frontier among weights verified by a save receipt. `frontierCount` covers the whole cohort; `limit` controls the number of points returned.

## Checkpoint disk capacity

`GET /api/run/checkpoint-storage?id=…` and `checkpoint-storage <id>` return the same read-only report for a managed, checkout CLI or registered archive run. All sizes are bytes. Unknown estimates are null.

| Field | Meaning |
| --- | --- |
| `policy` | Recipe source, target update, numbered save interval, format, regular-set retention count and policy. Zero disables native pruning. NeoSR TOML reports its separate state and per-network export counts when declared. |
| `observed` | Current logical file sizes in `models`, `models/resume_models` and `training_states`, partitioned into numbered, latest, other and visible temporary files. Hard links count per directory entry. `complete` describes inventory coverage. |
| `observed.recentCompleteSets` | Up to three measured sets among the 16 most recently modified numbered states. Each includes step, cadence match, bytes and component paths, roles and sizes. Container metadata, expected model roles, save order and stable file versions establish the sample. An optional per-weight `validationSave` matches its embedded identity to an existing receipt among the latest 128 evaluations. A receipt attests only that weight. |
| `projection` | `sampleSteps` identifies the evidence. `numberedSaveBytes` uses the largest sampled set. `remainingNumberedWriteBytes` counts cumulative writes from the reported `fromStep` to target, including a managed off-cadence target save. `futureRetainedNumberedBytes` estimates the portion of those future saves still present at target. `peakAdditionalNumberedBytes` includes the new set that lands before eligible regular sets are pruned. Both give zero credit for removing existing files. `regularRetentionWindowBytes` sizes a filled regular window. |
| `filesystem` | `volumes` lists each checkpoint filesystem and bytes available to the process. On a single measured filesystem, `availableBytes`, `wouldFitOneNewSet` and `wouldFitProjectedNumberedPeak` allow comparison with the estimate. Separate volumes retain their individual readings. |
| `limitations` | Missing or ambiguous evidence, excluded candidates and partial measurements. The response also includes the projection assumptions. |

`atomicTemporaryFileBytes` estimates the largest individual write. `newNumberedSaveHeadroomBytes` budgets a complete new set because each component's temporary file becomes its final file by rename. These two figures overlap. Forecasts assume similar future sizes, fresh numbered paths, successful eligible pruning and reaching the target. Manual stops, protected checkpoints, latest-weight replacement, backups, validation outputs, other filesystem activity and host VHD capacity can change actual disk needs. A completed managed attempt has zero remaining scheduled writes.

## Loss attribution on training data

The loss report accepts the trainer's CPU-capable on-the-fly degradation recipes. Each batch reports the norm of the difference between the true total gradient and the sum of independently measured term gradients (`independentTermResidualNorm`). A restoring clamp can produce a nonzero difference when losses jointly change the backward gate. For an active GAN loss, `adversarialBalance` reports the weighted GAN norm, the norm of the remaining objectives measured together, their ratio, and their cosine at the selected output or parameter. Undefined ratios and cosines are null. These observations run in the separate analysis job with zero optimizer updates.

## Splitting paired video

With `PairedVideoDataset`, the split-dataset action defaults to `groupBy: "scene"`. It publishes scene manifests and a recipe for windowed image-restoration training. Train and validation counts refer to complete windows. The sources remain in their frame directories.

## Offline checkpoint averaging

The chooser reads `GET /api/run/average-candidates?id=RUN&role=net_g&limit=100`. It lists numbered safetensors from the role's actual save directory with their versions and sizes. When `nextBefore` has a value, use it as the next page's `before`.

The default requires identical BatchNorm buffers. To average weights across changing BatchNorm state, choose `batchNorm: "fromLatest"` or use the CLI option `--batch-norm fromLatest`. This copies each complete running-mean, variance and counter group from the highest selected step. The derivation records that step and the number of copied tensors. Other fixed buffers still require equality. Evaluate the derived weights on the intended holdout to measure the effect.

Submit `POST /api/analysis` with `{"kind":"average-checkpoints","run":"<managed-run-id>","role":"net_g","checkpoints":[{"step":1},{"step":2}]}`. Each step may include the opaque `version` of the selected file. Admission pins the bytes' local file identity before queuing. The CLI equivalent is `./trainner-gui average-checkpoints RUN net_g 1 2`. The result exposes the derived weights as `averaged.safetensors` for an ordinary CPU evaluation or prediction. It represents model weights only and cannot resume the source optimizer.

`studio.checkpoint_average.average_checkpoints(inputs, destination, *, run_id, role, task, cancel=None, progress=None)` writes one weights-only safetensors artifact. Admission resolves at least two distinct numbered checkpoints from the same managed run, experiment and network role. Supply each as `{"path": "/absolute/.../models/net_g_8.safetensors", "version": "<artifact_version>"}` and pass the canonical absolute path of a new file in the analysis job's output directory. The worker checks the pins on open and again before atomic, no-overwrite publication. `cancel()` can raise the job's cancellation exception. Progress events supply `stage: "averaging"`, completed and total tensors, and input payload bytes read.

The result contains `schemaVersion: 1`, `method: "uniform"`, `runId`, `role`, `task`, `inputs`, `output`, `version`, `tensorCount`, `tensorBytes`, `averagedTensors` and `identicalBufferTensors`. Register `[destination]` as the artifact list. The file preserves the native model metadata and embeds the derivation under `trainner_checkpoint_average`. Each input records its experiment-relative path, step, local file version and optional native `saveId`. Local versions identify observed files on the source filesystem. The worker removes the optional top-level `trainner_save_id` from each raw or EMA input before comparing model metadata. The average has no trainer save association of its own. A contributor’s ID is recorded only as `inputs[].saveId` in the result and embedded derivation. Older files without that key still contribute their pinned file version.

The worker accumulates finite F16, BF16 and F32 values on CPU in bounded blocks using FP64, then casts once to each tensor's original dtype. Names, shapes, dtypes and native model metadata must match. Classification, segmentation and detection accept a single raw or EMA generator role per average and require ordered class names. When EMA is enabled, validation uses the averaged network, numbered EMA saves live under `models/`, and raw optimizer companions under `models/resume_models/`. A validation observation records the save identity only when the evaluated tensor state can be established. Detection also requires ordered category IDs. Integer/boolean and recognized fixed buffers must be byte-identical, except for BatchNorm groups explicitly selected with `fromLatest`. Quantized, observer, packed and unsupported tensor formats are refused. Use the resulting G or AE weights with the matching recipe for inference, fresh evaluation or initialization. A discriminator average can initialize the discriminator. Full-state continuation requires the original optimizer, scheduler and training state.
### Validation snapshot choices and worker replay

`POST /api/preflight` returns the recipe's metric names and ranking directions under `summary.snapshotMetrics`. `bestMetric` is the selected `val.save_best` name. `bestValidationDatasets` lists the validation sets eligible to produce that score when retention is enabled. `POST /api/curriculum/phase-preview` accepts `{"baseYaml":"…","edits":[]}` and returns the same three fields for the edited recipe, without scheduling a phase.

A stopped or failed managed run can advertise `resumeWorkerOverride: true` and its effective `trainWorkers` in `GET /api/run`. Submit a positive integer `trainWorkers` alongside its `id` to `POST /api/runs/resume`, or use `./trainner-gui resume ID --train-workers N`. This requires a complete numbered checkpoint with saved RNG state, sampler geometry and the matching position-seeded worker policy. The resumed attempt records the new worker count in its private options. Later resumes inherit it. Curriculum retries retain their own phase settings.

### Numeric records

`tabular_regression` advertises the input kind `records`. Prediction accepts a CSV file and returns ordered numeric arrays in `predictions.json` with `features`, `targets` and `count`. Its ONNX export needs no spatial dimensions. The `task-data` inspector accepts this recipe and returns each split's `source`, `counts.recordsInspected`, per-column role and min/max, and sampled rows with named values. A bounded inspection reads a prefix. An unknown full count is null. Pass `auditLimit: null` to inspect the full file using bounded working memory.

An image holdout reports `outputPixelFormat` when its observations agree on a grayscale (`y`) or RGB output. This is measured after the model's pixel adapters and applies to both scoring and saved images.

