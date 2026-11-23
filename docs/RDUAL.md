# DUAL execution and RDUAL clip training

## DUAL options and checkpoint compatibility

DUAL presets retain their geometry, EDBB convolutions and input-routed dynamic
expert convolutions. Light uses C128, 16 layers and PixelUnshuffle2.
The current factory options include:

```yaml
network_g:
  type: dual_light
  scale: 4
  use_checkpoint: true
  context_na: true
  local_na: true
  retrieval_na: true
```

The three NA switches select attention semantics, not merely faster backends:
context uses per-query 32/64 neighborhoods, local uses per-query 16 neighborhoods,
and retrieval reads up to 48 x 48 coarse keys per fine query.
Setting a switch to false retains its corresponding fixed-partition path.
These choices do not change state-dict keys, so preserve the original choices
when loading an older checkpoint. The current defaults are all true; a strict
weight load alone cannot detect a changed attention configuration.

`g_attention_impl`, `compile_g_attention_impl` and DUAL's former
`attn_type` are no longer accepted. Backend selection is automatic.
`context_overlap` remains a deprecated alias for `context_na`; it is not
overlap-add output blending. Preset-fixed geometry should not also be supplied
as a keyword (for example, `embed_dim` on `dual_light`).

CUDA BF16/FP16 training uses bundled Triton attention where the device, dtype
and geometry support it. Other paths use PyTorch SDPA. Selective checkpointing
saves expensive attention results while recomputing surrounding intermediates,
controlled by the single `use_checkpoint` switch. Compile preparation receives
the actual input shape and dtype. Evaluation/export disable checkpointing;
`prepare_for_export()` folds static EDBB factors on an eval copy, never on the
training instance. Dynamic expert routing remains input-dependent.
These paths do not promise bitwise identity across precision modes or backends.

DUAL and RDUAL also accept `routing_na` (default `false`) for per-position
expert routing over a neighborhood of up to 96 × 96 trunk positions. DUAL
accepts `regional_na` (default `false`) for per-position regional channel
attention. Save these switches in the training recipe alongside the checkpoint.
The regional path uses the four-grid blend during export. Both trainer exporters failed on the complete
DUAL Light model in the live tests, including with both switches disabled.

## RDUAL

`network_g.type: rdual` selects the dedicated clip trainer. RDUAL embeds the
DUAL-Light PU2 spatial trunk plus bidirectional recurrent feature transport.
It runs a reverse sweep and a forward sweep with flow-guided temporal attention.
Input/output layout is `B,T,3,H,W` / `B,T,3,scale*H,scale*W`; native scales
are 1, 2 and 4. Shot IDs stop recurrence across cuts. The frozen estimator is
NeuFlow v2, not a separately trained flow module.

A DUAL-Light checkpoint can warm-start the spatial trunk. The temporal output
projection starts at zero and training starts with a new optimizer.
Native RDUAL checkpoints include `flow.*` and `flow_loaded`. Starting from
scratch or from a checkpoint without trained flow uses the external file at
`path.neuflow_weights` if set. The file must exist and can be `.safetensors`
or `.pth`. With the option unset, the trainer first looks for
`experiments/pretrained_models/neuflow_v2.safetensors` in its working directory.
It then uses the official NeuFlow v2 mixed checkpoint in
`~/.cache/trainner/neuflow/neuflow_mixed.pth`, downloading and verifying it on
first use during training or sequence prediction. Recipe preflight checks the
inputs offline. The NeuFlow license is retained in
`traiNNer/archs/NEUFLOW_LICENSE.txt`.

Relevant fragments for a complete new clip-training recipe:

```yaml
high_order_degradation: true
input_pixel_format: rgb
output_pixel_format: rgb
clip_keep: none
network_g:
  type: rdual
  scale: 2
  use_checkpoint: true
  temporal_backend: auto
  local_na: true
  retrieval_na: true
datasets:
  train:
    type: videoclipdataset
    dataroot_gt: [/path/to/indexed/videos]
    lq_size: 128
    clip_lengths: [8, 12, 16, 20]
    downscale_algorithm: hermite
    descale_size: [1488, 837]
    descale_kernel: bilinear
    descale_border: zero
    descale_linear_light: null
    hr_size: [960, 540]
    hr_linear_light: gamma24
    descale_error_threshold: 0.035
    motion_weight_floor: 0.25
    cut_clip_prob: 0.05
    use_time_reverse: true
train:
  use_moa: false
val:
  val_enabled: false
```

These are source-specific example sizes, not universal video defaults to infer
from unknown material. Supply the usual name, scale, paths, batch, optimizer,
scheduler and loss settings as well. The implementation currently supports
single-device, RGB OTF clip training. Ordinary image validation, MoA,
LDL, HR high-frequency noise and generator distillation are rejected for clips.
VOSR as an extra loss is supported; full VOSR HR-target replacement is not.

`clip_keep: none` recomputes per-frame graphs and is the supported mode when
using CUDA graphs. `reverse` retains reverse-sweep graphs, and `both` retains
both sweeps' graphs, trading memory for less recomputation. The recurrent trainer
compiles per-frame step functions, not the whole clip. The architecture's
`use_checkpoint` controls selective layer checkpointing separately.
Static EDBB factors are composed once per clip and reused within its lifetime.

## Video preparation and degradation

In the workbench, choose the bundled **RDUAL OTF fidelity** recipe. In its training
dataset, choose a folder of episodes or select specific MKV, MP4, M2TS or MOV
files when the folder contains alternate encodes. Windows drive paths work
through WSL. Use the video files directly. Set `descale_size` and `descale_kernel`
to match the source, then use the dataset's video indexing controls. Compatible
indexes are reused. Allow write access to the adjacent `index` directory.

The CLI uses the same index format. From the checkout, run a command such as

```bash
python -m scripts.data_preparation.index_video_clips /path/to/episode.mkv --width 1488 --height 837 --kernel bilinear --border zero
```

The script writes `index/<video-stem>.json` beside each video. The dataset reads
timestamps, cuts, motion and sampled re-upscale error from that index.
Its conversion currently expects 8-bit `yuv420p`, limited-range BT.709,
left-sited chroma. It is not a metadata-aware general-purpose decoder for all
colour spaces or bit depths. Index and dataset geometry/kernel choices must
match the actual source upscale.

Shots exceeding the sampled error threshold are excluded. Each clip shares its
crop, flips, rotation and time direction; one supported length is drawn per batch.
For the accelerated clip path, all videos in a dataset must have the same even
frame dimensions. Workers send cropped YUV420 planes and geometry. The training
device reconstructs each HR crop. A clip shares its blur and noise-strength
settings, with independent noise samples for each frame. Video codecs operate
temporally with optional lead-in frames. Eligible training paths prepare
degradations ahead of consumption, so random draws can interleave across batches.

## WSR and RSR

The image WSR presets are in `options/train/WSR/`. The recurrent
`rsr_compact` video super-resolution training examples are in `options/train/RSR/`
and use the clip loader. `temporal_read: block` is the model default.
`aligned` uses frozen NeuFlow v2 weights and is experimental. When it needs
external flow weights in training or sequence prediction, it uses the same
search order as RDUAL. Configure the dataset paths and starting checkpoints for
the intended training run.

## Verification boundary

CPU tests cover spatial warm starts, recurrent dependency, cut handling,
checkpoint/recompute gradients, export folding, synthetic video indexing,
descaling and temporal codec behavior. Tests needing real pretrained NeuFlow,
CUDA or TensorRT assets are separate and skipped when those assets are absent.
No CPU-only port check establishes GPU training speed, inference latency,
VRAM usage, or learned image quality.

Implementation: `traiNNer/archs/dual_arch.py`,
`traiNNer/archs/rdual_arch.py`, `traiNNer/models/rdual_model.py`,
`traiNNer/data/video_clip_dataset.py`, `traiNNer/data/video_index.py`,
and `traiNNer/data/compress_video_batch.py`.
