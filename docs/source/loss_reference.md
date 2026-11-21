# Loss reference

## Loss scheduling

Add `start_iter`, `ramp_iters`, or `end_iter` to a loss entry to schedule it.
The fields also work in a legacy `*_opt` loss entry when `train.losses` is
omitted. An ordinary loss can have a unique `name` to distinguish several
instances of the same type. GAN, LDL, and VOSR retain their designated slots.

```yaml
train:
  total_iter: 12000
  losses:
    - type: l1loss
      name: reconstruction
      loss_weight: 1.0
    - type: l1loss
      name: lq_consistency
      loss_weight: -0.2
      start_iter: 2000
      ramp_iters: 500
      end_iter: 8000
```

The first active update is `start_iter + 1`. With a ramp, that update uses
`1 / ramp_iters` of the configured weight. The multiplier reaches 1 at
`start_iter + ramp_iters`. A zero or omitted ramp applies full weight from
the first active update. `end_iter` is inclusive. Omit it or set it to `null`
for an open-ended window. Omitted `start_iter` defaults to zero.

All microbatches in one accumulated update use the same multiplier. Rejected
attempts reuse the candidate update number, and resume continues from the
checkpoint's completed update count. Scheduled SR uses synchronous finite
checks for exact acceptance accounting.

In SR and recurrent training a negative ordinary loss weight still targets
the upscaled LQ image, using its magnitude for the objective. AE uses positive
ordinary reconstruction weights. LDL can be scheduled in SR with generator
EMA. Recurrent clip training retains its existing LDL restriction.

A GAN window controls when the discriminator runs. The ramp changes the
generator's GAN weight. Active discriminator steps keep their existing loss
weight and adaptive control. Outside the window, D keeps its scheduler and
adaptive EMA state. D learning-rate warmup begins at the first active GAN
update. Scheduled GAN training uses a positive GAN weight and one configuration
format. Remove a legacy GAN entry when using the loss list so checkpoint
discovery selects the same discriminator as training. VOSR contributes only
while GAN is active. Its own ramp multiplies its weight relative to the raw
GAN signal.

Every planned update must have an active configured objective. The validator
rejects a schedule gap at its first uncovered update. VOSR and auxiliary
objectives cannot cover a gap by themselves. Scheduled training requires a
single training process because candidate rejection is local to each rank.

## vosrloss

VOSR 2 restores the degraded LR with external fixed-window TensorRT engines.
The raw distance is balanced against the raw GAN term using detached running
means. `loss_weight` controls relative loss magnitude, not equal gradient norm.
Requires `gan_type: vanilla`, `lsgan` or `wgan_softplus` and HR patches at most
512 x 512. It does not replace the ordinary HR target.

```yaml
type: vosrloss
loss_weight: 0.1
engine_dir: /path/to/vosr2-512-engines
plugin: /path/to/vosr2_qkv.so
criterion: charbonnier
lanes: 2
still_frames: 2
```

Other criteria are `l1` and `perceptual`; the latter accepts
`perceptual_options`. Clip-only `still_frames` controls nearby still-region
target reuse. See [VOSR teacher integration](../VOSR_loss.md) for dependencies,
the separate HR-target mode, and verification boundaries.

This page lists all available parameters for each loss function in traiNNer-redux. While the default configs use recommended default values and shouldn't need to be modified by most users, advanced users may wish to inspect or modify loss function params to suit their specific use case.
## adistsloss


```yaml
type: adistsloss
window_size: 21
resize_input: false
loss_weight: 1.0
```
## aesoploss


```yaml
type: aesoploss
criterion: charbonnier
loss_weight: 1.0
```
## averageloss


```yaml
type: averageloss
criterion: l1
loss_weight: 1.0
```
## bcewithlogitsdiceloss


```yaml
type: bcewithlogitsdiceloss
weight_bce: 1.0
weight_dice: 1.0
eps: 1e-06
loss_weight: 1.0
```
## bicubicloss


```yaml
type: bicubicloss
criterion: l1
loss_weight: 1.0
```
## charbonnierloss


```yaml
type: charbonnierloss
reduction: mean
eps: 1e-12
loss_weight: 1.0
```
## colorloss


```yaml
type: colorloss
criterion: l1
loss_weight: 1.0
```
## contextualloss


```yaml
type: contextualloss
layer_weights: ~
crop_quarter: false
max_1d_size: 100
distance_type: cosine
b: 1.0
band_width: 0.5
use_vgg: true
net: vgg19
calc_type: regular
z_norm: false
loss_weight: 1.0
```
## cosimloss


```yaml
type: cosimloss
cosim_lambda: 5
loss_weight: 1.0
```
## distsloss


```yaml
type: distsloss
as_loss: true
load_weights: true
use_input_norm: true
clip_min: 0
loss_weight: 1.0
```
## ffloss


```yaml
type: ffloss
alpha: 1.0
patch_factor: 1
ave_spectrum: true
log_matrix: false
batch_matrix: false
loss_weight: 1.0
```
## fftloss


```yaml
type: fftloss
loss_weight: 1.0
reduction: mean
loss_weight: 1.0
```
## fliploss


```yaml
type: fliploss
loss_weight: 1.0
pixels_per_degree: 67.02064327658226
loss_weight: 1.0
```
## ganloss


```yaml
type: ganloss
gan_type: vanilla
real_label_val: 1.0
fake_label_val: 0.0
loss_weight: 1.0
```
## gradientvarianceloss


```yaml
type: gradientvarianceloss
patch_size: 16
criterion: charbonnier
loss_weight: 1.0
```
## hsluvloss


```yaml
type: hsluvloss
hue_weight: 0.3333333333333333
saturation_weight: 0.3333333333333333
lightness_weight: 0.3333333333333333
criterion: l1
downscale_factor: 1
loss_weight: 1.0
```
## l1loss


```yaml
type: l1loss
reduction: mean
loss_weight: 1.0
```
## ldlloss


```yaml
type: ldlloss
criterion: l1
loss_weight: 1.0
```
## linedistillerloss


```yaml
type: linedistillerloss
criterion: l1
debug: false
loss_weight: 1.0
```
## lumaloss


```yaml
type: lumaloss
criterion: l1
loss_weight: 1.0
```
## mseloss


```yaml
type: mseloss
reduction: mean
loss_weight: 1.0
```
## mssimloss


```yaml
type: mssimloss
channels: 3
downsample: false
test_y_channel: true
is_prod: true
color_space: yiq
include_luminance: false
loss_weight: 1.0
```
## msssiml1loss


```yaml
type: msssiml1loss
gaussian_sigmas: ~
data_range: 1.0
k: [0.01, 0.03]
alpha: 0.1
cuda_dev: 0
loss_weight: 1.0
```
## multiscaleganloss


```yaml
type: multiscaleganloss
gan_type: vanilla
real_label_val: 1.0
fake_label_val: 0.0
loss_weight: 1.0
```
## nccloss


```yaml
type: nccloss
loss_weight: 1.0
```
## psnrloss


```yaml
type: psnrloss
reduction: mean
to_y: false
loss_weight: 1.0
```
## perceptualanimeloss


```yaml
type: perceptualanimeloss
layer_weights: ~
criterion: l1
loss_weight: 1.0
```
## perceptualfp16loss


```yaml
type: perceptualfp16loss
layer_weights: ~
w_lambda: 0.01
alpha: ~
criterion: charbonnier
num_proj_fd: 256
phase_weight_fd: 1.0
stride_fd: 1
clip_min: 0
loss_weight: 1.0
```
## perceptualloss


```yaml
type: perceptualloss
layer_weights: ~
w_lambda: 0.01
alpha: ~
criterion: charbonnier
num_proj_fd: 256
phase_weight_fd: 1.0
stride_fd: 1
clip_min: 0
loss_weight: 1.0
```
## sparkloss

Frozen perceptual features from [umzi2/SparK](https://github.com/umzi2/SparK),
using the `weight_33_ep` release. The asset is named
`convnext_tiny_1kpretrained_timm_style.pth`, but it requires the fork's
Inception-style depthwise blocks and BatchNorm, not stock ConvNeXt.

Add this entry under `train.losses`:

```yaml
- type: sparkloss
  loss_weight: 1.0
  criterion: l1
  layer_weights:
    stage1: 1.0
    stage2: 1.0
    stage3: 1.0
    stage4: 1.0
  weights_path: ~
```

- `loss_weight` is applied once by the trainer. `1.0` above is an example, not a
  calibrated equivalent to VGG's weight or a validated GAN recipe.
- `criterion` is `l1` (default) or `mse`. Each stage contributes its mean feature
  distance times its stage weight; the stage contributions are summed.
- `layer_weights` defaults to all four stages equally. Omitted stages contribute
  zero; evaluation stops after the last selected stage. Stage channels are
  96/192/384/768, with strides 4/8/16/32. Selecting all four requires both image
  axes to be at least 32 pixels; arbitrary larger sizes are accepted without
  resizing or extra padding.
- Inputs are matching RGB tensors in the `[0, 1]` convention. ImageNet mean/std
  normalization matches the fork's preprocessing. Predictions are not clipped.
- The encoder's 23,974,296 parameters and BatchNorm statistics stay frozen in
  evaluation mode. Prediction features remain differentiable; target features
  use `no_grad`. There is no stochastic masking, classifier, or reconstruction
  decoder in this loss.
- Feature extraction and distance reduction run in FP32 even during BF16/FP16
  AMP training. The generator may still use AMP. This is not a training-speed
  optimization; GPU speed and memory depend on the workload.
- `weights_path: ~` downloads the pinned 33-epoch asset once into PyTorch's hub
  checkpoint cache, with SHA256 verification, and loads it using
  `weights_only=True`. A local path avoids downloading and must contain an
  exact-compatible, direct encoder state dictionary. Loading is strict; missing
  weights never silently become random features.

SparK supplies the pretrained encoder, not this restoration-loss aggregation.
The all-active dense feature path is used here; the original masked-reconstruction
pretraining objective is not run. This loss can coexist with VGG, but no artifact
suppression, PSNR/SSIM improvement, or best combination/weight is established by
the implementation checks.

## ssimloss


```yaml
type: ssimloss
loss_weight: 1.0
channels: 3
downsample: false
test_y_channel: true
color_space: yiq
crop_border: 0.0
loss_weight: 1.0
```
