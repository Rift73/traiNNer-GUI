# Training Guidelines
This page includes a summary of suggested guidelines to follow for training, and best practices based on research and community testing. Note these guidelines may not be optimal for every single architecture, where each may have their own individual quirks.


## Background
Super resolution methods generally fall into two categories: fidelity oriented and perceptual oriented.

Fidelity oriented super resolution typically uses bicubic degradations for the dataset and only optimizes a pixel loss such as L1. When researchers propose a new super resolution architecture, they use a standard research dataset such as DF2K, train from scratch and evaluate their architecture using the fidelity oriented method, and most offer their pretrained models for download on their original project repository. Most research models are trained on multiple NVIDIA data center GPUs, which allows the use of higher batch such as 32 or 64, and they are trained quickly because multiple GPUs are used in parallel. Since higher batch size results in higher fidelity, it's difficult to match the fidelity of research pretrains on consumer GPUs with limited VRAM on a single GPU.

- Training from scratch on L1 loss is stable as long as the architecture is stable, and the model continues to improve the more it's trained, though there are diminishing returns for training longer.
- Because these models are trained on datasets with bicubic degradations and pixel loss only, they produce blurry upscales which lack convincing textures. But they are faithful to the input image.

Perceptual oriented models tend to use more complex degradations, such as one or more of JPEG compression, blur, or noise, and optimize multiple losses, including a pixel loss, perceptual loss, and adversarial (GAN) loss. When training a perceptual model, a pretrain should always be used for the generator. Fidelity oriented models, like the official research models, are well suited for this purpose.

- Training with multiple losses including GAN loss is much less stable, and training can collapse when trained for too long.
- Perceptual oriented models are able to restore fine details on degraded images, but they may trade some faithfulness to the input image.

Most users find perceptual oriented models more practical and only ever train perceptual oriented models.

## Reading GAN health during still-image training

Set `logger.gan_health: true` to add the following scalars to console and TensorBoard logs at `logger.print_freq`. Each reading is from the final microbatch of that printed, accepted generator update. A skipped GAN schedule or a rejected window produces no sample. The trainer observes the existing backward and discriminator predictions.

| Scalar | Meaning |
| --- | --- |
| `grad_sr_gan` | L2 norm over the whole SR loss image batch of the weighted adversarial gradient, including its current ramp. |
| `grad_sr_other` | L2 norm of the other gradients reaching the same image, computed as total minus adversarial. |
| `grad_sr_ratio` | GAN norm divided by the other norm, present for a finite comparison with a positive denominator. A value above one means a larger adversarial gradient at this image. |
| `out_d_acc` | Fraction of real and fake discriminator scores on the correct side of the decision boundary. Each patch score counts once. Ties count as incorrect. |
| `out_d_abs` | Mean absolute distance of those scores from the same boundary. |

Gradient norms remove the accumulation divisor and the generator's backward loss scale. They measure the image after the optional output clamp and before the optional wavelet transform. The subtraction has the resolution of the image gradient dtype under AMP. Feature-only objectives can update G without sending a gradient through this image. If the clamp is active and a raw-image LDL, VOSR or output-distillation term bypasses it, the other norm and ratio are omitted. Use `grad_norm_g` for the existing accumulated parameter-gradient measurement.

D readings require an applied D step on the sampled microbatch. Hinge and logistic losses use zero, and least-squares GAN uses the midpoint of the configured labels, respecting inverted labels. BCE labels must straddle probability 0.5. Wasserstein critics have no calibrated real/fake threshold, so only the existing real and fake score means apply. These are training scores on the inputs D actually saw, including wavelet inputs when enabled.

The option covers eager, single-device still-image SR and its OTF variants with a generator optimizer and native `GANLoss`. Recurrent, AE, supervised, compiled and multi-device training are outside its scope. Leave it unset when importing a GAN recipe unless the user requests diagnostics. Sampling adds temporary image-sized gradient storage and reductions only at print boundaries.

## Training a Perceptual Model with an Existing Pretrain

When training a perceptual model, a pretrain model should always be used for the generator. While any model can be used for your pretrain, official research models are often a good choice. When you use a pretrain model, you risk introducing any flaws of the pretrain model into your own model. Official research models are likely to have the least amount of flaws, and they have high fidelity to give your model a good starting point.

1. Train a perceptual model with degradations:
   - Download an official pretrain for the architecture you want to train, and move it to `experiments/pretrained_models`
   - Open the `*_gan.yml` template and set the `pretrain_network_g` path to the pretrain you downloaded

## Training a Perceptual Model from Scratch

While using an existing pretrain is recommended when training a perceptual model, you might prefer to train a model completely from scratch, including training your own pretrain with your own dataset. When training from scratch, the general strategy is to train the model with easy to learn tasks before difficult tasks. So to train a 4x scale model to handle degradations, a common process is:

1. Train 2x bicubic from scratch:
   - LR images are simply bicubic downscales of the HR images.
   - Only Charbonnier loss is enabled.
   - The training settings of the official paper models are often a safe choice for batch size, crop size, scheduler milestones, and total iterations. The training settings are often available on their GitHub repo or described in their paper. For example the training settings for DAT_2_X2 are [here](https://github.com/zhengchen1999/DAT/blob/main/options/Train/train_DAT_2_x2.yml):
      - This model is trained until distortion validation metrics (PSNR and SSIM) peak, which is often at least 500,000 iterations, depending on the architecture
      - With the AdamW optimizer, the learning rate is 2e-4 when training from scratch
      - Larger batch size benefits training from scratch. Gradient accumulation can be used to train with a larger effective batch size. For example the official DAT 2 training settings use a batch size of 8 per GPU on 4 NVIDIA A100 GPUs which is a total batch size of 32. A single RTX 4090 does not have enough VRAM to train 2x DAT 2 with batch 32, but it can be trained with batch 8 and accum_iter 4, for a total effective batch size of 32.
   - The `*_fidelity.yml` templates are set up to train with these settings
   - This model is your 2x fidelity pretrain
2. Train 4x bicubic:
   - Use the 2x pretrain from the previous step as a pretrain for this 4x model, with `strict_load_g` set to `False`
   - The LR images are bicubic downcsales of the HR images
   - Only Charbonnier loss is enabled
   - The same batch and lq crop size (`lq_size`) from the 2x training settings are used for the 4x training settings. The training settings of the official paper models are a safe choice for scheduler milestones and total iterations. The milestones and total iterations are usually cut in half from the previous step.
       - This model is trained until distortion validation metrics (PSNR and SSIM) peak, which is often at least 250,000 iterations, depending on the architecture.
       - The learning rate typically is reduced to 1e-4 when finetuning.
   - This model is your 4x fidelity pretrain

3. Train 4x with degadations
   - Use the 4x pretrain from the previous step as a pretrain for this 4x model, with `strict_load_g` set to `True`
   - The LR images are degraded with the types of degradations you want the model to handle, such as JPEG or h264
   - All of the typical losses are enabled for this step: MSSIM, Perceptual, HSLuv, GAN
   - The learning rate is 1e-4 for the generator and discriminator optimizers
   - This stage of training benefits from larger `lq_size` and batch size can be reduced to allow larger `lq_size`
   - This model is trained until it looks good. Distortion validation metrics (PSNR and SSIM) are not a good indicator of good perceptual quality. Perceptual validation metrics (TOPIQ, LPIPS, DISTS) are correlated with human perception of visual quality to some extent, but your eyes should be the final judge
   - The `*_gan.yml` templates are set up to train with these settings

## Automatic recovery and checkpoint safety

On single-rank WSL/Linux, the normal `python train.py -opt ...` command supervises
training automatically. Twenty consecutive nonfinite losses or gradient
rejections trigger a fresh worker from the last complete checkpoint. Recognized
fatal CUDA-context errors and native crashes can also recover from that worker's
recorded checkpoint. Configuration errors, ordinary Python exceptions, OOM, and
Ctrl+C do not trigger this retry path.

Retries are unlimited by default (`--recovery-max-restarts 0`). Repeated failures
without a newer complete checkpoint wait 5, 10, 20, 40, then 60 seconds between
retries. A newer checkpoint resets the backoff. Use a positive restart limit to
bound attempts without progress, or `--no-auto-recover` to disable supervision.
Unlimited retries do not repair a persistently bad checkpoint; review the log
and `training_recovery.jsonl` if the same failure keeps recurring. Recovery stops
if the recorded checkpoint is missing/incomplete or the previous worker group
cannot be stopped safely.

Checkpoint files are written to temporary files in their destination directory
and individually published by atomic replacement. A whole checkpoint set is
not one atomic transaction: recovery only advances after the required files
have all been saved. Periodic filesystem write failures are logged without
pruning the last healthy checkpoint; the next scheduled save tries again.
Unexpected training crashes do not save potentially partial model/optimizer
state. A graceful Ctrl+C save uses the last completed update's iteration label.

GAN updates validate both G and D before stepping either optimizer. Rejection
discards accumulated gradients and provisional adaptive-D history. Actual AMP
overflow still backs off its scaler; a healthy scaler does not advance merely
because the other network rejected the update. Adaptive-D history is included
in new training states; older states load with a warning and fresh controller
history.

Eligible CUDA Adam/AdamW runs select fused updates when neither `fused`
nor `foreach` is set. A resumed optimizer retains its saved settings.
For a single fused generator optimizer with accumulation one and the gradient
scaler disabled, the GPU can reject a nonfinite update before the host reads
the result. The optimizer skips its weight and moment update. EMA can still
average once toward unchanged weights. A later call corrects the iteration
and scheduler counts. A checkpoint saved at that boundary can carry a label
one update ahead.
FP16 runs with an enabled scaler use synchronous overflow handling.

### Restore model buffers on a rejected update

Set `train.rollback_buffers_on_reject: true` for single-device still-image SR, including paired and on-the-fly degradation training. The default is `false`. Before the first forward of each accumulation window, the trainer snapshots the online generator and discriminator buffers. If either network produces a rejected loss or gradient, both recover their pre-window values. A mid-window checkpoint writes those same committed values and then restores the live microbatch state. Accepted windows, including scheduled or adaptive discriminator skips, follow the usual update and EMA rules.

This protects fixed, writable registered tensors updated in place, including BatchNorm running means, variances and counters, spectral-normalization power-iteration vectors and nonpersistent buffers. The check covers the online networks only. Aether QAT has its own observer and phase rollback. Custom networks with lazy, replaced or resized buffers, parameters mutated by forward, or mutable Python state are outside this contract.

Allow one extra device copy of all registered G/D buffers, including constants, and one copy pass per window. A rejection copies the saved values back. A mid-window save temporarily needs a second buffer copy. Models with protected buffers use immediate finite checks before either optimizer steps. Buffer-free models retain their usual eligibility for the deferred GPU guard.

## Architecture-specific compiler and export hooks

Architectures can provide `get_compile_inductor_options(compile_mode)` returning
a JSON-compatible dictionary. These overrides are merged with the selected
compile mode's options; when the architecture supplies `compile_cache_key_tag`,
the overrides are included in that tag as well. Architectures without the hook
retain the ordinary compile path.

An architecture's `onnx_constant_folding_size_threshold` is passed to OnnxSlim.
When set, the legacy exporter and subsequent ONNX Runtime optimization do not
perform unbounded constant folding. This is an architecture hook, not a new
training YAML option; it does not itself change model weights or precision.


## Importing NeoSR compact training objectives

The paired and on-the-fly compact recipes at NeoSR commit `3fe738f4` declare the same generator and discriminator objectives. Their TOML import records the generator's PyTorch AdamW settings, L1 pixel loss, the GAN score formula (BCE with logits), and the RGB MS-SSIM objective. The GAN weight of 0.3 applies to the generator. The discriminator sums its real and fake terms at unit weight. [Paired recipe](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/options/train_compact.toml) · [On-the-fly recipe](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/options/train_compact_otf.toml)

Use the registered objective directly in a native recipe:

```yaml
train:
  losses:
    - type: neosrmssimloss
      loss_weight: 1.0
```

`neosrmssimloss` compares supplied RGB tensors at five scales. It uses an 11×11 zero-padded Gaussian (σ = 1.5) at every scale, takes a spatial mean for each image and channel, floors the bases at 1e-6 before fractional powers, and includes luminance at the last scale. It accepts the compact recipes' default kernel settings. The trainer applies the scalar weight once. The existing `mssimloss` has separate luminance, padding, aggregation, and color-space conventions. [NeoSR loss implementation](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/losses/ssim_loss.py)

Both compact recipes import the following GT-referenced objectives at unit weight. They can also be selected under `train.losses` in an image restoration recipe.

- `neosrldlloss` measures the residual between the prediction and reference. It weights both images by the global variance raised to 0.2 and the unbiased local variance, with gradients through both factors. The local window uses reflect padding and FP64 moments. `ksize` defaults to 7 and must be odd and at least 3. Each spatial dimension must exceed half the window size rounded down. The default criterion is `l1`. Alternatives are `l2`, `huber` and `chc` (clipped Charbonnier). Native `ldlloss` retains its separate EMA residual mask. [Pinned LDL implementation](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/losses/ldl_loss.py)
- `neosrconsistencyloss` measures CIE L* lightness and Oklab chroma in FP32 on matching RGB images. The prediction uses an inward-gradient clamp at [1/255, 1]. By default, a 21-tap Gaussian blurs only the lightness input, and each spatial dimension must exceed 10. Set `blur: false` to use unblurred lightness. The default `chc` criterion clips elementwise Charbonnier distances to [0, 1] before averaging each component. `l1` is also available. With `cosim: true`, it adds the equally weighted lightness and chroma cosine distances when their combined value is below 0.001. `brightness` and `saturation` scale reference lightness and chroma respectively and both default to 1. [Pinned consistency implementation](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/losses/consistency_loss.py)

NeoSR configurations that select an upsampled LQ target or enable `match_lq_colors` require review. The two compact recipes use their paired GT targets.

Both compact recipes select `train.ema_mode: neosr`, `ema_decay: 0.999` and `ema_power: 0.75`. The first applied generator update copies the full generator state into a frozen evaluation model. For subsequent update number `k` after that copy (starting at 1), the decay is `min(ema_decay, 1 - (1 + k) ** -ema_power)`. Omitting the power in NeoSR mode selects fixed decay. Each update averages the parameters and copies the current buffers, including integer counters. Accumulation microbatches, discriminator-only steps and rejected optimizer transactions leave the EMA untouched. In this mode the trainer resolves the generator's finite check before advancing the average. The native G/D transaction still rejects both updates when either fails. [Pinned EMA](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/models/phase.py) · [Pinned update boundary](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/models/image.py)

Validation and portable `net_g_ema` weights use that evaluation model. Full-state resume restores the separate online generator, EMA weights and successful generator-update count. Keep the EMA mode, decay and power for a continuation. To change them, start a new run from generator weights. Existing native-mode recipes keep their averaging behavior and default power of 10.

Both compact imports set `train.loss_output_clamp: restore` and `train.gan_wavelet_start_iter: 80000`. Native recipes default to `none` and `null`, respectively.

`restore` bounds predicted RGB to [0, 1] for ordinary image losses and the GAN. These terms share one clamped prediction. Its backward pass sums their gradients first, then allows only inward gradients for values outside the interval. At either endpoint the gradient passes through. `hard` uses the ordinary [1/255, 1] clamp and blocks gradients outside that interval. Raw generator output remains available for validation, inference, observations, native EMA-based LDL, VOSR and distillation. Ground truth retains its original values. NeoSR TOML `clamp = true` imports as `hard`, false selects the native default, and omission imports as `restore`. [NeoSR clamp](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/losses/base.py)

The GAN receives ordinary images before `gan_wavelet_start_iter`. Starting with that update, its generator, real-discriminator and fake-discriminator passes all use three scaled sym19 stationary-wavelet detail bands of BT.601 luma. The transform carries gradients on the generator pass. Discriminator training detaches the fake input. Every microbatch for the same candidate update uses the same view, including a retried update. Images must be at least 19 pixels high and wide for circular padding. Zero selects wavelets from the first update. An explicit NeoSR GAN-local `wavelet_init` overrides the global start. [NeoSR adversarial view](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/losses/wavelet_guided.py)

The discriminator optimizer in both recipes imports as `NeoSRAdanScheduleFree`. Their `fdl_opt` imports as `neosrfdlloss` with the DINOv2 backbone and weight 0.02. Resolve any remaining `neosr_review` entries before launching an imported run. [Pinned FDL implementation](https://github.com/Corpsecreate/neosr/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/neosr/losses/fdl_loss.py)

## AetherNet quantization-aware training

AetherNet supports scale 2, 3, 4 or 8. Use `aether_small`, `aether_medium` or `aether_large` for the pinned presets, or `aether` for a configurable depth and width. Omitting `train.qat` selects ordinary float training. The following settings can be added to an image restoration recipe with a generator optimizer and image loss.

```yaml
scale: 2
use_amp: false
use_compile: false
network_g:
  type: aether_small
train:
  total_iter: 1000000
  ema_mode: neosr
  ema_decay: 0.999
  ema_power: 0.75
  qat:
    type: int8
    calibration_iters: 500
    quantize_residual: false
```

Preparation fuses the large and small convolution branches before the optimizer and EMA acquire their parameters. DeploymentNorm keeps collecting statistics until its freeze update. The generator simulates symmetric INT8 activations per tensor and weights per output channel. The image input and output convolutions, the first pre-upsampling convolution and the attention sigmoids stay float. The prepared model runs in eager FP32 on one device. EMA can be disabled or set to `neosr`, which averages parameters and copies every observer range, scale, enable flag and norm buffer on each applied G update.

| Setting | Update boundary |
|---|---|
| `calibration_iters` | Updates 1 through N collect ranges with fake quantization disabled. The default of zero enables fake quantization on update 1. |
| `freeze_norm_iter` | The first update with frozen norm statistics. Defaults to `int(0.7 * total_iter)`. |
| `freeze_observers_iter` | The first update with frozen activation and weight ranges. Defaults to `int(0.8 * total_iter)`. |
| `quantize_residual` | Selects fake quantization of the block residual addition. Omission inherits the QAT checkpoint topology or the architecture setting, which defaults to true. Explicit settings must agree. |

Set a freeze update to `-1` to keep collecting statistics. All microbatches in a candidate update share its phase. A rejected G/D transaction restores the generator buffers to the start of the window. A checkpoint taken between microbatches contains the last accepted buffers. Training validation holds observers and norms fixed and retains the current fake-quant flag. During calibration this gives a float prediction. Standalone QAT evaluation enables fake quantization using the saved ranges.

A fresh run may start from ordinary weights, a deployed float checkpoint with `network_g.fused_init: true`, or complete QAT weights. Leave `network_g.fused_init` false for training from scratch. Both raw and EMA QAT files contain the architecture's observer tensors. Full resume requires those original role-specific weights when EMA is enabled, the matching training state, the same residual topology and the same resolved phase boundaries. A weights-only start begins a new update schedule.

This integration covers the AetherNet image generator. The CPU training check used a one-block, scale-2 `aether`. Larger presets, accelerator execution and deployment backend coverage have separate verification requirements. [Pinned AetherNet architecture and recipe notes](https://github.com/korvin2000/neosr-dev/blob/3fe738f43d66d17271db8b97a67c7e563f39cefa/docs/aether.md)
