# VOSR 2 teacher integration

This fork supports two distinct uses of a VOSR 2 restoration. Neither mode is enabled by default.

## Runtime assets

The teacher is the fixed 512 x 512 FP8 TensorRT window runner in
`traiNNer/losses/vosr2_trt.py`. It requires CUDA, a compatible TensorRT Python
runtime, the `vosr2_qkv` plugin, and ten compatible stage engines:
`encoder_pre`, `encoder_qkv`, `encoder_post`, `decoder_pre`,
`decoder_qkv`, `decoder_post`, `attention_grouped`, `dino`, `dit_kv`,
and `dit_cached`, each with a `.plan` suffix. These binaries and model
weights are not distributed in this repository. TensorRT is imported only
when a teacher is constructed; ordinary training does not require it.

The degraded RGB LR is quantized to 8-bit, enlarged with Pillow-compatible
bicubic arithmetic, reflect-padded to 512, and restored with fresh noise.
The result is cropped and wavelet-colour-corrected against the enlarged LR.
Each restored HR patch must fit within 512 x 512. `lanes` controls concurrent
windows, not the training batch size. The teacher is frozen and is not part
of the compiled generator or loss graph.

## Additional teacher loss

Add to the existing `train.losses` list:

```yaml
- type: vosrloss
  loss_weight: 0.1
  engine_dir: /path/to/vosr2-512-engines
  plugin: /path/to/vosr2_qkv.so
  criterion: charbonnier
  lanes: 2
  still_frames: 2
```

`criterion` can be `charbonnier`, `l1`, or `perceptual`. The latter accepts
`perceptual_options` for the existing FP32 perceptual loss (without its own
`loss_weight`). The ordinary HR target remains unchanged.

This mode requires a GAN loss with `gan_type: vanilla`, `lsgan`, or
`wgan_softplus`. The trainer uses detached running means of the raw GAN
term and the raw VOSR distance to scale the VOSR distance. Its `loss_weight`
is relative to the raw GAN loss magnitude, not a guarantee of equal
parameter-gradient magnitudes or equal influence on image quality.
The balancing state is committed only after an accepted training attempt.
A teacher restoration is detached from autograd; gradients flow into the SR
prediction through the selected distance.

For RDUAL clips, the teacher evaluates key frames and can reuse their targets
only on still regions of nearby frames; `still_frames: 0` disables that reuse.
This setting has no effect on single-image training. See
`traiNNer/models/rdual_model.py` for the explicit still-region thresholds.

## Replace the HR target

For single-image on-the-fly degradation training:

```yaml
high_order_degradation: true
input_pixel_format: rgb
output_pixel_format: rgb
otf_hf_noise_prob: 0.0
train:
  vosr_target:
    engine_dir: /path/to/vosr2-512-engines
    plugin: /path/to/vosr2_qkv.so
    lanes: 2
```

Merge this fragment into a complete training config; it does not supply
optimizer, loss, or dataset settings.

After the normal degradation pipeline and training-pair queue select the LR,
its VOSR restoration replaces `gt`. All losses and the discriminator therefore
see that restored target. The inference tensor is cloned into an ordinary
tensor before losses can save it for backward.

This mode rejects simultaneous `vosrloss`, `target_dataroot_gt`, HR noise,
non-RGB formats, missing/oversize HR crop settings, and disabled OTF degradation.
RDUAL clip training does not support `train.vosr_target`.

## Evidence boundary and references

CPU tests cover bicubic arithmetic, colour correction, balancing, option
validation, and queue/target wiring with a stand-in teacher. Real TensorRT
teacher tests require the external engines, plugin, fixtures and a GPU;
CPU success does not establish runtime performance or restoration quality.

- Upstream model: https://github.com/cswry/VOSR
- Teacher and distances: `traiNNer/losses/vosr_loss.py`
- Engine execution: `traiNNer/losses/vosr2_trt.py`
- Single-image integration: `traiNNer/models/sr_model.py`,
  `traiNNer/models/realesrgan_model.py`
- Clip integration: `traiNNer/models/rdual_model.py`
