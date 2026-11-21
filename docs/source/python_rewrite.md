# Python rewrite: maintainer's guide

This checkout is a separate Python readability and maintainability rewrite of
traiNNer-redux, starting from `17e9e667842e5fe551ffaa2d1d7015d1557fcd2a`.
The aim is to make ownership and execution order easier to follow without
changing the training algorithm. Existing CLI commands, YAML options, model
registrations, and checkpoint formats remain the compatibility boundary.

Use the existing [getting-started guide](getting_started.md) to run training,
the [configuration reference](config_reference.md) for options, and the
[training guidelines](training_guidelines.md) for training and recovery policy.
This page explains implementation contracts, not a second configuration format.

## Custom fork compatibility

This checkout follows the custom fork, including its uncommitted November 21
port. The Python entry points, YAML recipes, model and loss registrations, and
checkpoint layout remain compatible. The native PTX and WebP helpers belong to
the degradation implementation shared with the fork. See the [PTX build
instructions](../../traiNNer/csrc/README.md) and the [portable WebP build
script](../../traiNNer/csrc/webp_reconstruct/build.sh).

| When changing | Start here |
| --- | --- |
| DUAL routing, regional attention, or recurrent clip training | [DUAL/RDUAL and RSR guide](../RDUAL.md) and the [model factory](../../traiNNer/models/__init__.py). RDUAL and RSR select clip trainers before the generic image-degradation branch. |
| WSR still-image training or RSR temporal behavior | [WSR architecture](../../traiNNer/archs/wsr_arch.py) and [RSR architecture](../../traiNNer/archs/rsr_arch.py). The RSR trainer inherits the RDUAL clip lifecycle. |
| Still-image, clip, or DVD inputs | [Loader and collation](../../traiNNer/data/__init__.py) establish batch layout and worker policy. [RealESRGANModel](../../traiNNer/models/realesrgan_model.py) advances degradation batches and installs the result for training. For alignment, codec defaults, and supported datasets see the [configuration reference](config_reference.md). |
| VOSR-based supervision | [VOSR teacher guide](../VOSR_loss.md) distinguishes the auxiliary loss from replacing the HR target. It lists the external engine and plugin requirements. |
| GAN conditioning, distillation, or nonfinite updates | [SRModel](../../traiNNer/models/sr_model.py) owns the optimizer transaction and LDL-EMA forward. Its inference, validation, and distillation helpers own their respective paths. See the [training guidelines](training_guidelines.md) for synchronous and deferred rejection. |

## Read the code in execution order

| Location | Responsibility |
| --- | --- |
| [`train.py`](../../train.py) | CLI boundary, training lifecycle, accepted-update accounting, scheduled logging/saving/validation, and exit handling. |
| [`training/data.py`](../../traiNNer/training/data.py) | Construct train/validation datasets, samplers, and loaders; calculate existing epoch/iteration counts. |
| [`training/checkpoints.py`](../../traiNNer/training/checkpoints.py) | Periodic save orchestration: write, verify a complete fresh set, publish recovery progress, then apply retention. |
| [`traiNNer/training_recovery.py`](../../traiNNer/training_recovery.py) | Torch-free supervisor entry, worker failure reports, complete-checkpoint selection, and opt-in retention. |
| [`traiNNer/models/__init__.py`](../../traiNNer/models/__init__.py) | Select the SR, recurrent clip, degradation, paired-degradation, or autoencoder trainer and determine degradation-ahead eligibility. |
| [`BaseModel`](../../traiNNer/models/base_model.py) | Shared device/compiler setup, optimizer/scheduler lifecycle, checkpoint interfaces, and validation dispatch. |
| [`checkpoint_payload.py`](../../traiNNer/models/checkpoint_payload.py) | Export-key normalization and CPU-layout conversion requests, checkpoint-wrapper selection, and common-prefix removal; serialization and atomic file publication stay in `BaseModel`. |
| [`model_compile.py`](../../traiNNer/models/model_compile.py) | Resolve architecture Inductor options, enable the precompile-cache policy, and build process-wide compile ABI tags. |
| [`SRModel`](../../traiNNer/models/sr_model.py) | Super-resolution training state, loss/backward/update ordering, EMA, inference, and validation. |
| [`sr_distillation.py`](../../traiNNer/models/sr_distillation.py) | Frozen-teacher distance, feature-return contract, schedule scaling, and feature transforms; pairing and loss orchestration stay on `SRModel`. |
| [`sr_inference.py`](../../traiNNer/models/sr_inference.py) | Overlapping-tile blending and the validation-only network-mode/cache context. |
| [`sr_validation.py`](../../traiNNer/models/sr_validation.py) | One dataset's validation sweep, tensor/fallback metrics, image-path selection, and image export. |
| [`RealESRGANModel`](../../traiNNer/models/realesrgan_model.py) and [`RealESRGANPairedModel`](../../traiNNer/models/realesrgan_paired_model.py) | Specialized data preparation; inherit the SR update contract. |
| [`realesrgan_noise.py`](../../traiNNer/models/realesrgan_noise.py) | High-frequency target-noise generation with explicit options, project RNG, and the model's Beta sampler; queue/feed ordering stays on the model. |
| [`AEModel`](../../traiNNer/models/ae_model.py) | Separate autoencoder training path sharing `BaseModel`; not a synonym for the SR update path. |
| [`traiNNer/utils/compile_cache.py`](../../traiNNer/utils/compile_cache.py) | Existing compiler-runtime cache compatibility support. |

Model and dataset registries, architecture implementations, degradation
operators, losses, and option definitions remain in their existing packages.
Moving orchestration does not transfer ownership of their mathematical behavior.
`SRModel` still owns networks, optimizer state, metric results, and validation
preparation keys; the extracted helpers do not maintain a second copy of that
state. Its existing inference/validation methods remain callable wrappers.

### Launch and initialization

The CLI invokes `maybe_supervise_training()` **before importing Torch**. On
supported single-rank Linux/WSL launches, the waiting supervisor must not create
a CUDA runtime. A supervised worker does not launch another supervisor. Importing
`train` and calling `train_pipeline()` remains an in-process route, and CLI help
is parsed before requiring CUDA.

Worker initialization preserves this order:

1. Parse options, check CUDA availability, resolve recovery/retention, configure
   deterministic behavior, and seed the run.
2. Select/load resume state on CPU; prepare experiment directories and copy the
   user configuration before initializing the root logger.
3. Initialize external/TensorBoard logging and build train/validation loaders.
4. Construct the selected model, restore optimizer/scheduler/scaler state when
   resuming, then prepare the **next update's** warm-up learning rate.
5. Construct the prefetcher and enter the epoch/microbatch loop.

Do not move root-logger initialization ahead of experiment setup. Do not consume
a scheduler step while preparing initial or resumed warm-up. Dataset-size and
accumulation calculations still determine epoch counts; they are not a new
termination or sampling policy.

## A microbatch is not a completed update

`current_accum_iter` tracks the accumulation window. At its boundary, the loop
temporarily increments `current_iter` and passes `apply_gradient=True` to the
model. A synchronously reported rejection rolls the candidate iteration back
at that boundary. The eligible CUDA fast path can report a rejected step on
a later call.
`last_completed_iter` records the last accepted update for safe interruption
labeling. These counters must not be merged.

For each microbatch, the pipeline feeds data inside `data_feed_context()`, calls
`optimize_parameters()`, and attempts optional compiled-cache publication. It
then honors `reset_accumulation_requested` and computes whether an update was
accepted. Only an accepted accumulation boundary advances the learning-rate
schedule, ordinary iteration logging, periodic checkpointing, and scheduled
validation. The relative order is update, scheduler, log, checkpoint, validation.

The base acceptance flags are a model/pipeline protocol. Custom training code
must reset them for each attempt and explicitly reject failed attempts; returning
early is not by itself a rejection.
Within an accepted update, individual scheduler advancement still honors the
corresponding optimizer's skip flag.

### SR generator/discriminator transaction

`SRModel.optimize_parameters()` deliberately keeps a single ordered update
transaction:

1. Start an attempt and, when required, mark a CUDA Graph step boundary.
2. Freeze discriminator parameters for generator work; prepare conditioning,
   pixel-format conversion, and optional teacher targets.
3. Evaluate generator losses in their existing order, divide the backward loss
   by `accum_iters`, check finiteness, and backpropagate. At an update boundary,
   unscale, measure gradients, check finiteness, and optionally clip them.
4. Evaluate and backpropagate discriminator losses if configured and not skipped
   by adaptive-D. At the boundary, perform the corresponding gradient checks.
5. Only after both applicable paths pass, step/update their scalers and
   optimizers, clear gradients, commit adaptive-D history, aggregate logs, and
   update EMA when eligible.

Generator readiness does not mean the generator has already stepped: the
discriminator may still reject the whole window. Never move the G optimizer step
above D validation. A final adaptive-D skip must clear partial D gradients rather
than carry them into the next effective batch.

Loss ordering, weighting, autocast regions, detaches, accumulation division,
pixel-format conversion, distillation schedules, and gradient clipping are
numerical contracts. Extracting a helper is not permission to reorder them.

### Rejection and AMP state

On the synchronous path, a nonfinite loss or gradient rejects the entire accumulated batch, releases its
output, clears G/D gradients, restores the adaptive-D controller's window-start
value, and requests an accumulation reset. No optimizer or EMA update commits
for that rejected attempt. Logging/recovery counters still record the rejection.

GradScaler state also has a lifecycle: `unscale_()` has already changed its phase.
Rejection clears that phase, retains genuine overflow backoff, and restores a
healthy scaler's state if the other network caused rejection. Treating both
scalers as either unconditional updates or unconditional rollbacks is incorrect.
On the single-generator, fused CUDA Adam/AdamW path with accumulation one and
the gradient scaler disabled, the device can skip a bad update before the host
sees it. The weights and optimizer moments stay put. EMA may advance once
toward those unchanged weights. A later call accounts for the rejection in
the scheduler and iteration totals. A checkpoint at the boundary can carry a
label one update ahead. Logging, validation, saving, and normal shutdown settle
pending device results.

Consecutive rejected attempts eventually raise `PersistentNonfiniteError`; see
the [recovery policy](training_guidelines.md#automatic-recovery-and-checkpoint-safety).

## EMA, validation, and compiler caches

These are different caches with different lifetimes:

* **EMA weights** update at an eligible accepted SR update boundary, not after
  every microbatch. Synchronous scaler rejection skips EMA. The deferred path
  follows the exception described above.
* **Validation preparation** is keyed by inference-network identity and iteration.
  Some architectures build reversible eval-only caches. Preparing once per
  weight identity avoids rebuilding them for every image; an EMA update
  invalidates the key even though the EMA module remained in eval mode.
* **Compiled artifacts** are an optional startup optimization, not a checkpoint
  or proof of numerical equivalence. Cache identity incorporates supported
  architecture/compiler/precision/layout/geometry information. Changed contracts
  may legitimately cause a cold compile.

Validation selects EMA when available. During training without EMA, it can use
the unwrapped training network and must restore its training mode afterward.
The validation context temporarily changes `is_train`, switches schedule-free
optimizers to evaluation weights, and restores modes in `finally`. Contexts
cannot be nested. Standalone `test()` also manages these transitions when it is
not already inside a validation context. Reuse across datasets applies only when
the prepared inference state remains valid; training-mode restoration invalidates
it for the non-EMA training network.

Keep image-save scheduling separate from metric evaluation. The pipeline passes
each dataset's existing `should_save_img()` decision to validation; disabling an
image write is not a request to skip the validation pass.

Compilation happens after device transfer and before optional parallel wrapping.
Existing architecture preparation/options hooks and cache ABI tags remain the
integration points. A compiled-artifact publication failure logs a warning and
disables further publication attempts in that process without aborting healthy
training. Required compiler APIs and invalid architecture options still fail
explicitly; they are not silently replaced by an eager execution fallback.

## Checkpoints are files plus a completion policy

`SRModel.save()` determines which networks belong in the checkpoint. With EMA,
the inference EMA network is written to the models directory and the training
generator to the resume-models directory; D is a resume network when present.
Without EMA, the generator uses the ordinary models directory. `BaseModel`
retains the existing network serialization and training-state interfaces.
Payload normalization keeps existing wrapper precedence, single-pass prefix
handling, tensor sharing, and subclass prefix-removal dispatch. It is not a
recursive migration or a change to parameter-key names. Export retains the
existing `Tensor.to("cpu", memory_format=...)` operation rather than inserting
clones or contiguity repairs; an already-CPU noncontiguous tensor can therefore
retain the baseline safetensors serialization failure.

Training state includes epoch/iteration, optimizer and scheduler states, enabled
AMP scaler state, EMA step state, and adaptive-D history where applicable. An
interrupted partial accumulation window saves committed adaptive-D history, not
its provisional microbatch value. Older states without adaptive-D history remain
loadable with the existing warning. The `latest` network save (`current_iter=-1`)
does not produce a numbered resumable `.state` file.

Atomic replacement protects **one file**, not an entire G/EMA/D/state set.
Recovery advances its anchor only after every required file is complete and
freshly written. Optional retention acts only after a successful complete
regular save, preserves protected/manual checkpoints, and does not run merely
because training started. Do not conflate a successful network write with a
complete resumable checkpoint.

Periodic filesystem write failures do not advance the recorded recovery anchor.
That does not guarantee the files of an already-recorded iteration were left
unchanged if that same iteration is overwritten; see the baseline issue below.
Ordinary unexpected training failures and persistent nonfinite failures do not
save potentially partial model/optimizer state. A recognized poisoned CUDA
context bypasses GPU checkpointing and CUDA cleanup; the outer handler publishes
a CPU-side failure report. A graceful Ctrl+C uses the last completed iteration
label. These are distinct exit paths, not a shared unconditional `finally: save`.

## Extending the code safely

* Add an architecture or loss through its existing registry and typed options;
  see [contributing](contributing.md) and the existing
  [compiler/export hooks](training_guidelines.md#architecture-specific-compiler-and-export-hooks).
* Put data-synthesis changes in the owning dataset/degradation model, preserving
  `feed_data()` and `data_feed_context()` behavior. Do not fold random data
  preparation into the optimizer transaction. High-frequency target noise uses
  both project NumPy and Torch RNG streams: preserve draw order, conditional
  grayscale sampling, dtype conversion, and the overridable Beta sampler, not
  merely the output distribution. The extracted noise helper does not own or
  reseed a separate generator.
* For SR-specific additions, use the existing generator-forward/auxiliary-loss
  hooks where their contract fits. Preserve unique loss labels and the common
  finite-check/commit boundary. Feature-distillation handling has its own path;
  a forward hook is not automatically used by every configuration.
* Keep shared changes compatible with `AEModel` as well as `SRModel` and its
  degradation subclasses. Public model methods and the normal entry points are
  the caller boundary; internal helpers should remain narrowly owned.
* If changing persisted state or a compiler-facing graph contract, explicitly
  review load compatibility and cache identity. This rewrite does not introduce
  a checkpoint migration or a new plugin API.

## Verification scope

The CPU regressions in [tests.zip](../../tests.zip) cover training merges,
gradient norms, validation, discriminator compilation, compile-cache publication
and reuse, image-save scheduling, and checkpoint recovery. Extract the archive
in the repository root before running the selection below.
Some tests mock compiler/device boundaries; the full SR integration test also
needs datasets and pretrained metric/loss dependencies. Select checks according
to their actual requirements, not just the containing file name.

### November 21 fork compatibility check

Both WSL checkouts ran the same seeded 2× WSR Compact recipe on an RTX 5090
with BF16, compilation disabled, and a DVD degradation on every batch. The
test used four generated 64 × 64 RGB images, batch size one, LQ patches of
32 × 32, one persistent loader worker, and CUDA prefetch. After two updates,
the generator and EMA matched tensor for tensor. The saved training states
matched, including optimizer moments, scheduler, scaler entries, EMA step,
and iteration. Both trainers then resumed the fork's iteration-two checkpoint
in separate experiment directories. The same comparison passed after update
three. All eight runtime registry inventories matched in both checkouts.

The initial comparison also exposed a failure with `persistent_workers: false`
in both checkouts. A forked loader worker inherited the parent's RNG streams
and failed when applying its own seed. The rewrite now distinguishes inherited
streams from streams initialized by the current process. Each replacement
worker starts its project and codec-routing streams from the existing worker
seed policy. A second initialization in the same process still raises.

After the fix, the rewrite completed five updates with one nonpersistent
worker and an enlargement ratio of one on the same four-image WSR setup. It
saved update five in the second epoch, with the EMA step also at five.

These checks cover this single-GPU eager training and checkpoint path. Compiled
throughput, external VOSR teacher engines, multi-GPU behavior, and long-run
convergence require separate evidence.

### Recorded rewrite checks (2026-10-22 UTC)

The bounded CPU checks used the existing WSL Python environment with
`CUDA_VISIBLE_DEVICES=`; no dependency installation or GPU validation was part
of this pass. These results describe the selected checks, not the entire suite.

| Check | Recorded result |
| --- | --- |
| Final integrated selected CPU regressions | 171 passed, 2 skipped, 14 deselected; 2 existing Torch JIT deprecation warnings. |
| Real periodic-save integration hook | 4 passed separately; also included in the final integrated selection above, not an additional four tests to sum into that total. |
| Normal CLI help | `python -B train.py --help` exited successfully with the existing options, including recovery controls. This checks the help/launch boundary, not a training run. |
| Persistent-nonfinite training-loop fixtures | 2 failed before reaching the intended injected failure: the fixture's `train` namespace lacks `warmup_iter`. Both failures reproduced with the baseline `train.py` loaded in memory. Tests were not edited; the intended recovery-path assertions remain unverified by these fixtures. |
| Supplemental nonfinite-loop diagnostic | 2 passed after supplying only `warmup_iter=-1` to the fixture in memory. This reaches the intended failure/cleanup assertions, but is not a pass of the unchanged test fixtures. No test file was modified. |
| Ephemeral baseline differential probes | Selected CPU comparisons matched baseline compiler contracts/events, inference/validation results, distillation values/gradients, degradation outputs and NumPy/Torch RNG states, checkpoint normalization, serialization roundtrips, and expected writer failures. These were in-memory probes, not new committed tests or real CUDA compilation. |
| Documentation links and whitespace | All 31 local guide links resolved; `git diff --check` passed. |

The final selection used the following existing files and filter. Run from the
repository root in the already-installed environment; the recorded run hid CUDA,
limited CPU threads to two, disabled bytecode writing, and disabled pytest's
cache provider. The filter is intentionally explicit: this was not a full-suite
pass. CUDA-dependent OTF checks were skipped, and the full SR integration and
selected supervisor/fixture checks were not included in this passing selection.

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_models/test_training_merge.py \
  tests/test_models/test_sr_model_gradient_norm.py \
  tests/test_utils/test_validation_image_schedule.py \
  tests/test_data/test_realesrgan_otf_crop.py \
  tests/test_data/test_harness_fast_paths.py \
  tests/test_utils/test_training_recovery.py \
  tests/test_models/test_compile_cache.py \
  tests/test_models/test_discriminator_compile.py \
  tests/test_data/test_prefetch_dataloader.py \
  tests/test_models/test_sr_model.py \
  tests/test_utils/test_compile_cache_runtime.py \
  tests/test_otf_rng_stream.py \
  -k 'not normal_command_is_supervised_with_original_options and not invalid_recovery_options_fail_before_launch and not normal_command_end_to_end_recovery_without_a_separate_launcher and not real_subprocess_stops_after_two_no_progress_restarts and not real_training_loop and not test_srmodel' \
  -p no:cacheprovider
```

The documentation build dependencies (`sphinx`, `myst_parser`, and
`sphinx_toolbox`) were absent from the existing environment, so the Sphinx site
was not built. The guide is included in the existing MyST-enabled documentation
toctree, but rendered-site output has not been verified.

### Adversarial follow-up: compatibility and inherited defects

A further CPU-only, baseline/current comparison covered 158 integrated
training/validation/rejection/checkpoint scenarios with 3,214 paired state
snapshots; 32 mocked full-pipeline orchestration flows; and 384 validation
mode/cache/exception cases. Six registries and all five model classes' method
signatures/inheritance matched. Additional probes exercised actual checkpoint
serialization, resume followed by the next update, and atomic-write faults.
These bounded diagnostic probes found no demonstrated rewrite regression in the
shipped paths; they are not extra committed tests to add to the 171-test total,
nor evidence of GPU or long-run equivalence.

The probes also exposed or confirmed baseline defects. They remain **unfixed**
in this compatibility-preserving rewrite, in the following follow-up order:

1. **Recorded-iteration overwrite can mix checkpoint generations.** Fault
   injection while overwriting an already-recorded iteration left valid newer
   network files paired with an older training-state file. Persistent-nonfinite
   receipt validation accepted that mixed set because it does not enforce the
   anchor's recorded file versions. CUDA/native-crash receipt validation rejected
   the same changed set through its version checks. Ordinary saves to increasing
   iteration numbers passed the probes; this finding concerns partial replacement of an
   existing anchor, not every checkpoint save. A separate fix should protect
   active checkpoint generations from partial overwrite and apply consistent
   recorded-set validation to persistent-nonfinite recovery.
2. **End-of-run accounting can overrun the last accepted update.** The loop tests
   `current_iter > total_iters` after incrementing at an accumulation boundary.
   With remaining data, it can process extra partial microbatches after the final
   accepted update and label final validation with an overrun candidate iteration.
   No extra optimizer step occurred in the reproduced case, but forwards can
   still affect buffers or RNG. The outer epoch loop also lacks a total-iteration
   exit. A separate fix should stop processing at the completed-update limit and
   derive final labels from the last completed update.
3. **Validation selection can escape restoration.** If `_select_inference_net()`
   raises, `is_train` has already been set false but the restoration `try/finally`
   has not begun. Both versions retain that state. A separate, lower-priority fix
   should include selection in the protected mode-transition scope.

There is also an internal instrumentation migration: imported dependency names
`calculate_metric`, `calculate_psnr_pt`, `calculate_ssim_pt`, `tensor2img`,
`imwrite`, and `ThreadPoolExecutor` now belong to `traiNNer.models.sr_validation`,
not `traiNNer.models.sr_model`. External monkeypatches targeting those old module
attributes must move. No repository caller depended on them, the model methods
remain available, and no compatibility shim was added. The proposed defect fixes
above were not authorized or applied as part of this pass.

The October readability pass preserved the then-current algorithms and used the
existing tests. Later fork ports also carry the upstream implementations and
their behavioral tests.
Readability, static checks, and bounded CPU regressions do **not** establish GPU
performance, compiled-kernel parity, distributed correctness, long-run training
stability, convergence, or bitwise checkpoint-resume equivalence. Those require
separately authorized, matched runtime validation; they are not implied by the
module layout or the existence of a successful unit test.
