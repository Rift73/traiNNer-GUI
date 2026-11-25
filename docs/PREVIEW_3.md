# Preview 3 accumulation ledger

Snapshot of the Ubuntu dev worktree on 2026-11-25 at 01:35:42 MYT, including local changes over `40ce905d9c92bb0fc56cb86fbb948495514757bf`. The release record below identifies the published build.

## Baseline and inventory before live iteration reporting

The [published Preview 2 release](https://github.com/Rift73/traiNNer-GUI/releases/tag/v0.1.0-preview.2) is `v0.1.0-preview.2` at `8e70c15a77300658306a5676a2c9fd285aba638c`. The clean public checkout and remote tag resolve to that commit. The retained final Windows ZIP matched the published asset digest `a49f176f9ad3bd29956037907c792efe2a4fbaa7179f4ee900be18f937bc84e4`. Its managed tarball matched the tagged allowlist and all 1,273 source files by bytes, including the packaged `trainner-gui` mapping to `scripts/managed-wsl-launcher.sh`. Its internal SHA-256 manifest and the tarball digest embedded in the EXE agreed.

| Comparison | Result |
| --- | --- |
| Managed payload, both directions | 1,273 → 1,274 paths. 1,243 byte-identical, 30 changed in place, one added (`studio/ai_context.py`), zero removed or missing. Of the 30, 23 are workbench or manifest edits, three are older public-preparation differences, and four were updater edits at the time of this inventory. |
| Public tag vs dev files | The tag has 25 paths outside the payload. All occur in dev except the `tests.zip` container. Of 958 dev-only paths, excluding this ledger, 96 occur inside that ZIP, three are new checks, one is the payload addition and 858 are private checkout extras. Of those extras, 857 match the pre-release dev HEAD and one is its older local EXE. |
| Archived checks, both directions | All 96 test, helper and fixture members exist byte-for-byte in dev. Dev adds [queue reorder](../tests/test_studio_queue_reorder.py), [queue resume and extension](../tests/test_studio_queue_resume.py) and the [updater discovery check](../tests/windows/updater-discovery.cpp). |

The inventory covers both allowlists plus Git-visible sources and fixtures. Ignored runtime state and generated build output are outside its scope. The final local release notes are in `research/trainner_preview2_phone_bf0714e2/notes.md` under the Windows user profile. Earlier Preview 2 builds used the same tag name. The older WSL smoke extraction in `scratch/trainner-preview2-yvxOVipp/managed` has 1,267 entries, with 643 hashes different from the final tag. The final baseline includes the recipe editor and ZIP datasets, remote access, phone monitor layout and ETA, RDUAL indexing and speedups, DUAL Light speedups, discriminator additions, official weight downloads, live checkpoint saving and immediate resume with settings.

## Workbench changes in dev

| Feature | Behavior |
| --- | --- |
| Reorder released queue entries | Move waiting entries by drag or Move up/down on desktop and phone. The persisted order controls dispatch. New releases join the tail. A conflicting move refreshes the line. Manually queued recipes that consume another job's checkpoint still require the producer to run first. |
| Queue a stopped run | Add a Studio-managed run to the released line from its newest complete numbered checkpoint. Dispatch starts a linked run with the pinned state, weights, recipe and seed. Repeat requests reuse the open entry. Run and queue views show its status and position. |
| Extend a completed run | Queue a linked continuation from the final full checkpoint for a positive number of extra iterations. The new target uses the completed step, with the saved recipe target as the legacy fallback. It preserves saved optimizer and scheduler state and the original QAT freeze boundaries. Preflight rejects missing checkpoint files, exhausted early stopping and extensions beyond a bounded scheduler's saved horizon. Curriculum phases use their retry workflow. |
| Copy a telemetry path | Copy the selected trainer log or TensorBoard event file. The service supplies the native path and Windows drive or WSL UNC form where available. |
| Copy for AI | Copy a local run summary with the current attempt, progress, available log, recipe and artifact paths, plus continuation lineage when present. It uses the selected Telemetry source or the default source and offers selectable text if clipboard access fails. |

The [Studio API](../studio/API.md) and [UI design notes](../ui/DESIGN.md) record the behavior. The three focused WSL pytest cases passed using checkpoint fixtures and mocked trainer launches. Local QA handoffs on November 24–25 recorded read-only desktop and emulated phone queue inspection, and 12 matching clipboard comparisons on desktop and Chromium phone emulation over Tailscale.

## Live iteration progress

The trainer publishes its confirmed optimizer iteration independently of log frequency. Rank zero writes one integer in a per-attempt shared memory mapping. Accumulation microbatches and rejected updates do not increment the counter. A resumed run starts at the iteration restored from disk. Existing training attempts and external CLI runs can continue to display their logged progress. Losses, ETA, speed and memory retain their own measurement iteration. Desktop and phone share a lightweight progress read for the visible counters and bars. For fast runs the browser steps through confirmed integers between samples. After a reconnect or a gap too large to paint, it shows the latest confirmed value.

An isolated 240-update CPU training run with `print_freq: 1000` produced zero periodic telemetry rows. The Studio HTTP endpoint returned intermediate iterations and reached 240. A separate CPU fixture exercised accumulation and checkpoint resume. On this WSL machine the isolated publisher cost about 23 ns per update, or 69 ns when accounting for deferred GPU outcomes on the host. These are CPU component measurements. End-to-end GPU training throughput was not benchmarked. After the local Studio service reload, the actual MSA continuation reported 12,915 through 12,919 consecutively over its Tailscale URL while the log-based monitor remained at 12,911. A touch-enabled 430 × 932 Chromium browser on the same Tailscale URL rendered every value from 13,524 through 13,530 on the live run page. The physical iPhone Safari UI was not exercised.

## Windows automatic updates

Preview 3 is the first supported starting version for automatic preview-channel updates. Opening the packaged Windows EXE checks for newer releases in the background. Preview installations can receive later previews, including GitHub prereleases, or graduate to stable. Stable installations receive stable releases. Use a higher version tag for every update.

The launcher verifies the published ZIP size and SHA-256. The managed backend stages side by side and switches on a later idle launch. A running trainer or unfinished queue entry, including one on hold, defers the switch. A custom WSL checkout retains its own update workflow. A Preview 3 shortcut launches the updater from the active installation after an upgrade. When moving from Preview 2, open the Preview 3 EXE and point any existing shortcut to it to receive future GitHub prereleases.

The Windows LLVM build and release-selection fixture passed. An isolated copy of the packaged Preview 3 worker contacted the real GitHub feed and correctly found only older releases. With an isolated empty profile and a throwaway older version tag, the actual launcher downloaded the public Preview 2 ZIP, verified and staged it, then applied it on the next ordinary launch. Managed fixtures covered an existing versioned installation, opening a newer EXE manually, busy deferral and idle activation. A separate queue fixture deferred running, pending and held entries and admitted a completed queue. A full managed Windows-to-WSL cutover using an installed training runtime was not exercised.

## Perceptual losses and graceful queue pause

The recipe editor and local trainer now expose six optional losses. The table reports the median time for a complete loss forward and prediction backward on an RTX 5090, using four 192 × 192 RGB crops, channels-last FP32 and `torch.compile(mode="reduce-overhead")`. Each paired measurement alternated the original differentiable implementation and the optimized implementation over five blocks of ten updates, following warm-up. Timings came from synchronized, unprofiled runs.

| Recipe type | Original implementation | Optimized implementation |
| --- | ---: | ---: |
| `ssimulacra2loss` | 1.916 ms | 0.532 ms |
| `butteraugliloss` (p3 pooling) | 2.342 ms | 1.447 ms |
| `vmafloss` (NEG) | 0.878 ms | 0.493 ms |
| `vifloss` | 0.461 ms | 0.403 ms |
| `admloss` | 0.566 ms | 0.496 ms |
| `chromawaveletloss` | — | 0.375 ms |

On the reference image pairs, the largest absolute score differences were below 0.05 points for SSIMULACRA2, 0.00002 for Butteraugli with max pooling and 0.008 points for frame-independent VMAF against libvmaf with matched luma inputs. Finite-gradient checks covered BF16 autocast, an exact match and a distorted crop in the same batch for all six. Nsight Systems located the filtering bottlenecks and Nsight Compute profiled selected kernels. Whole-training throughput and trained model quality remain unmeasured. [Loss options and input conventions](source/loss_reference.md)

**Pause the queue** checkpoints the active queued run at an update boundary and preserves its position. **Resume the queue** resumes that entry from the saved training state. During the local verification, MSA stopped at iteration 16,265 and advanced past it after resuming, with DUNET still next. An isolated check also covered a GUI restart during the pause.

## Recipe formatting and first-use VOSR assets

Structured recipe edits use indented YAML for loss lists and nested objects. The YAML pane can also format a compact draft. VOSR loss and VOSR training targets share a first-use cache for the ten Linux RTX 5090 / TensorRT 11.4.0.106 engines and their plugin. Both local paths are optional. The live desktop GUI was exercised in an isolated Chrome session by opening a compact draft and adding VOSR through the loss picker. A cold cache downloaded all eleven files from Feral, verified their compressed and unpacked hashes, and loaded the Linux plugin from anonymous memory. [VOSR runtime details](VOSR_loss.md)

## TensorBoard comparisons

**Runs → Compare with… → Browse TensorBoard run folder…** compares the full history across the event files in a run folder. Compare against a GUI run or a second run folder. Both sides default to full history, with individual event files available for inspection.

**Axis ranges** lets you set either end of the iteration and value axes on comparison, telemetry and validation score charts. Blank bounds use automatic scaling. Comparison and telemetry read the selected iteration interval from the indexed logs.

## Resumed training charts

Telemetry opens the full checkpoint history, including the original stopped run. Dashed lines mark the recorded resume iterations. The dropdown keeps individual logs and their copyable paths. When a run resumes from an older checkpoint, the combined view drops the superseded points and the individual attempts retain them.

Verified on the local MSA run at desktop and phone viewport sizes. Its three logs contain 129, 33 and 88 generator-loss samples. The combined chart matches all 250 values through iteration 25,000 and marks resumes at 12,911 and 16,265. The GUI service reloaded during the existing DUNET run. Preview 3 packaging must include `studio/lineage_history.py`, now in the managed allowlist.

## Same-run queue continuations

**Add to queue** on a stopped run and **Extend training** on a completed run now create a queue entry and a new attempt in the existing run folder. The run ID, seed, original recipe and checkpoints carry forward. The Recipe tab shows the effective recipe for the selected run’s current attempt. Replay creates a separate run. The sidebar groups older split continuations under the latest run and links the earlier attempts.

Verified with isolated queue fixtures for resume, extension, pause and service restart between launching and recording the attempt. Changed checkpoint files and a superseded attempt both blocked dispatch. The local desktop and phone layouts showed four top-level runs, with the original MSA and DUNET records accessible. The active DUNET attempt retained its ID through the GUI service reload.

## Fresh recipes from previous runs

**Open as recipe** prepares a new training run at iteration zero. It restores the original pretrained weights when copying a managed continuation. **Make fresh recipe** applies the same change to an open draft while keeping edits such as its name and loss stack. **Resume** continues from the newest complete checkpoint.

## Recipe VRAM test and live training speed

- **VRAM test** in the recipe editor runs 20 optimizer updates on the GPU with the draft's training settings, including accumulation. It reports PyTorch's peak allocated and reserved VRAM and sampled device-wide use. The test uses the trainer's exclusive lease, saves no experiment artifacts, and leaves the queue and run history alone.
- The active run overview shows **Live speed (5 s)** beside the logged telemetry. The browser measures completed iterations over elapsed time using its existing progress requests. It starts a new measurement after a resume or reconnect. Stopped runs retain their last logged speed.

## Force stop

Active managed runs have a **Force stop** button. A stopping queue entry also offers it. After confirmation the supervisor terminates that attempt and its workers, and records it as stopped. Resume uses a previously saved training state.

## Faster DUAL validation

DUAL validation now reuses folded convolution weights, fuses neighborhood attention on the RTX 5090, and overlaps image decoding with inference. The CUDA SSIM metric retains double precision. On the 2× DUAL Light DFLIP recipe, validating all 209 Urban100 and Manga109 images fell from 61 seconds to about 8 seconds per warmed pass. All restored tensors and all 418 exported PNGs matched the original in a paired check.

Comparison, telemetry and validation score charts update while open. New dataset scores appear as soon as validation records them, including while TensorBoard buffers its copy. Returning to a phone tab catches up with the run and preserves the chart controls.

Validation images load as you scroll. In the comparison viewer, **Open in Explorer** selects the file shown as Image A or Image B on the Windows training PC. The run header labels its five-second iteration rate **Speed**.

On the run overview, **Power**, **Temperature**, and **VRAM** refresh every five seconds. The WSL service maps the CUDA adapter to its Windows dedicated-memory counter. The desktop and phone gateway share each sample.

## Release preparation

- Keep the public archive instructions, portable benchmark paths and source packaging metadata. These differences predate Preview 2 in dev. Reconcile the dev recipe generator's per-architecture ignore files with the public root ignore rules before assembling the public tree. Include new checks in the public `tests.zip` and new runtime files in the managed package allowlist.
- Existing phone checks used browser emulation. A physical phone and an installed managed-WSL upgrade remain validation limits. The local Preview 3 verification ZIP (SHA-256 `132de04406ef99cd8850307c08b3fbcc79fb42ba041516786badb9e3641a253c`) matched its 1,275 payload paths, including the live iteration UI. That ZIP predates the perceptual losses and graceful queue pause. The new runtime files are listed for the next package build. That earlier verification used Preview 2 as its public baseline.

## Published Preview 3

Published as GitHub’s latest release from commit `4c0f81aaa1d69fb57f671a84a52460c62cc62801`. The Windows x64 ZIP downloaded from the release matched the packaged ZIP, SHA-256 `1c60109f5e70b4430d0b74db988d915e2d03639af89d6604eb8ad97b6a63fe1b`. All 1,302 managed files matched the release source and the embedded manifest. The public test archive matched 102 development files. [Release and Windows download](https://github.com/Rift73/traiNNer-GUI/releases/tag/v0.1.0-preview.3)

The local run monitor and the Tailscale phone layout received three successive shared RTX 5090 samples. The VRAM row and the Windows adapter counter both read 27.4 GB in a paired check. The same DFLIP attempt advanced across the service reload.

## Preview 3 replacement

This build retains version `v0.1.0-preview.3`. To replace the first Preview 3 build, extract the replacement Windows ZIP and open its launcher.

- Butteraugli and SSIMULACRA2 are available under validation metrics. Existing recipes keep their selected metrics.
- DUAL Light and WSR can reuse composed convolution weights across microbatches and combine their gradients at the optimizer step on supported hardware. On an RTX 5090, the 192-pixel Charbonnier pretrain benchmark at batch 1 with 32 accumulations completed 30 optimizer updates in 17.71 seconds for DUAL Light (previously 24.97) and 7.42 seconds for WSR (previously 9.03). WSR used about 30 MiB more peak allocated memory.
- The trainer also includes shared loss and sampling optimizations from the-database/traiNNer-redux, an optional decoded-image cache and opt-in fused SPAN training. The cache budget is per data worker.
- Returning from the TensorBoard run folder picker clears the comparison panel’s edge highlight.
- VGG19 perceptual loss accepts `tensorrt_fp16_backward: true` for the standard five convolution taps. It builds a level-5 engine for the training crop on first use. The loss uses the original FP32 forward pass and FP16 input-gradient convolutions. With the RTX 5090 illustration recipe, batch 4 with 8 accumulations took 35.75 s for 30 updates versus 37.31 s native. Batch 1 with 32 accumulations took 39.15 s versus 39.51 s native. The TensorRT engine uses a non-default CUDA stream to keep the host free to submit work. This option defaults to off.
- RDUAL training accepts `train.rdual_flow_tensorrt: true` to run frozen NeuFlow v2 in FP32 with a level-5 TensorRT engine. The flow encoder reuses features across adjacent pairs, including both directions. The generator's `clip_keep` setting is independent. On an RTX 5090 with four 192×192 clips of 20 frames, guidance took 56.30 ms versus 120.88 ms in PyTorch. With the discriminator updating every step and `clip_keep: none`, twenty full GAN updates took 50.08 s versus 50.90–51.30 s native. Across four restored clips, PSNR against the same source references differed by at most 0.0026 dB. The option defaults to off and requires CUDA, ONNX and TensorRT.
