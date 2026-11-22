# Using the workbench

## First run

1. Open a training template and save a named recipe. Choose the task and network, then select the data. Image restoration accepts prepared pairs or on-the-fly degradation. Recurrent restoration uses sequences. Classification, semantic segmentation and COCO detection use their respective task settings and dataset formats.
2. Check the recipe and inspect the data. The paired-data inspector checks names and geometry; the alignment tool measures possible displacement in a selected pair. Whole-image tasks can inspect class folders, segmentation labels or COCO annotations. Check the reported coverage when using a sampled audit.
3. To capture the prepared training pair after augmentation, set `logger.training_sample_steps: [1]`. The resulting raw tensors preserve their dtype and range. Start the run or place it in the queue. The run page shows its log, measured validation scores and available files. Numbered checkpoints support resuming. Studies prepare controlled variants with matched seeds. Curricula connect successive stages using full training state or selected weights.
4. Select a completed checkpoint for prediction or holdout evaluation. Evaluation uses the CPU by default. An explicit CUDA evaluation reserves the managed training slot while it runs. A recorded paired comparison checks source identities and metric definitions. For offline evaluations, the CLI accepts `job:ID` as either side of `compare`. Autoencoder predictions reconstruct the observed image. Temporal models predict the center of each complete window from a folder of numbered frames; different scene prefixes stay separate.

In **Options → Training**, open the arrow beside `optim_g.type` to browse the installed optimizers, or type a name directly. After installing a trainer extension, use **Read the lists again** in the trainer environment dialog to refresh the choices.

For a first run without an image dataset, open **TabularRegression / Tabular_MLP**. Two small CSV files ship with the template. It learns a rectangle's perimeter from its width and height, evaluates on four different rectangles, and keeps the weights with the lowest validation RMSE. To inspect the columns, ranges and overlap between splits, select **Runs → Analysis jobs → New analysis job → Data audit** with the recipe open. Prediction accepts a CSV with just the feature columns. The JSON result names the targets in network output order.

After installing a local architecture or objective, run `./trainner-gui catalog --refresh` to discover its options. CLI commands also expose examples of the data and checkpoint workflows:

```sh
./trainner-gui split-data recipe.yml --validation-fraction 0.1 --group-by scene
./trainner-gui extract-patches recipe.yml --split train --crop-size 96 --stride 80
./trainner-gui convert-weights checkpoint.pth --payload params_ema
./trainner-gui frontier RUN_ID --dataset Set5 --x psnr --y lpips --available-only
./trainner-gui average-checkpoints RUN_ID net_g 20000 25000 30000
```

Split the sources first when train and validation will use disjoint scenes. Extract patches separately from each resulting dataset. Patch sizes and strides use low-resolution pixels.

For on-the-fly still-image training with Windows data in WSL, `RealESRGANDataset` accepts a ZIP of the original encoded images. Choose **Store** when creating the archive, then set `io_backend: {type: zip}` and point `dataroot_gt` at the archive in the recipe's YAML editor. The trainer reads members directly from the Windows drive and verifies their CRCs. Ordinary image folders also work without preparation. Create the archive outside the source folder, with paths relative to the image root. Treat it as a snapshot for the duration of training. After changing the archive, reopen the dataset so all workers use the same version. Automatic discovery uses the same relative paths and ordering as a loose folder. Use a manifest to select specific files or to repeat them in a particular order.

For large still-image holdouts, `val.tile_mode: context` stitches tile interiors on the CPU. Select `val.tile_size` and `val.tile_overlap` in input pixels to control the window and its context. Imported NeoSR recipes with tiling enabled use this policy. A checkpoint evaluation records the tile settings with its scores.

When evaluating a model that predicts only Y or UV, choose where the other planes come from. The existing paired-validation default supplies them from the target. Set `val.complementary_planes: input_bicubic` to use the input for those planes and produce RGB with the same policy in validation, prediction and ONNX export.

The normal trainer CLI, YAML recipes and existing checkpoints remain available. Training assets and output files are local to this machine. With a Windows project selected, editable recipes and run outputs live on that drive. For traditional checkouts, the workbench stores recipes and managed run outputs with its service state under `$XDG_STATE_HOME/traiNNer-GUI` (by default `~/.local/state/traiNNer-GUI`) in a subdirectory for this checkout. To use a different location, supply `--state-dir` before the command. To recover a Windows project after reinstalling the backend, select the same folder in Windows setup and open its run in **Runs**. Resume from a complete numbered checkpoint. Keep referenced datasets and pretrained files available at the paths in the recipe. Stop training before moving a project or opening it through another backend. The training queue and analysis history belong to the Linux service. Closing the browser leaves training under its supervisor. Use Run controls to stop a training job.

## Open on a phone with Tailscale

Open **traiNNer-GUI.exe** and choose **Open on phone** on the desktop. On your phone, connect Tailscale with the same account. Scan the QR code with your camera to open the workbench in Safari or Chrome, or use **Copy link**. You can bookmark the address. Keep the desktop workbench running while you use the phone.

If the PC is disconnected, connect Tailscale from the Windows system tray and leave the phone dialog open while it reconnects. If phone access was turned off for this workspace, enable it in the dialog. For a command-line status check, run `.\traiNNer-GUI.exe --phone-status` from the extracted Windows folder or `./trainner-gui tailscale status` in your Linux checkout.

The phone uses the PC's training service. Its file picker browses the trainer's folders, including mounted Windows drives. Tap a folder once to open it, then choose **Use this folder** to select it. For a checkpoint, tap the filename and confirm. You can also paste a Windows or WSL path and tap **Go**. Back returns to the folder you last visited, and Up opens the parent. The desktop browser continues to use its ordinary localhost URL.

- `./trainner-gui tailscale on` starts or retries phone access and prints its address. Use this after connecting Tailscale if it was offline at launch.
- `./trainner-gui tailscale off` closes phone access and remembers that choice. The desktop trainer keeps running.
- The regular checkout enables phone access automatically. For a separate `--state-dir`, enable phone access once from the desktop dialog or with `tailscale on`.

The PC needs a signed-in Tailscale installation, tailnet HTTPS enabled, and a tailnet access rule permitting the returned port. The integration uses a private foreground **Tailscale Serve** route and accepts the PC owner's Tailscale identity. The gateway closes when the Studio service exits. Existing Serve routes keep their ports. [Tailscale Serve documentation](https://tailscale.com/docs/features/tailscale-serve)

## Code and formats

- **Training and evaluation.** `train.py` and `test.py` use PyTorch networks, optimizer state and existing native kernels. Task providers can supply their own model factory, dataset, settings and metrics. See the [rewrite maintainer's guide](python_rewrite.md).
- **Native tools.** `native/studio/` contains the Windows launcher, dataset readers and inspectors, alignment measurement, scalar-history indexing, source fingerprints and weight comparison. [Dataset formats and selection](../../native/studio/data/README.md).
- **Local service.** `studio/` owns recipe editing, process supervision, queues, CPU analysis and the local HTTP API. [API](../../studio/API.md) · [Studies](../../studio/STUDIES.md) · [Curricula](../../studio/CURRICULA.md).
- **NeoSR interchange.** Importing a NeoSR training TOML produces an editable YAML recipe. Unmapped values stay in the draft for review before launch. [Original traiNNer documentation](https://trainner-redux.readthedocs.io/en/latest/index.html).

