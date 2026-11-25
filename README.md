# traiNNer-GUI

Train models locally from your browser. Edit YAML recipes, inspect datasets, and monitor training and validation. Choose a template for image or video restoration, classification, segmentation, detection, or tabular regression. You can also import NeoSR recipes.

## Training performance

- **CUDA Graphs from Windows.** Training runs in WSL 2. On supported NVIDIA workloads, PyTorch’s `reduce-overhead` compile mode replays GPU work with CUDA Graphs to cut CPU launch overhead. `max-autotune` also benchmarks kernels during the initial compile to find faster ones for the rest of training. [Compile options](docs/source/config_reference.md#compile_mode)
- **Keep image datasets on Windows.** In 300-update DiverSeg-IP benchmarks, the original loader read loose Windows files at **274 images/s** in the first run and **1,212 images/s** in a later benchmark. The optimized loader reached **1,468 images/s** from the same Windows folder. In a separate full-epoch comparison (523,392 images), a Windows ZIP created with **Store** compression sustained **1,713 images/s**, compared with **1,620 images/s** from loose files on WSL ext4. All runs used 2× WSR Compact on an RTX 5090, batch 32, with 128 px LQ crops. [Windows dataset setup](docs/source/workbench.md#first-run)
- **Faster on-the-fly degradation.** The loader reads images ahead, and the trainer prepares upcoming batches while the GPU trains on the current one. With the revamped JPEG, WebP and video-codec paths, a WSR Compact batch of 32 went from **124 ms per training update to 77 ms with real video codecs**, or **70 ms with the default 50% video-codec approximation**. [Degradation options](docs/source/config_reference.md#compress_video_approx_prob-and-compress_webp_approx)

## Start on Windows

1. Download the [Windows x64 ZIP](https://github.com/Rift73/traiNNer-GUI/releases/latest/download/traiNNer-GUI-Windows-x64.zip), extract it, and open **traiNNer-GUI.exe**.
2. Choose **Set up dedicated Ubuntu** for a new installation. Setup offers to install WSL 2 if needed. If prompted, restart Windows and reopen the EXE. If you already have a working trainer in WSL, choose **Use an existing WSL checkout**.
3. Choose a local NTFS project folder. For a first run with bundled data, open **TabularRegression / Tabular_MLP** and select **Start training**.

Starting with Preview 3, keep opening the same EXE to receive newer previews and stable releases automatically. It downloads updates in the background and applies them on an idle launch. If you connected your own WSL checkout, you manage its code updates. The browser runs on Windows and training runs in Linux. The managed CUDA runtime requires an NVIDIA Ampere or newer GPU and a compatible Windows driver. CPU recipes can run without an NVIDIA GPU.

To use your phone, connect Tailscale on both devices, select **Open on phone** in the desktop workbench, and scan the QR code.

## Your files

The Windows project contains your recipes, logs, weights, and resumable training state. Back up the **whole project folder** plus any external datasets or files its recipes reference. Back up earlier Linux and CLI runs in their original locations.

## Train on a rented Linux GPU

Run these commands in the server’s SSH terminal.

```sh
git clone --depth 1 https://github.com/Rift73/traiNNer-GUI.git
cd traiNNer-GUI
./trainner-gui remote
```

Connect your desktop and phone to Tailscale with the same account. Setup prints private links to both views of the same trainer. On later visits, run `./trainner-gui remote` from the checkout. [Remote setup and Colab requirements](docs/source/workbench.md#train-on-a-remote-linux-server)

## Linux and WSL checkouts

From the checkout, run `./install.sh`, then `./trainner-gui url`. Open the address it prints. Existing traiNNer YAML recipes and the `train.py` CLI are supported.

[Workbench and Tailscale phone access](docs/source/workbench.md) · [Installation and CLI guide](docs/source/getting_started.md) · [Training options](docs/source/config_reference.md) · [Task providers](docs/source/task_providers.md) · [License and acknowledgements](LICENSE/README.md)

## Acknowledgements

Thanks to the contributors of [traiNNer](https://github.com/victorca25/traiNNer) and [NeoSR](https://github.com/neosr-project/neosr) for their work on super-resolution training. traiNNer-GUI builds on [traiNNer-redux](https://github.com/the-database/traiNNer-redux) and includes optimizer and loss implementations adapted from [NeoSR-dev](https://github.com/korvin2000/neosr-dev).
