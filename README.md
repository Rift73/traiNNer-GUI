# traiNNer-GUI

Train models locally from your browser. Edit YAML recipes, inspect datasets, and monitor training and validation. Choose a template for image or video restoration, classification, segmentation, detection, or tabular regression. You can also import NeoSR recipes.

## Start on Windows

1. Download the [Windows x64 ZIP](https://github.com/Rift73/traiNNer-GUI/releases/latest/download/traiNNer-GUI-Windows-x64.zip), extract it, and open **traiNNer-GUI.exe**.
2. Choose **Set up dedicated Ubuntu** for a new installation. Setup offers to install WSL 2 if needed. If prompted, restart Windows and reopen the EXE. If you already have a working trainer in WSL, choose **Use an existing WSL checkout**.
3. Choose a local NTFS project folder. For a first run with bundled data, open **TabularRegression / Tabular_MLP** and select **Start training**.

Keep using the same EXE. It downloads updates in the background and applies them on an idle launch. If you connected your own WSL checkout, you manage its code updates. The browser runs on Windows and training runs in Linux. The managed CUDA runtime requires an NVIDIA Ampere or newer GPU and a compatible Windows driver. CPU recipes can run without an NVIDIA GPU.

To use your phone, connect Tailscale on both devices, select **Open on phone** in the desktop workbench, and scan the QR code.

## Your files

The Windows project contains your recipes, logs, weights, and resumable training state. Back up the **whole project folder** plus any external datasets or files its recipes reference. Back up earlier Linux and CLI runs in their original locations.

## Linux and WSL checkouts

From the checkout, run `./install.sh`, then `./trainner-gui url`. Open the address it prints. Existing traiNNer YAML recipes and the `train.py` CLI are supported.

[Workbench and Tailscale phone access](docs/source/workbench.md) · [Installation and CLI guide](docs/source/getting_started.md) · [Training options](docs/source/config_reference.md) · [Task providers](docs/source/task_providers.md) · [License and acknowledgements](LICENSE/README.md)

## Acknowledgements

Thanks to the contributors of [traiNNer](https://github.com/victorca25/traiNNer) and [NeoSR](https://github.com/neosr-project/neosr) for their work on super-resolution training. traiNNer-GUI builds on [traiNNer-redux](https://github.com/the-database/traiNNer-redux) and includes optimizer and loss implementations adapted from [NeoSR-dev](https://github.com/korvin2000/neosr-dev).
