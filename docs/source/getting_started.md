## Getting Started

### Initial setup

On Windows, extract the desktop release and open `traiNNer-GUI.exe`. The launcher connects to the configured WSL 2 trainer. First-time setup can provision a dedicated Linux environment or link an existing checkout. Choose a local Windows project folder for editable recipes and training outputs. Keep that folder when replacing or moving the runtime.

The commands below run inside Linux. A source installation uses Python 3.12. Run `./install.sh` in the checkout, then `./trainner-gui url` to open the desktop workbench.

The installer creates or reuses the checkout's `venv`. A fresh install uses the live trainer's CUDA 13.4 builds, PyTorch `2.16.0.dev20261102+cu134` and torchvision `0.30.0.dev20261102+cu134`. The PyTorch minimum is the 2.14 series, including its development builds. An existing pair meeting the floor is retained if its dependencies resolve together. To select current nightly builds instead, run `./install.sh --latest-torch`.

The other direct runtime dependencies use the versions in `requirements/live-runtime.txt`. On x86-64 WSL/Linux with Python 3.12 and the reference pair, the installer also matches the live transitive runtime dependencies, including Triton and the CUDA libraries. Other PyTorch builds resolve their own platform dependencies. To install the optional ONNX packages at the recorded versions, activate the venv and run `python -m pip install -c requirements/live-runtime.txt ".[onnx]"`.

### Training a Model

#### Do a quick test run
The tabular regression example includes a tiny dataset and runs on the CPU. From the checkout:

```sh
venv/bin/python train.py -opt options/_templates/train/TabularRegression/Tabular_MLP.yml
```

For image restoration, choose an architecture template and supply your own paired images or on-the-fly degradation dataset.

#### Set up config file
1. Copy an architecture template from `options/_templates/train/` to a writable YAML file. For example, copy `options/_templates/train/SPAN/SPAN_fidelity.yml` to `span.yml` in the checkout root.
2. At the top of the file, set the `name` to the name of the model you want to train. Give it a unique name so you can differentiate it from other training runs.
3. Set the scale depending on what scale you want to train the model on. 2x doubles the width and height of the image, for example. Not all architectures support all scales. Supported scales appear next to the scale in a comment, so `# 2, 4` means the architecture only supports a scale of 2 or 4.
4. Set the paths to your dataset HR and LR images, at `dataroot_gt` and `dataroot_lq` under the `train:` section. The HR images and LR images should match in number of images and filenames. For each matching LR/HR pair, the image resolutions should work with the selected scale, so if a scale of 2 is selected then each HR must be 2x the resolution of its matching LR image.
5. If you want to enable validation during training, set `val_enabled` to `true` and set the paths to your validation HR and LR images, at `dataroot_gt` and `dataroot_lq` under the `val` section.
6. If you want to use a pretrain model, set the path of the pretrain model at `pretrain_network_g` and remove the `#` to uncomment that line.

#### Run command to start training

##### Single GPU

Run the following command to start training. Change `./recipe.yml` to point to the config file you set up in the previous step.
```
source venv/bin/activate
python train.py --auto_resume -opt ./recipe.yml
```

For example, to train with the SPAN config:
```
source venv/bin/activate
python train.py --auto_resume -opt ./span.yml
```

To pause training, press `ctrl+C` or close the command window. To resume training, run the same command that was used to start training. The `--auto_resume` flag will resume training from when it was paused.

On single-rank Linux/WSL CLI launches, a lightweight supervisor also handles persistent nonfinite generator losses/gradients. After 20 consecutive rejected attempts it reloads the latest complete regular checkpoint, without saving the failing weights. Recovery preserves the configured seed and uses the configured limit for restarts without a newer checkpoint. There must be a complete checkpoint to recover from, periodic saving must be enabled, and recovery requires full resume with `restart_scheduler: false`. It is not an exact replay of the interrupted data/augmentation stream. Ordinary exceptions (including OOM) are not automatically retried.

Use `--no-auto-recover` to opt out, or `--recovery-max-restarts N` and `--recovery-delay SECONDS` to adjust the bounded policy. Distributed training and native Windows do not use this supervisor. In the synchronous guard, rejected nonfinite generator gradients never advance the optimizer, EMA, scheduler or accepted iteration, but an enabled GradScaler retains overflow backoff so it can recover on a later attempt.

Compiled-function caches are retained across compatible resumes. EMA initialization re-primes architecture-owned geometry caches before the first compiled forward. Cache hits still depend on graph/runtime/shape compatibility; changing the model or configuration may require recompilation.

OTF training uses a centered mirror-padded work canvas for undersized sources. The final aligned LR/GT crop is centered on axes where any sample in the batch needed padding; other axes remain random. This differs from the old black-padding recipe. The zero-range Gaussian/Poisson fast paths preserve their output clamp/round operations but intentionally skip unused random draws. Consequently, the same seed does not imply the same training trajectory across the old and new harness versions. Keep `manual_seed` and `deterministic` explicit when comparing runs, and record the harness version as well as the YAML.

##### Multi GPU

Run the following command to start training with multiple GPUs. Change `--nproc-per-node 4` to the number of GPUs you want to use for training.

```
source venv/bin/activate
torchrun --nproc-per-node 4 train.py --launcher pytorch --auto_resume -opt ./recipe.yml
```

#### Test models

Models are saved in the `safetensors` format to `experiments/<name>/models`, where `name` is whatever was used in the config file. [chaiNNer](https://github.com/chaiNNer-org/chaiNNer) can be used to run most models. If you want to run the model on images during training to monitor the progress of the model, set up validation in the config file, and find the validation results in `experiments/<name>/visualization`.

The test script can also be used to test trained models, which is required to test models with architectures not yet supported by chaiNNer. For example, to test SPANPlus model, open the config file at `./options/test/SPANPlus/SPANPlus.yml`, and update the following:
1. Edit the `dataroot_lq` option to point to a folder that contains the images you want to run the model on.
2. Make sure the options under `network_g` match the options under `network_g` in the training config file that you used. For example, if you trained `SPANPlus_STS`, then set the type to `SPANPlus_STS` under `network_g` in the test config file as well.
3. Update `pretrain_network_g` to point to the path of the model you want to test.

Then run this command to run the model on the images as specified in the config file:
```
source venv/bin/activate
python test.py -opt ./options/test/SPANPlus/SPANPlus.yml
```

### Convert models to ONNX

If you want to convert your PyTorch models to ONNX format, you can use the `convert_to_onnx.py` script to do so. First install the additional dependencies for ONNX:
```
source venv/bin/activate
python -m pip install -c requirements/live-runtime.txt ".[onnx]"
```

Then open a config file corresponding to the architecture of the model you trained. For example, if you trained `SPANPlus`, open the config file at `./options/onnx/SPANPlus/SPANPlus.yml`, and update the following:
1. Set the `name` to the name of your model, the ONNX filename will include this name.
2. Make sure the setting under `network_g` match the settings you used to train your model.
3. Set `pretrain_network_g` to point to the path of your `.safetensors` or `.pth` model that you want to convert.
4. Set the options in the `onnx` section of the config file as needed.

Then run this command to do the conversion (make sure the path points to the `.yml` file you edited):

```
source venv/bin/activate
python convert_to_onnx.py -opt ./options/onnx/SPANPlus/SPANPlus.yml
```

The converted onnx files will be saved to the `onnx` directory.
