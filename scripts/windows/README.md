# traiNNer-GUI on Windows

Download **traiNNer-GUI-Windows-x64.zip**, extract the entire folder, and open **traiNNer-GUI.exe** as your normal Windows user. Keep **trainner-gui-source.tar.gz** beside the executable for managed setup. The executable checks that this archive came from the same build.

## First launch

- **Set up dedicated Ubuntu** installs Ubuntu 24.04 into a WSL 2 distribution named **traiNNer-GUI**. The distribution is stored in **%LOCALAPPDATA%\traiNNer-GUI\wsl** under the Windows account that opened the launcher. Studio runs as Linux user **trainer**. The managed Linux checkout is at **/opt/trainner-gui**.
- **Use an existing WSL checkout** opens **Choose the traiNNer-GUI checkout in WSL**. Choose the checkout under the WSL network folder shown in the picker. Setup checks that it is in WSL 2 and can verify the project. Older checkouts require the current project-capable traiNNer-GUI source.
- **Cancel** closes the setup dialog. If you already use the launcher from an established checkout, use **--configure** to choose a workspace or switch environments.

The **Choose a local NTFS workspace folder** picker lets you select or create a folder on a local fixed Windows drive. The same folder can be reopened later.

If WSL needs to be installed, **Install Windows Subsystem for Linux?** asks you to choose **Yes** or **No**. **Yes** requests administrator approval for **wsl.exe --install --no-distribution**. After setup asks you to restart, restart Windows and open **traiNNer-GUI.exe** again as the same Windows user. The project selection is already saved for the resumed setup.

CUDA training in the managed environment requires an NVIDIA Ampere or newer GPU and a Windows driver that supports CUDA 13.4 in WSL. If the GPU check fails, setup offers **Set up the CPU-only workbench?**. Choose **Yes** for the editor and CPU recipes. Choose **No** to see the diagnostic and retry after correcting the problem. Both modes use the managed pinned Python dependencies. The first installation downloads Ubuntu and about 3.6 GiB of Python packages. Allow roughly 20 GiB free for the runtime, downloads and temporary build files.

The pinned Python package downloads total about **3.57 GiB** based on published package sizes. Allow about **20 GiB** for the managed WSL ext4 filesystem, including the installed runtime and cached downloads.

## Updates

Keep launching the EXE from the folder you extracted. The packaged launcher checks the [latest GitHub release](https://github.com/Rift73/traiNNer-GUI/releases/latest) in the background. An offline launch opens the installed version. If you downloaded the original Preview 1, extract this refreshed ZIP over the original launcher folder once to keep your shortcut.

On a managed install, an update prepares the new trainer alongside the current one. Switching versions waits for training and queued work to finish. Matching Python dependencies are reused. Your Windows workspace keeps the same project UUID, recipes, checkpoints and logs. Refresh a connected phone after the switch to load the updated UI. If your original Preview 1 shortcut used `--state-dir`, keep that argument on the replacement shortcut so its queue is checked during migration.

If you chose **Use an existing WSL checkout**, the launcher can update independently. Update that checkout through your normal Git workflow.

## Workspace and setup files

- The Windows workspace contains **.trainner-gui-project.json**. It stores the version and project UUID, for example {"version":1,"project_id":"a12e4400-9392-4f6a-84cb-e245776bd4fe"}. Keep this file with the workspace. Opening an existing marker preserves its UUID.
- Studio receives **TRAINNER_GUI_PROJECT_DIR** as the translated WSL path and **TRAINNER_GUI_PROJECT_ID** as the expected UUID. It checks the mounted folder and marker before using the project. Studio saves recipes and managed run files under **recipes** and **runs** in the project. Service, queue, and Tailscale state live in the WSL ext4 state directory under **projects/<UUID>**.
- **%LOCALAPPDATA%\traiNNer-GUI\desktop.conf** records the active distribution, Linux checkout path, Windows workspace path and UUID. **setup.conf** holds the pending managed setup until it finishes, including the selected project, requested runtime mode and the reboot checkpoint when applicable. The dedicated distribution and prepared runtime persist across restarts. Open the launcher again to resume setup or start Studio.
- **%LOCALAPPDATA%\traiNNer-GUI\setup.log** records setup stages, exit codes and recent Linux and WSL command output. To retry a partial runtime installation, use the same complete ZIP. For service errors, look for **service.log** in the project's WSL state directory.

## Studio lifetime

A Windows project keeps one hidden WSL client connected while its Studio service is running. Reopening the same project reuses it. Closing the browser leaves training available. The client exits when the service stops.

## Launcher commands

In PowerShell, run a GUI executable through **Start-Process -Wait** when an installer or script needs its exit code. The launcher supports these arguments.

- **--url** prints the Studio loopback URL as one line.
- **--phone-status** prints the selected Studio project's existing Tailscale status JSON. Open Studio first to start its configured phone gateway. To query it from the extracted folder, run `.\traiNNer-GUI.exe --phone-status | ConvertFrom-Json` in PowerShell.
- **--configure** opens setup and saves the selection. It exits after setup so a Windows installer can wait for it.
- **--configure --managed --project-dir "D:\Studio Projects\Example" --non-interactive** sets up the managed environment when WSL is ready. Add **--cpu-only** to select the CPU workbench.
- **--configure --distro Ubuntu --root /home/me/traiNNer-GUI --project-dir "D:\Studio Projects\Example" --non-interactive** saves an existing checkout without the folder dialogs.
- **--state-dir /path/on/ext4** sets the Studio base state directory for one launch or status query. Project state uses a **projects/<UUID>** subfolder of that directory.

**--url**, **--phone-status** and **--non-interactive** report failures on stderr. Configuration exits with **0** on success, **3010** when Windows needs a restart, **1223** if setup was cancelled, or a nonzero failure code. Linux provisioning uses exit **4** for a CUDA check failure. In noninteractive mode, a missing WSL installation requires a separate interactive run for Windows approval.

## Build the Windows asset

Run **Build-Desktop.ps1 -Package -Destination <Windows release directory>** with the existing Windows C++ Build Tools, clang-cl, CMake and Ninja. Use Python 3 in the source checkout's WSL distribution, or Python 3.12 through the Windows Python launcher for a Windows checkout. The build invokes **scripts/package-windows.py** to produce the source archive, embeds its hash into the x64 launcher, and assembles **traiNNer-GUI-Windows-x64.zip**.

For a new public version, pass its exact GitHub tag to the package build, for example `-Package -ReleaseTag v0.1.1`. Upload **traiNNer-GUI-Windows-x64.zip** to that release and mark it **Latest** for automatic delivery. Use a newer tag for each future update.

Running **Build-Desktop.ps1** without **-Package** builds the developer executable for an existing checkout. Use **-BuildDirectory** and **-Destination** to keep scratch builds outside the checkout.
