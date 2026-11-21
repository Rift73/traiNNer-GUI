"""Install the checkout using the live trainer's Python package versions."""

import argparse
import platform
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = "https://download.pytorch.org/whl/nightly/cu134"
LIVE_PAIR = ("torch==2.16.0.dev20261102+cu134", "torchvision==0.30.0.dev20261102+cu134")
CONSTRAINTS = ROOT / "requirements" / "live-runtime.txt"


def installed_pair(torch_requirement: str) -> tuple[str, str] | None:
    try:
        torch = metadata.version("torch")
        vision = metadata.version("torchvision")
    except metadata.PackageNotFoundError:
        return None
    # Ask the venv's own resolver to check the floor offline, including prereleases.
    probe = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "--no-index", "--no-deps",
         "--pre", torch_requirement, f"torch=={torch}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        return None
    return f"torch=={torch}", f"torchvision=={vision}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-torch", action="store_true",
                        help="Update torch and torchvision together from the CUDA 13.4 nightly index.")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.exit(1, "Run install.sh or install.bat to install into the checkout's virtual environment.\n")

    with (ROOT / "pyproject.toml").open("rb") as stream:
        deps = tomllib.load(stream)["project"]["dependencies"]
    torch_requirement = next(req for req in deps if req.startswith("torch>="))
    pip = [sys.executable, "-m", "pip", "install"]
    if args.latest_torch:
        subprocess.run([*pip, "--pre", "--upgrade", "--index-url", INDEX,
                        torch_requirement, "torchvision"], check=True)
        pair = tuple(f"{pkg}=={metadata.version(pkg)}" for pkg in ("torch", "torchvision"))
    else:
        pair = installed_pair(torch_requirement) or LIVE_PAIR
    print("Installing traiNNer-GUI with " + " and ".join(pair), flush=True)
    constraints = ["--constraint", str(CONSTRAINTS)]
    if (pair == LIVE_PAIR and sys.platform == "linux" and platform.machine() == "x86_64"
            and sys.version_info[:2] == (3, 12)):
        constraints.extend(["--constraint", str(CONSTRAINTS.with_name("live-wsl-py312.txt"))])
    subprocess.run([*pip, "--extra-index-url", INDEX, *constraints, *pair, str(ROOT)],
                   check=True)
    # Confirm the native vision extension is loadable with the selected torch.
    subprocess.run([sys.executable, "-c",
                    "import torch, torchvision; torch.ops.torchvision.nms; "
                    "print('Installed torch', torch.__version__, 'and torchvision', torchvision.__version__)"],
                   check=True)


if __name__ == "__main__":
    main()
