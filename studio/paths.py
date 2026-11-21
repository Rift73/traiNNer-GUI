"""User path resolution shared by the headless tools on Linux and WSL."""
from __future__ import annotations

import os
import re
from pathlib import Path
from trainner_data.file_version import artifact_version as artifact_version


def local_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("Provide a file or folder path.")
    value = value.strip()
    if os.name == "posix" and re.match(r"^[A-Za-z]:[/\\]", value):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    elif os.name == "posix" and value.startswith(("\\\\", "//")):
        parts = value.replace("\\", "/").split("/")
        if len(parts) >= 5 and parts[2].casefold() in ("wsl.localhost", "wsl$"):
            if parts[3].casefold() != os.environ.get("WSL_DISTRO_NAME", "Ubuntu").casefold():
                raise ValueError("The path belongs to a different WSL distribution.")
            value = "/" + "/".join(parts[4:])
        else:
            raise ValueError("Mount the network share in WSL and use its Linux path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


