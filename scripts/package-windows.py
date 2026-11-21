#!/usr/bin/env python3
"""Stage the allowlisted Windows release from the current checkout."""

import argparse
import gzip
import hashlib
import io
import re
import struct
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = "trainner-gui-source.tar.gz"
LAUNCHER_NAME = "traiNNer-GUI.exe"
MANIFEST = ".trainner-gui-payload.sha256"
ALLOWLIST = ROOT / "scripts/package-windows-files.txt"
BUNDLE_NOTICES = {
    "scripts/windows/README.md": "README-Windows.md",
    "scripts/package-windows-weight-notices.txt": "THIRD-PARTY-WEIGHTS.txt",
}
EXECUTABLES = {
    "install.sh",
    "scripts/provision-wsl.sh",
    "traiNNer/csrc/webp_reconstruct/build.sh",
    "trainner-gui",
}
ARCHIVE_PATH = re.compile(r"[A-Za-z0-9_+-][A-Za-z0-9_.+-]*(?:/[A-Za-z0-9_+-][A-Za-z0-9_.+-]*)*")


def payload_files() -> list[str]:
    files = [
        line for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if files != sorted(set(files)):
        raise ValueError("The Linux payload allowlist must be sorted and unique.")
    return files


def add_file(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o755 if name in EXECUTABLES else 0o644
    archive.addfile(member, io.BytesIO(data))


def build_source(output: Path) -> None:
    if output.name != SOURCE_NAME:
        raise ValueError(f"The host expects the source archive named {SOURCE_NAME}.")
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    records = []
                    for name in payload_files():
                        if not ARCHIVE_PATH.fullmatch(name):
                            raise ValueError(f"Invalid archive path {name!r}.")
                        # Keep the checkout launcher intact. The managed entry point
                        # sets the isolated interpreter, toolkit and recorded mode.
                        path = ROOT / (
                            "scripts/managed-wsl-launcher.sh" if name == "trainner-gui" else name
                        )
                        if not path.is_file() or path.resolve() != path:
                            raise ValueError(f"Missing or redirected payload file {path.relative_to(ROOT)}.")
                        data = path.read_bytes()
                        add_file(archive, name, data)
                        records.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
                    add_file(archive, MANIFEST, "".join(records).encode("ascii"))
    except BaseException:
        if created:
            output.unlink()
        raise
    with output.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    print(digest)


def validate_launcher(launcher: Path, digest: str) -> None:
    data = launcher.read_bytes()
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError("A Windows PE launcher built from this checkout is required.")
    pe = struct.unpack_from("<I", data, 60)[0]
    if data[pe : pe + 6] != b"PE\0\0\x64\x86":
        raise ValueError("The Windows launcher must target x86-64.")
    if digest.encode("ascii") not in data and digest.encode("utf-16-le") not in data:
        raise ValueError("Rebuild the Windows launcher with the SHA-256 of this source archive.")


def build_bundle(folder: Path, output: Path) -> None:
    source = folder / SOURCE_NAME
    launcher = folder / LAUNCHER_NAME
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    validate_launcher(launcher, digest)
    with tarfile.open(source, "r:gz") as tar:
        names = tar.getnames()
        if names != [*payload_files(), MANIFEST] or any(not item.isfile() for item in tar):
            raise ValueError("The source archive members differ from the Linux payload allowlist.")
        notices = {}
        for path, name in BUNDLE_NOTICES.items():
            member = tar.extractfile(path)
            if member is None:
                raise ValueError(f"The source archive is missing {path}.")
            with member:
                notices[name] = member.read()
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w") as archive:
                archive.write(source, SOURCE_NAME)
                archive.write(launcher, LAUNCHER_NAME, compress_type=zipfile.ZIP_DEFLATED)
                for name, data in notices.items():
                    archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    except BaseException:
        if created:
            output.unlink()
        raise
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source", help="Build the Linux tarball and print its SHA-256.")
    source.add_argument("archive", type=Path)
    bundle = commands.add_parser("bundle", help="ZIP the matched tarball, launcher and setup guide.")
    bundle.add_argument("staging_directory", type=Path)
    bundle.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "source":
            build_source(args.archive)
        else:
            build_bundle(args.staging_directory, args.archive)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
