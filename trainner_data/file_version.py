"""Identify the observed version of a local artifact without reading its contents."""
import hashlib
import os


def artifact_version(info: os.stat_result) -> str:
    stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return hashlib.sha256(repr(stamp).encode("ascii")).hexdigest()
