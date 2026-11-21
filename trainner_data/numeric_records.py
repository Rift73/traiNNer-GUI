"""Resolve the ordered numeric columns shared by CSV readers and recipe inspection."""

import csv
from itertools import chain
from pathlib import Path
from typing import TextIO


def selected_columns(stream: TextIO, path: Path, columns: list[str]) -> list[int]:
    try:
        header = next(csv.reader(stream, strict=True), [])
    except csv.Error as exc:
        raise ValueError(f"{path} has an invalid CSV header: {exc}") from exc
    if not header or len(set(header)) != len(header):
        raise ValueError(f"{path} needs a CSV header with distinct column names.")
    positions = {name: index for index, name in enumerate(header)}
    missing = [name for name in columns if name not in positions]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}.")
    return [positions[name] for name in columns]


def read_values(
    stream: TextIO, path: Path, positions: list[int], max_rows: int | None = None
):
    """Read the next numeric records with the same conversion for audit and training."""
    import numpy as np

    first = next((line for line in stream if line.strip()), None)
    if first is None:
        return np.empty((0, len(positions)), dtype=np.float32)
    try:
        values = np.loadtxt(
            chain((first,), stream),
            delimiter=",",
            dtype=np.float32,
            ndmin=2,
            comments=None,
            quotechar='"',
            usecols=positions,
            max_rows=max_rows,
        )
    except ValueError as exc:
        raise ValueError(f"Could not read numeric columns from {path}: {exc}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{path} needs finite float32 values in each selected column.")
    return np.ascontiguousarray(values)
