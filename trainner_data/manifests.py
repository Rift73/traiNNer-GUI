"""Interpret the trainer's on-disk sample selection without importing its runtime."""

import re
from concurrent.futures import ThreadPoolExecutor
from os import path as osp


_SHAPE = re.compile(r"\s+\(\s*\d+\s*(?:,\s*\d+\s*){1,2}\)(?:\s+\d+)?\s*$")


def manifest_filename(line: str) -> str:
    # A final " (width,height,channels)" is metadata; spaces and dots in the filename are significant.
    return _SHAPE.sub("", line.rstrip())


def lmdb_keys(folder: str) -> list[str]:
    if not folder.endswith(".lmdb"):
        raise ValueError(f"Expected a .lmdb directory, got {folder}.")
    with open(osp.join(folder, "meta_info.txt"), encoding="utf-8") as src:
        return [osp.splitext(manifest_filename(line))[0] for line in src if line.strip()]


def paths_from_lmdb(folders: list[str]) -> list[str]:
    with ThreadPoolExecutor() as executor:
        return [key for keys in executor.map(lmdb_keys, folders) for key in keys]


def paired_paths_from_lmdb(
    folders: tuple[list[str], list[str]], keys: tuple[str, str], *, include_storage: bool = False
) -> list[dict[str, str]]:
    """Generate paired paths from LMDB files for multiple folder pairs, optimized for large datasets.

    Args:
        folders (tuple[list[str], list[str]]): Two lists of folder paths.
            The first list is for input folders, and the second is for GT folders.
        keys (tuple[list[str], list[str]]): Two lists of keys identifying folders.
            The first list corresponds to input folders, and the second to GT folders.
            Note that these keys are different from LMDB keys.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing paired LMDB paths.
    """
    input_folders, gt_folders = folders
    input_key, gt_key = keys
    paired_paths = []

    assert len(input_folders) == len(gt_folders), (
        "The lengths of input_folders and gt_folders must be the same."
    )

    def process_folder_pair(
        input_folder: str, gt_folder: str
    ) -> list[dict[str, str]]:
        if not (input_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
            raise ValueError(
                f"{input_key} folder and {gt_key} folder should both be in LMDB formats. "
                f"Received {input_key}: {input_folder}; {gt_key}: {gt_folder}"
            )

        input_lmdb_keys = set(lmdb_keys(input_folder))
        gt_lmdb_keys = set(lmdb_keys(gt_folder))

        if input_lmdb_keys != gt_lmdb_keys:
            raise ValueError(
                f"Keys in {input_key}_folder and {gt_key}_folder are different."
            )

        return [
            {f"{input_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key,
             **({f"{input_key}_db": input_folder, f"{gt_key}_db": gt_folder} if include_storage else {})}
            for lmdb_key in sorted(input_lmdb_keys)
        ]

    with ThreadPoolExecutor() as executor:
        future_paths = executor.map(
            process_folder_pair,
            input_folders,
            gt_folders,
        )

        for paths in future_paths:
            paired_paths.extend(paths)

    return paired_paths


def paired_manifest_lq_path(
    input_folder: str, gt_name: str, filename_tmpl: str, *, flat_fallback: bool = True
) -> str:
    """Resolve a GT manifest entry using the trainer's nested then legacy flat LQ lookup."""
    basename, ext = osp.splitext(osp.basename(gt_name))
    input_name = f"{filename_tmpl.format(basename)}{ext}"
    input_path = osp.join(input_folder, osp.dirname(gt_name), input_name)
    if flat_fallback and osp.dirname(gt_name) and not osp.isfile(input_path):
        # Earlier metadata files could name nested GTs paired to a flat LQ directory.
        flat = osp.join(input_folder, input_name)
        if osp.isfile(flat):
            input_path = flat
    return input_path


def paired_paths_from_meta_info_file(
    folders: tuple[list[str], list[str]],
    keys: tuple[str, str],
    meta_info_file: str,
    filename_tmpl: str,
) -> list[dict[str, str]]:
    """Generate paired paths from a meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by white space.

    Example of a meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (tuple[list[str], list[str]]): Two lists of folder paths.
            The first list is for input folders, and the second is for GT folders.
        keys (tuple[list[str], list[str]]): Two lists of keys identifying folders.
            The first list corresponds to input folders, and the second to GT folders.
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folders.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing paired paths.
    """
    input_folders, gt_folders = folders
    input_key, gt_key = keys
    paired_paths = []

    assert len(input_folders) == len(gt_folders), (
        "The lengths of input_folders and gt_folders must be the same."
    )

    with open(meta_info_file) as fin:
        gt_names = [manifest_filename(line) for line in fin if line.strip()]

    def process_paths(
        input_folder: str, gt_folder: str
    ) -> list[dict[str, str]]:
        paths = []
        for gt_name in gt_names:
            input_path = paired_manifest_lq_path(input_folder, gt_name, filename_tmpl)
            gt_path = osp.join(gt_folder, gt_name)
            paths.append({f"{input_key}_path": input_path, f"{gt_key}_path": gt_path})
        return paths

    with ThreadPoolExecutor() as executor:
        results = executor.map(
            process_paths, input_folders, gt_folders
        )

    for res in results:
        paired_paths.extend(res)

    return paired_paths
