from concurrent.futures import ThreadPoolExecutor
from os import path as osp

import numpy as np

from traiNNer.utils import get_root_logger, scandir
from trainner_data.manifests import (
    paired_paths_from_lmdb as paired_paths_from_lmdb,
    paired_paths_from_meta_info_file as paired_paths_from_meta_info_file,
    paths_from_lmdb as paths_from_lmdb,
)


def check_missing_paths(missing_from_paths: set[str], key: str, folder: str) -> None:
    if len(missing_from_paths) == 0:
        return

    missing_subset = sorted(missing_from_paths)[:10]
    raise ValueError(
        f"{len(missing_from_paths)} files are missing from {key}_paths ({folder}). The first few missing files are:\n"
        + "\n".join(missing_subset)
    )


def generate_frame_indices(
    crt_idx: int, max_frame_num: int, num_frames: int, padding: str = "reflection"
) -> list[int]:
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, "num_frames should be an odd number."
    assert padding in (
        "replicate",
        "reflection",
        "reflection_circle",
        "circle",
    ), f"Wrong padding mode: {padding}."

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == "replicate":
                pad_idx = 0
            elif padding == "reflection":
                pad_idx = -i
            elif padding == "reflection_circle":
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == "replicate":
                pad_idx = max_frame_num
            elif padding == "reflection":
                pad_idx = max_frame_num * 2 - i
            elif padding == "reflection_circle":
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_folder(
    folders: tuple[list[str], list[str]],
    keys: tuple[str, str],
    filename_tmpl: str,
) -> list[dict[str, str]]:
    """Generate paired paths from multiple folders.

    Args:
        folders (tuple[list[str], list[str]]): Two lists of folder paths.
            The first list is for input folders, and the second is for GT folders.
        keys (tuple[list[str], list[str]]): Two lists of keys identifying folders.
            The first list corresponds to input folders, and the second to GT folders.
        filename_tmpl (str): Template for each filename. Note that the template
            excludes the file extension. Usually, the filename_tmpl is for files
            in the input folders.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing paired paths.
    """
    input_folders, gt_folders = folders
    input_key, gt_key = keys
    paired_paths = []

    if len(input_folders) != len(gt_folders):
        raise ValueError(
            f"Paired data needs equally many {input_key} and {gt_key} roots. "
            f"Got {len(input_folders)} and {len(gt_folders)}."
        )

    def warn_same_roots(input_folder: str, gt_folder: str) -> None:
        if input_folder == gt_folder:
            get_root_logger().warning(
                "%s and %s datasets have the same path, this may be unintentional: %s",
                input_key,
                gt_key,
                gt_folder,
            )

    def native_default_pairs() -> list[dict[str, str]] | None:
        import subprocess
        import sys
        from pathlib import Path

        if sys.platform != "linux" or sys.getfilesystemencoding() != "utf-8":
            return None
        inspector = (
            Path(__file__).resolve().parents[2]
            / "native"
            / "studio"
            / "data"
            / "build"
            / "clang"
            / "pair-folders"
        )
        if not inspector.is_file():
            return None

        roots = [
            path
            for pair in zip(gt_folders, input_folders, strict=True)
            for path in pair
        ]
        try:
            completed = subprocess.run(
                [str(inspector), "--train-pairs", *roots],
                check=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )
            # In UTF-8 the byte order agrees with Python's string sort. Paths
            # containing undecodable bytes use the ordinary Python selection.
            fields = completed.stdout.decode("utf-8").removesuffix("\0").split("\0")
        except (OSError, subprocess.CalledProcessError, UnicodeError):
            return None
        if not completed.stdout:
            return []
        if not completed.stdout.endswith(b"\0") or len(fields) % 4:
            return None

        result = []
        iterator = iter(fields)
        for tag, group, gt_name, input_name in zip(
            iterator, iterator, iterator, iterator, strict=True
        ):
            if tag != "pair" or not group.isascii() or not group.isdecimal():
                return None
            index = int(group) - 1
            if not 0 <= index < len(gt_folders):
                return None
            result.append(
                {
                    f"{input_key}_path": osp.join(input_folders[index], input_name),
                    f"{gt_key}_path": osp.join(gt_folders[index], gt_name),
                }
            )
        return result

    if filename_tmpl == "{}" and gt_folders:
        native_pairs = native_default_pairs()
        if native_pairs is not None:
            for input_folder, gt_folder in zip(input_folders, gt_folders, strict=True):
                warn_same_roots(input_folder, gt_folder)
            return native_pairs

    def scan_folder(folder: str) -> list[str]:
        return list(scandir(folder, recursive=True))

    with ThreadPoolExecutor() as executor:
        folder_scans = list(executor.map(scan_folder, input_folders + gt_folders))

    input_scans = folder_scans[: len(input_folders)]
    gt_scans = folder_scans[len(input_folders) :]

    for input_folder, gt_folder, input_names, gt_names in zip(
        input_folders,
        gt_folders,
        input_scans,
        gt_scans,
        strict=True,
    ):
        warn_same_roots(input_folder, gt_folder)

        def by_stem(names: list[str], root: str) -> dict[str, str]:
            mapped = {osp.splitext(name)[0]: name for name in names}
            if len(mapped) != len(names):
                seen: set[str] = set()
                for name in names:
                    stem = osp.splitext(name)[0]
                    if stem in seen:
                        raise ValueError(
                            f"Ambiguous pair in {root}: more than one file has stem {stem!r}."
                        )
                    seen.add(stem)
            return mapped

        gt_stem_to_name = by_stem(gt_names, gt_folder)
        input_stem_to_name = by_stem(input_names, input_folder)

        gt_stems = set(gt_stem_to_name.keys())
        input_stems = set(input_stem_to_name.keys())

        if filename_tmpl == "{}":
            missing_from_gt = input_stems - gt_stems
            missing_from_input = gt_stems - input_stems

            check_missing_paths(missing_from_gt, gt_key, gt_folder)
            check_missing_paths(missing_from_input, input_key, input_folder)

            assert len(input_stems) == len(gt_stems), (
                f"{input_key} and {gt_key} datasets have different number of images: "
                f"{len(input_stems)}, {len(gt_stems)}."
            )

            for stem in sorted(gt_stems):
                paired_paths.append(
                    {
                        f"{input_key}_path": osp.join(
                            input_folder, input_stem_to_name[stem]
                        ),
                        f"{gt_key}_path": osp.join(gt_folder, gt_stem_to_name[stem]),
                    }
                )
        else:
            claimed: dict[str, str] = {}
            for gt_name in sorted(gt_names):
                gt_dir = osp.dirname(gt_name)
                gt_basename = osp.splitext(osp.basename(gt_name))[0]

                templated_basename = filename_tmpl.format(gt_basename)
                input_stem = (
                    osp.join(gt_dir, templated_basename)
                    if gt_dir
                    else templated_basename
                )

                if input_stem in input_stem_to_name:
                    if input_stem in claimed:
                        raise ValueError(
                            f"Ambiguous pair in {input_folder}: GT files "
                            f"{claimed[input_stem]!r} and {gt_name!r} both select "
                            f"{input_stem_to_name[input_stem]!r}."
                        )
                    claimed[input_stem] = gt_name
                    paired_paths.append(
                        {
                            f"{input_key}_path": osp.join(
                                input_folder, input_stem_to_name[input_stem]
                            ),
                            f"{gt_key}_path": osp.join(gt_folder, gt_name),
                        }
                    )

    return paired_paths


def paths_from_folder(folder: str) -> list[str]:
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def generate_gaussian_kernel(kernel_size: int = 13, sigma: float = 1.6) -> np.ndarray:
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters

    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)  # type: ignore
