"""Write the clip-training index of each video: <folder>/index/<video stem>.json."""

import argparse
import time
from pathlib import Path

import av
from traiNNer.data.descale import TRANSFERS
from traiNNer.data.video_index import write_index
from trainner_data.video_indexes import clip_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "videos", nargs="+", type=Path, help="Video files or folders to index."
    )
    parser.add_argument("--width", type=int, default=1488, help="Descale width.")
    parser.add_argument("--height", type=int, default=837, help="Descale height.")
    parser.add_argument("--kernel", default="bilinear", help="Descale kernel.")
    parser.add_argument("--border", default="zero", help="Descale border.")
    parser.add_argument(
        "--descale-linear-light",
        choices=TRANSFERS,
        help="Transfer curve for a linear-light descale.",
    )
    parser.add_argument("--threads", type=int, default=4, help="Decoder threads.")
    args = parser.parse_args()
    try:
        videos = clip_paths(args.videos)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    for video in videos:
        start = time.perf_counter()
        try:
            out = write_index(
                video,
                width=args.width,
                height=args.height,
                kernel=args.kernel,
                border=args.border,
                descale_linear_light=args.descale_linear_light,
                threads=args.threads,
            )
        except (OSError, ValueError, av.FFmpegError) as exc:
            parser.exit(1, f"{video}: {exc}\n")
        print(f"{out} ({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
