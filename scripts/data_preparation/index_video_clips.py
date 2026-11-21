"""Write the clip-training index of each video: <folder>/index/<video stem>.json."""

import argparse
import time

from traiNNer.data.video_index import write_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="Video files to index.")
    parser.add_argument("--width", type=int, default=1488, help="Descale width.")
    parser.add_argument("--height", type=int, default=837, help="Descale height.")
    parser.add_argument("--kernel", default="bilinear", help="Descale kernel.")
    parser.add_argument("--border", default="zero", help="Descale border.")
    parser.add_argument("--threads", type=int, default=4, help="Decoder threads.")
    args = parser.parse_args()
    for video in args.videos:
        start = time.perf_counter()
        out = write_index(
            video,
            width=args.width,
            height=args.height,
            kernel=args.kernel,
            border=args.border,
            threads=args.threads,
        )
        print(f"{out} ({time.perf_counter() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
