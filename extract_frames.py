# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command-line utility for extracting video frames using FFmpeg.

The script invokes FFmpeg via ``subprocess`` to extract frames at a given
frame rate. Progress is logged to the console.

Example:
    $ python extract_frames.py input.mp4 frames/ --fps 10

Example output:
    INFO:root:Extracting frames from input.mp4 to frames at 10fps
    INFO:root:Completed extraction in 2.1 seconds
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract video frames with FFmpeg")
    parser.add_argument("video", type=Path, help="Path to input video file")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write extracted frames",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frame rate for extraction (frames per second)",
    )
    return parser.parse_args()


def extract_frames(video: Path, output_dir: Path, fps: float) -> None:
    """Run FFmpeg to extract frames from ``video`` into ``output_dir``.

    Args:
        video: Path to the video file.
        output_dir: Directory to write JPEG frames.
        fps: Number of frames per second to extract.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        str(output_dir / "%06d.jpg"),
    ]

    logging.info("Extracting frames from %s to %s at %sfps", video, output_dir, fps)
    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logging.error("FFmpeg failed with return code %s", exc.returncode)
        raise
    elapsed = time.time() - start
    logging.info("Completed extraction in %.1f seconds", elapsed)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    extract_frames(args.video, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
