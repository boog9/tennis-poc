# tennis-poc

This repository contains a simple example of running object detection with YOLOX-S from [MMDetection](https://github.com/open-mmlab/mmdetection).

## Usage

1. Place `.jpg` images in the `frames` directory.
2. Install dependencies (PyTorch, MMDetection and their prerequisites).
3. Run `python yolox_detect.py`.
4. Detection results are written line-by-line to `detections.jsonl`.

## Frame Extraction

Run `extract_frames.py` to pull JPEG frames from a video using FFmpeg at a
specific frame rate.

Example:

```bash
python extract_frames.py input.mp4 frames/ --fps 10
```

This writes numbered JPEGs to the `frames/` directory.
