# tennis-poc

This repository contains simple utilities for object detection and frame extraction.

## Usage

1. Place `.jpg` images in the `frames` directory.
2. Install dependencies (PyTorch and Detectron2).
3. Run `python detect_objects.py frames/ detections.json`.
4. Detection results are written line-by-line to `detections.json`.

## Frame Extraction

Run `extract_frames.py` to pull JPEG frames from a video using FFmpeg at a specific frame rate.

Example:

```bash
python extract_frames.py input.mp4 frames/ --fps 10
```

This writes numbered JPEGs to the `frames/` directory.
