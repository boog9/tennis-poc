# tennis-poc

This repository contains a simple pipeline for running object detection on extracted video frames. The project uses either YOLOX (via MMDetection) or Detectron2 and includes small CLI utilities for frame extraction.

## Usage

1. Place `.jpg` images in the `frames` directory.
2. Install dependencies (PyTorch, MMDetection/Detectron2 and their prerequisites).
3. Run `python yolox_detect.py` or `python detect_objects.py <frames_dir> <out.json>`.
4. Detection results are written to `detections.jsonl` or the specified JSON file.

## Frame Extraction

Use `extract_frames.py` to pull JPEG frames from a video using FFmpeg at a specific frame rate.

```bash
python extract_frames.py input.mp4 frames/ --fps 10
```

This writes numbered JPEGs to the `frames/` directory.

## Docker Setup

A `Dockerfile` is provided for running the pipeline in an isolated environment. The image uses a PyTorch base with optional CUDA support and installs FFmpeg and Detectron2.

```bash
# Build the image (GPU capable by default)
docker build -t tennis-poc .

# Run with access to your frames and output directories
docker run --gpus all -v $(pwd):/app -it tennis-poc bash
```

Inside the container you can run the detection scripts as usual:

```bash
python detect_objects.py frames/ detections.json
```

Set `--gpus all` only when an NVIDIA GPU is available. For CPU-only usage, omit the flag.
