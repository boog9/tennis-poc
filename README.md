tennis‑poc

This repository contains a simple pipeline for running object detection on extracted video frames using Detectron2. It includes small CLI utilities for frame extraction.

## Usage


1. Place `.jpg` images in the `frames` directory.
2. Install dependencies using the requirements file. Detectron2 must be
   installed separately since it provides CUDA-specific wheels:
   ```bash
   pip install -r requirements.txt
   pip install detectron2==0.6 \

       -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
   ```
3. Run `python detect_objects.py <frames_dir> <out.json>`.
4. Detection results are written to the specified JSON file.

## Frame Extraction

Use `extract_frames.py` to pull JPEG frames from a video using FFmpeg at a specific frame rate.

```bash
# Build the image (GPU capable by default)
docker build -t tennis-poc .

# Run with access to your frames and output directories
docker run --gpus all -v $(pwd):/app -it tennis-poc bash
```

This writes numbered JPEGs to the `frames/` directory.

## Docker Setup

A `Dockerfile` is provided for running the pipeline in an isolated environment. The image uses a PyTorch base with optional CUDA support and installs FFmpeg and all packages from `requirements.txt`. Detectron2 is installed from its official wheel repository as part of the build.

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
