tennis‑poc



A fully–containerised pipeline for frame extraction and object detection on tennis‑match videos.The provided Docker image ships with PyTorch 2.1, Detectron2 ≥ 0.6, FFmpeg and every package in requirements.txt, so nothing needs to be installed on the host beyond Docker (and NVIDIA drivers if you want GPU acceleration).


Quick start

1  Build the image

# GPU‑enabled image (default)
docker build -t tennis-poc .


# CPU‑only variant
# docker build --build-arg TORCH_IMAGE=pytorch/pytorch:2.1.0-cpu-py3.12 -t tennis-poc .

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
