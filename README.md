# tennis-poc

## Usage

```bash
python extract_frames.py input.mp4 frames/ --fps 10
```

This writes numbered JPEGs to the `frames/` directory.

## Docker Setup

A `Dockerfile` is provided for running the pipeline in an isolated environment. The image uses a PyTorch base with optional CUDA support and installs FFmpeg and all Python packages from `requirements.txt` (Detectron2, MMDetection, etc.).

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
