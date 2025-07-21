tennis‑poc


A fully–containerised pipeline for frame extraction and object detection on tennis‑match videos.The provided Docker image ships with PyTorch 2.1, Detectron2 ≥ 0.6, FFmpeg and every package in requirements.txt, so nothing needs to be installed on the host beyond Docker (and NVIDIA drivers if you want GPU acceleration).

Quick start

1  Build the image

# GPU‑enabled image (default)
docker build -t tennis-poc .


# CPU‑only variant
# docker build --build-arg TORCH_IMAGE=pytorch/pytorch:2.1.0-cpu-py3.12 -t tennis-poc .

2  Extract frames inside the container


docker run --rm -v "$(pwd)":/app -it tennis-poc \
    python extract_frames.py input.mp4 frames/ --fps 10


This command writes numbered JPEG files to frames/.

3  Run object detection inside the container

# With GPU
docker run --rm --gpus all -v "$(pwd)":/app -it tennis-poc \
    python detect_objects.py frames/ detections.json

# CPU‑only hosts – just drop the --gpus flag

The file detections.json will contain one JSON object per frame with bounding boxes, confidences and class labels.

Directory layout

├── Dockerfile               # Container definition
├── README.md                # You are here
├── requirements.txt         # Python deps (CUDA‑agnostic)
├── extract_frames.py        # Frame extraction helper
├── detect_objects.py        # Detectron2 inference script
├── models/                  # (Optional) custom weights
├── frames/                  # Output of extract_frames.py
└── detections.json          # Sample detection output

Image contents

Component

Version / Source

PyTorch 2.1.0

CUDA / cuDNN 11.8 / 8 (runtime)

Detectron2 ≥ 0.6 (wheel for CUDA 11.8 + Torch 2.1)

FFmpeg Latest Debian package

Python 3 3.12 (from base image)

Tips & Troubleshooting

For larger datasets mount extra volumes, e.g. -v /data/videos:/videos.

Override the working directory with -w if you need a different path.

To add YOLOX via MMDetection, extend the Dockerfile – the base image already includes CUDA‑compatible PyTorch.
If you encounter “CUDA driver not found” errors, check that the host driver ≥ the CUDA runtime version (11.8).

Contributing

Fork → feature branch → PR against main.

Ensure pre‑commit run --all-files passes (Black, Flake8, isort).

Add/adjust unit tests where appropriate.

License

Apache 2.0 – see LICENSE for details.

Happy detecting & enjoy the rallies 🏸 (closest emoji to a tennis racquet!)
