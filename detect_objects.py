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
"""Detect players and tennis balls in image frames using Detectron2.

The script loads a COCO-pretrained model and processes a directory of
JPEG frames, writing detections to a JSON file.

Example:
    $ python detect_objects.py frames/ detections.json

Example output (truncated):
    [
        {
            "frame": "000001.jpg",
            "bbox": [50.2, 30.1, 80.3, 100.5],
            "score": 0.92,
            "class": "person"
        },
        ...
    ]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect players and tennis balls")
    parser.add_argument(
        "frames_dir", type=Path, help="Directory containing JPEG frames"
    )
    parser.add_argument("output_json", type=Path, help="Path to output JSON file")
    return parser.parse_args()


def load_model(device: str | None = None) -> DefaultPredictor:
    """Initialize Detectron2 ``DefaultPredictor`` with COCO weights.

    Args:
        device: Optional device string (e.g. "cuda" or "cpu"). Defaults to
            ``cuda`` when available.

    Returns:
        A configured ``DefaultPredictor`` instance.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_cfg()
    config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


def detect_objects(frames_dir: Path, output_json: Path) -> None:
    """Run detection on all ``.jpg`` files in ``frames_dir``.

    Args:
        frames_dir: Directory of input JPEG frames.
        output_json: File to write detections as JSON array.
    """
    predictor = load_model()
    class_names = list(getattr(predictor, "metadata", {}).get("thing_classes", []))
    target_ids = {
        idx for idx, name in enumerate(class_names) if name in {"person", "sports ball"}
    }

    frames = sorted(frames_dir.glob("*.jpg"))
    results: List[Dict[str, Any]] = []

    for img_path in frames:
        img = cv2.imread(str(img_path))
        if img is None:
            logging.warning("Skipping unreadable frame %s", img_path)
            continue
        outputs = predictor(img)
        instances = outputs["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy().tolist()
        scores = instances.scores.cpu().numpy().tolist()
        classes = instances.pred_classes.cpu().numpy().tolist()
        for bbox, score, cls in zip(boxes, scores, classes):
            if cls not in target_ids:
                continue
            results.append(
                {
                    "frame": img_path.name,
                    "bbox": [float(x) for x in bbox],
                    "score": float(score),
                    "class": class_names[cls] if class_names else int(cls),
                }
            )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    detect_objects(args.frames_dir, args.output_json)


if __name__ == "__main__":
    main()
