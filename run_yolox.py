#!/usr/bin/env python3.12
"""Run YOLOX-s on frames and save detections."""

from pathlib import Path
import json
import torch
from mmdet.apis import init_detector, inference_detector


def main():
    config_url = (
        "https://raw.githubusercontent.com/open-mmlab/mmdetection/"
        "v3.1.0/configs/yolox/yolox_s_8x8_300e_coco.py"
    )
    checkpoint_url = (
        "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
        "yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211126_140254-1bd4e0c8.pth"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(config_url, checkpoint_url, device=device)
    class_names = list(model.dataset_meta.get("classes", []))

    frames = sorted(Path("frames").glob("*.jpg"))

    with open("detections.jsonl", "w", encoding="utf-8") as f:
        for idx, img_path in enumerate(frames):
            result = inference_detector(model, str(img_path))
            pred = result.pred_instances
            bboxes = pred.bboxes.cpu().numpy()
            labels = pred.labels.cpu().numpy()
            scores = pred.scores.cpu().numpy()

            for bbox, label, score in zip(bboxes, labels, scores):
                record = {
                    "frame": img_path.name,
                    "time": idx,
                    "obj": class_names[label] if label < len(class_names) else str(label),
                    "conf": float(score),
                    "bbox": [float(x) for x in bbox],
                }
                json.dump(record, f)
                f.write("\n")


if __name__ == "__main__":
    main()
