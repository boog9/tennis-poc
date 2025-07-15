import json
import time
from pathlib import Path

from mmdet.apis import init_detector, inference_detector


CONFIG = "yolox_s_8x8_300e_coco.py"  # YOLOX-s config from mmdetection
CHECKPOINT = "yolox_s_8x8_300e_coco.pth"  # Path to pretrained weights


def main():
    # Initialize the model
    model = init_detector(CONFIG, CHECKPOINT, device="cuda:0")

    frames_dir = Path("frames")
    if not frames_dir.exists():
        raise FileNotFoundError("frames directory not found")

    with open("detections.jsonl", "w") as f:
        for img_path in sorted(frames_dir.glob("*.jpg")):
            start = time.time()
            result = inference_detector(model, str(img_path))
            elapsed = time.time() - start
            classes = model.CLASSES

            for label, bboxes in enumerate(result):
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    record = {
                        "frame": img_path.name,
                        "time": elapsed,
                        "obj": classes[label],
                        "conf": score,
                        "bbox": [x1, y1, x2, y2],
                    }
                    json.dump(record, f)
                    f.write("\n")


if __name__ == "__main__":
    main()
