import json
import os
import time

from mmdet.apis import init_detector, inference_detector

CONFIG = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco.py"
CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco.pth"

model = init_detector(CONFIG, CHECKPOINT, device="cpu")

frames_dir = os.path.join(os.path.dirname(__file__), "frames")

with open("detections.jsonl", "w") as f_out:
    for frame_name in sorted(os.listdir(frames_dir)):
        if not frame_name.lower().endswith(".jpg"):
            continue
        frame_path = os.path.join(frames_dir, frame_name)
        result = inference_detector(model, frame_path)
        # mmdetection uses COCO categories
        # get bounding boxes and class labels
        if isinstance(result, tuple):
            bbox_result, _ = result
        else:
            bbox_result = result

        # for each class
        for cls_id, bboxes in enumerate(bbox_result):
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox.tolist()
                out_rec = {
                    "frame": frame_name,
                    "time": time.time(),
                    "obj": int(cls_id),
                    "conf": float(score),
                    "bbox": [x1, y1, x2, y2]
                }
                f_out.write(json.dumps(out_rec) + "\n")
