import json
import time
from pathlib import Path

from mmdet.apis import init_detector, inference_detector
import torch


def load_model():
    config = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/master/configs/yolox/yolox_s_8x8_300e_coco.py'
    checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211122_005924-1f976df1.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return init_detector(config, checkpoint, device=device)


def main():
    frames_dir = Path('frames')
    out_file = Path('detections.jsonl')
    model = load_model()
    classes = model.dataset_meta.get('classes', getattr(model, 'CLASSES', []))

    with out_file.open('w') as f:
        for img_path in sorted(frames_dir.glob('*.jpg')):
            start = time.time()
            result = inference_detector(model, str(img_path))
            elapsed = time.time() - start

            detections = result.pred_instances
            for bbox, score, label in zip(
                detections.bboxes, detections.scores, detections.labels
            ):
                x1, y1, x2, y2 = map(float, bbox)
                conf = float(score)
                obj = classes[int(label)] if classes else int(label)
                record = {
                    'frame': img_path.name,
                    'time': elapsed,
                    'obj': obj,
                    'conf': conf,
                    'bbox': [x1, y1, x2, y2]
                }
                f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
