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
"""Tests for the detect_objects module."""
from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import sys
import pytest

sys.modules.setdefault("cv2", SimpleNamespace(imread=lambda x: object()))
sys.modules.setdefault(
    "torch",
    SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
)
_d2 = SimpleNamespace()
sys.modules.setdefault("detectron2", _d2)
sys.modules.setdefault(
    "detectron2.config", SimpleNamespace(get_cfg=lambda: SimpleNamespace())
)
sys.modules.setdefault(
    "detectron2.engine", SimpleNamespace(DefaultPredictor=object)
)
sys.modules.setdefault(
    "detectron2.model_zoo",
    SimpleNamespace(get_config_file=lambda x: "", get_checkpoint_url=lambda x: ""),
)
_d2.config = sys.modules["detectron2.config"]
_d2.engine = sys.modules["detectron2.engine"]
_d2.model_zoo = sys.modules["detectron2.model_zoo"]

import detect_objects


class DummyArray:
    def __init__(self, data: list[list[float]] | list[float]):
        self.data = data

    def cpu(self) -> "DummyArray":
        return self

    def numpy(self) -> "DummyArray":
        return self

    def tolist(self) -> list[list[float]] | list[float]:
        return self.data


class DummyPredictor:
    """Predictor returning fixed detections for testing."""

    def __init__(self) -> None:
        self.metadata = {"thing_classes": ["person", "cat", "sports ball"]}

    def __call__(self, img: object) -> dict:
        instances = SimpleNamespace(
            pred_boxes=SimpleNamespace(tensor=DummyArray([[0, 0, 1, 1], [2, 2, 3, 3]])),
            scores=DummyArray([0.9, 0.8]),
            pred_classes=DummyArray([0, 2]),
        )
        return {"instances": instances}


@pytest.fixture()
def frame_dir(tmp_path: Path) -> Path:
    for i in range(2):
        (tmp_path / f"{i:06d}.jpg").write_bytes(b"0")
    return tmp_path


def test_detect_objects_writes_filtered_detections(frame_dir: Path, tmp_path: Path) -> None:
    out_file = tmp_path / "out.json"
    with mock.patch.object(detect_objects, "load_model", return_value=DummyPredictor()), \
         mock.patch.object(detect_objects.cv2, "imread", return_value=object()):
        detect_objects.detect_objects(frame_dir, out_file)

    data = json.loads(out_file.read_text())
    assert len(data) == 4
    assert all(item["class"] in ("person", "sports ball") for item in data)
    assert {d["frame"] for d in data} == {"000000.jpg", "000001.jpg"}
