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
"""Tests for extract_frames CLI utility."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from extract_frames import extract_frames


@pytest.fixture()
def tmp_video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    path.write_text("fake video")
    return path


def test_extract_frames_invokes_ffmpeg(tmp_video: Path, tmp_path: Path) -> None:
    with mock.patch.object(subprocess, "run") as mock_run:
        extract_frames(tmp_video, tmp_path, fps=10.0)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(tmp_video),
        "-vf",
        "fps=10.0",
        str(tmp_path / "%06d.jpg"),
    ]
    mock_run.assert_called_once_with(cmd, check=True)
