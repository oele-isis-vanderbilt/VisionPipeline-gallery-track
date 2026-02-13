from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


# -------------------------
# JSON I/O
# -------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, separators=(",", ":"))


# -------------------------
# Artifact output planning (detect-like)
# -------------------------

@dataclass(frozen=True)
class OutputPlan:
    """Compute output locations only when saving is enabled."""
    run_dir: Optional[Path]
    json_path: Optional[Path]
    frames_dir: Optional[Path]
    video_path: Optional[Path]


def plan_outputs(
    *,
    video_path: str | Path,
    out_dir: str | Path = "out",
    run_name: Optional[str] = None,
    save_json_flag: bool = False,
    save_frames: bool = False,
    save_video_name: Optional[str] = None,
) -> OutputPlan:
    """Return planned output paths. If nothing is being saved, all are None."""
    wants_any = bool(save_json_flag or save_frames or save_video_name)
    if not wants_any:
        return OutputPlan(run_dir=None, json_path=None, frames_dir=None, video_path=None)

    run_dir = Path(out_dir) / (run_name or Path(video_path).stem)
    json_path = run_dir / "tracked.json" if save_json_flag else None
    frames_dir = (run_dir / "frames") if save_frames else None
    video_out = (run_dir / save_video_name) if save_video_name else None
    return OutputPlan(run_dir=run_dir, json_path=json_path, frames_dir=frames_dir, video_path=video_out)


# -------------------------
# Video I/O
# -------------------------

class VideoReader:
    """Thin wrapper around cv2.VideoCapture with safe properties."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {self.path}")
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        self._w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
        return (self._w, self._h)

    @property
    def frame_count(self) -> int:
        return self._n

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        return True, frame

    def release(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class VideoWriter:
    """Simple MP4 writer using cv2.VideoWriter (default mp4v for compatibility)."""

    def __init__(self, path: str | Path, fps: float, size: Tuple[int, int], fourcc: str = "mp4v") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._size = (int(size[0]), int(size[1]))
        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(self.path, fourcc_code, self._fps, self._size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
            frame = cv2.resize(frame, self._size)
        self.writer.write(frame)

    def release(self) -> None:
        try:
            self.writer.release()
        except Exception:
            pass