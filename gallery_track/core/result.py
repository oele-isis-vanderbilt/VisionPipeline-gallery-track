from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrackPaths:
    """Filesystem paths produced when artifact saving is enabled.

    All fields are optional and only populated when corresponding flags are enabled.
    """
    run_dir: Optional[Path] = None
    json_path: Optional[Path] = None
    video_path: Optional[Path] = None
    frames_dir: Optional[Path] = None


@dataclass(frozen=True)
class TrackResult:
    """Result object returned by `gallery_track.track_video()`.

    - payload: track-v1 JSON dict (always present)
    - paths: populated only when saving is enabled
    - stats: small run statistics (frames processed, fps, etc.)
    """
    payload: Dict[str, Any]
    paths: TrackPaths
    stats: Dict[str, float]