"""
track-lib package

gallery_track/
├── cli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── build_gallery.py
│   ├── export_reid.py
│   └── track_video.py
├── core/
│   ├── __init__.py
│   ├── device.py
│   ├── io.py
│   ├── normalize.py
│   ├── pipeline.py
│   ├── reid_registry.py
│   ├── result.py
│   └── schema.py
├── tools/
│   ├── __init__.py
│   ├── build_gallery.py
│   └── reid_export.py
├── trackers/
│   ├── __init__.py
│   ├── base.py
│   ├── gallery_hybrid.py
│   └── gallery_only.py
├── viz/
│   ├── __init__.py
│   └── draw.py
└── __init__.py

Lightweight top-level API.

Important:
- Do NOT import heavy dependencies (BoxMOT, cv2) at import time.
- Keep this module import-safe so tools like `python -m gallery_track.tools.reid_export`
  work even if tracking dependencies are not available.
"""

from __future__ import annotations

from importlib import metadata as _metadata
from typing import Any, List

__all__ = [
    "track_video",
    "TrackResult",
    "available_trackers",
]

def __getattr__(name: str) -> Any:
    # Lazy imports to avoid importing BoxMOT / cv2 on package import.
    if name == "track_video":
        from .core.pipeline import track_video as _track_video
        return _track_video
    if name == "TrackResult":
        from .core.result import TrackResult as _TrackResult
        return _TrackResult
    raise AttributeError(name)

def __version__() -> str:
    try:
        return _metadata.version("gallery-track")
    except Exception:
        return "0.0.0"

def available_trackers() -> List[str]:
    # Keep this import-safe. Trackers available are part of this package design.
    return ["gallery_hybrid", "gallery_only"]