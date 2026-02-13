from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import BaseTrack, TrackState
from .gallery_hybrid import GalleryHybridTracker
from .gallery_only import GalleryOnlyTracker

# ---------------------------------------------------------------------
# Public tracker registry
# ---------------------------------------------------------------------

_TRACKERS: Dict[str, Type[Any]] = {
    "gallery_hybrid": GalleryHybridTracker,
    "gallery_only": GalleryOnlyTracker,
}


def available_trackers() -> List[str]:
    """Return available tracker backend names."""
    return sorted(_TRACKERS.keys())


def create_tracker(name: str, **kwargs: Any) -> Any:
    """Instantiate a tracker by name."""
    if name not in _TRACKERS:
        raise ValueError(f"Unknown tracker: {name}. Available: {', '.join(available_trackers())}")
    return _TRACKERS[name](**kwargs)


__all__ = [
    # registry
    "available_trackers",
    "create_tracker",
    # trackers
    "GalleryHybridTracker",
    "GalleryOnlyTracker",
    # base exports (useful for downstream modules)
    "BaseTrack",
    "TrackState",
]