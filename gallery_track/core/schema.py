from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_track_v1_header(
    det_payload: Dict[str, Any],
    *,
    tracker_name: str,
    track_classes: Optional[Sequence[int]],
    filter_gallery_for_tracked_classes: bool,
    tracker_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Create the top-level track-v1 payload skeleton from a normalized det payload.

    Keeps `video`, `detector`, and `frames` from detection. Adds a compact `tracker` block.
    """
    out: Dict[str, Any] = {}

    out["schema_version"] = "track-v1"
    out["parent_schema_version"] = str(det_payload.get("schema_version") or "det-v1")

    # Carry forward upstream context when present
    if "video" in det_payload:
        out["video"] = det_payload["video"]
    if "detector" in det_payload:
        out["detector"] = det_payload["detector"]

    out["created_at_utc"] = _utc_now_iso()

    out["tracker"] = {
        "name": tracker_name,
        "class_filter": {
            "track_classes": (list(track_classes) if track_classes is not None else None),
            "filter_gallery_for_tracked_classes": bool(filter_gallery_for_tracked_classes),
        },
        # tracker_meta should already be compact and serializable
        "config": tracker_meta,
    }

    # Frames will be filled/rewritten by pipeline
    out["frames"] = []
    return out


def compact_tracker_meta(**kwargs) -> Dict[str, Any]:
    """Helper to build a minimal serializable config dict (drops None values)."""
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        out[k] = v
    return out