from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _parse_class_filter(classes: Optional[Sequence[int]]) -> Optional[set[int]]:
    if classes is None:
        return None
    return set(int(x) for x in classes)


def normalize_detection_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an input detections payload to a predictable internal shape.

    Supports:
    - det-v1 (preferred): frames[*].frame, detections[*].bbox/score/class_id/class_name
    - legacy-ish: frames[*].frame_index, detections[*].bbox/conf/class

    Output guarantees:
    - payload["schema_version"] exists
    - frames is a list
    - each frame has "frame" (int) and "detections" (list)
    - each detection has:
        - "bbox" as 4 floats [x1,y1,x2,y2]
        - "score" as float (if present, else 1.0)
        - "class_id" as int when possible (if present), otherwise omitted
        - "class_name" preserved if present
      plus any extra keys preserved (keypoints, segments, etc.)
    """
    data = deepcopy(payload)

    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError("Invalid detection payload: missing 'frames' list")

    schema_version = str(data.get("schema_version") or "")
    # Do not force schema_version, but prefer det-v1 semantics when possible
    data["schema_version"] = schema_version if schema_version else "det-v1"

    for fi, fr in enumerate(data["frames"]):
        if not isinstance(fr, dict):
            raise ValueError(f"Invalid frame entry at index {fi}: expected dict")

        # frame index normalization
        if "frame" in fr:
            fr_idx = int(fr["frame"])
        elif "frame_index" in fr:
            fr_idx = int(fr["frame_index"])
        else:
            fr_idx = fi
        fr["frame"] = fr_idx
        fr.pop("frame_index", None)

        # detections normalization
        dets = fr.get("detections")
        if dets is None or not isinstance(dets, list):
            dets = []
            fr["detections"] = dets

        for di, det in enumerate(dets):
            if not isinstance(det, dict):
                raise ValueError(f"Invalid detection at frame={fr_idx} index={di}: expected dict")

            bbox = det.get("bbox") or det.get("xyxy")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                raise ValueError(f"Frame {fr_idx} det {di} missing/invalid 'bbox'")
            det["bbox"] = [float(v) for v in bbox[:4]]
            det.pop("xyxy", None)

            # score/conf normalization
            if "score" in det:
                det["score"] = float(det["score"])
            elif "conf" in det:
                det["score"] = float(det["conf"])
                det.pop("conf", None)
            else:
                det["score"] = float(det.get("score", 1.0))

            # class normalization
            if "class_id" in det:
                # keep as-is but try coercion to int
                try:
                    det["class_id"] = int(det["class_id"])
                except Exception:
                    pass
            elif "class" in det:
                # legacy key
                try:
                    det["class_id"] = int(det["class"])
                except Exception:
                    # if non-numeric, keep original under class_name and omit class_id
                    det.setdefault("class_name", str(det["class"]))
                det.pop("class", None)

            # class_name kept if present; otherwise leave unset

    return data


def split_tracked_vs_passthrough(
    frame: Dict[str, Any],
    track_classes: Optional[Sequence[int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split detections into (tracked, passthrough) based on class_id filter.

    If track_classes is None: everything is tracked, passthrough is empty.
    If a detection has no class_id:
      - it is treated as passthrough when a class filter is provided
      - it is treated as tracked when no class filter is provided
    """
    dets: List[Dict[str, Any]] = frame.get("detections", []) or []
    if track_classes is None:
        return dets, []

    wanted = _parse_class_filter(track_classes)
    tracked: List[Dict[str, Any]] = []
    passthrough: List[Dict[str, Any]] = []

    for d in dets:
        cid = d.get("class_id", None)
        if cid is None:
            passthrough.append(d)
            continue
        try:
            cid_i = int(cid)
        except Exception:
            passthrough.append(d)
            continue
        if wanted is not None and cid_i in wanted:
            tracked.append(d)
        else:
            passthrough.append(d)

    return tracked, passthrough