from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _hash_to_color(key: str | int, s: float = 0.9, v: float = 0.95) -> Tuple[int, int, int]:
    """Stable BGR color from an id using HSV hashing."""
    import hashlib

    if not isinstance(key, str):
        key = str(key)
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16)
    hue = (h % 360) / 360.0

    i = int(hue * 6)
    f = hue * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(b * 255), int(g * 255), int(r * 255)


def _legible_text_color(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = bgr
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if y > 160 else (255, 255, 255)


def draw_box_label(
    img: np.ndarray,
    xyxy: Sequence[float],
    color: Tuple[int, int, int],
    text: Optional[str] = None,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    if text:
        tf = max(thickness - 1, 1)
        ts = 0.5 + 0.1 * (thickness - 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, ts, tf)
        th = th + 4
        y0 = max(0, y1 - th)
        cv2.rectangle(img, (x1, y0), (x1 + tw + 6, y1), color, -1)
        # Keep text baseline inside the image
        ty = max(0, y1 - 4)
        cv2.putText(
            img,
            text,
            (x1 + 3, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            ts,
            _legible_text_color(color),
            tf,
            cv2.LINE_AA,
        )


def draw_keypoints(
    img: np.ndarray,
    keypoints: Optional[Sequence[Sequence[float]]],
    skeleton: Optional[Sequence[Tuple[int, int]]] = None,
    color: Tuple[int, int, int] = (0, 255, 255),
    radius: int = 3,
) -> None:
    if not keypoints:
        return
    pts = np.asarray(keypoints, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return
    if skeleton is not None:
        for i, j in skeleton:
            if i < len(pts) and j < len(pts):
                pi = (int(pts[i, 0]), int(pts[i, 1]))
                pj = (int(pts[j, 0]), int(pts[j, 1]))
                cv2.line(img, pi, pj, color, 2, cv2.LINE_AA)
    for x, y, *rest in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)


def draw_polygon(
    img: np.ndarray,
    points: Optional[Sequence[Sequence[float]]],
    color: Tuple[int, int, int],
    alpha: float = 0.2,
    thickness: int = 2,
) -> None:
    if not points:
        return
    poly = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    dets: Iterable[Dict[str, Any]],
    *,
    class_names: Optional[Sequence[str]] = None,
    show_conf: bool = True,
    skeleton: Optional[Sequence[Tuple[int, int]]] = None,
    keypoint_radius: int = 4,
) -> np.ndarray:
    """Draw detections with track IDs and optional extras on a frame."""
    for det in dets:
        bbox = det.get("bbox") or det.get("xyxy")
        if not bbox:
            continue

        tid = det.get("track_id")
        gid = det.get("gallery_id")
        cls_id = det.get("class_id")
        cls_name = det.get("class_name")
        score = det.get("score")

        color = _hash_to_color(gid if gid is not None else (tid if tid is not None else "_"))

        label_parts: List[str] = []
        if gid is not None:
            label_parts.append(f"ID:{gid}")
        elif tid is not None:
            label_parts.append(f"ID:{tid}")

        # class label
        if cls_name:
            label_parts.append(str(cls_name))
        elif cls_id is not None:
            if isinstance(cls_id, int) and class_names and 0 <= cls_id < len(class_names):
                label_parts.append(str(class_names[cls_id]))
            else:
                label_parts.append(f"cls:{cls_id}")

        if show_conf and score is not None:
            try:
                label_parts.append(f"{float(score):.2f}")
            except Exception:
                pass

        label = " ".join(label_parts) if label_parts else None
        draw_box_label(frame, bbox, color=color, text=label)

        kps = det.get("keypoints")
        if kps is not None:
            draw_keypoints(frame, kps, skeleton=skeleton, color=color, radius=keypoint_radius)

        seg = det.get("segments")
        if seg is not None:
            # det-v1: either [[x,y], ...] (single polygon) or [[[x,y],...], ...] (multi-polygon)
            if isinstance(seg, list) and seg and isinstance(seg[0], list) and seg and isinstance(seg[0][0], (list, tuple)):
                draw_polygon(frame, seg[0], color=color)
            else:
                draw_polygon(frame, seg, color=color)

    return frame