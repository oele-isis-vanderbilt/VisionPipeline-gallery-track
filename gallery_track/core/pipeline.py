from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from ..trackers import create_tracker
from ..viz.draw import draw_frame
from .io import OutputPlan, VideoReader, VideoWriter, load_json, plan_outputs, save_json
from .normalize import normalize_detection_payload, split_tracked_vs_passthrough
from .result import TrackPaths, TrackResult
from .schema import compact_tracker_meta, make_track_v1_header
from .device import resolve_device
from .reid_registry import resolve_reid_weights

def track_video(
    *,
    dets_json: str | Path,
    video: str | Path,
    tracker: str = "gallery_hybrid",
    # class filtering
    classes: Optional[Sequence[int]] = None,
    filter_gallery: bool = False,
    # tracker knobs (hybrid)
    track_thresh: float = 0.45,
    match_thresh: float = 0.8,
    track_buffer: int = 25,
    frame_rate: int = 30,
    per_class: bool = False,
    max_obs: int = 30,
    # reid knobs
    reid_weights: Optional[str | Path] = None,
    gallery: Optional[str | Path] = None,
    reid_frequency: int = 10,
    gallery_match_threshold: float = 0.25,
    device: str = "auto",
    half: bool = False,
    models_dir: str | Path = "models",
    # artifacts (all OFF by default)
    save_json_flag: bool = False,
    save_frames: bool = False,
    save_video: Optional[str] = None,
    out_dir: str | Path = "out",
    run_name: Optional[str] = None,
    display: bool = False,
    save_fps: Optional[float] = None,
    fourcc: str = "mp4v",
    class_names: Optional[Sequence[str]] = None,
    no_progress: bool = False,
) -> TrackResult:
    """Run tracking over a video using detections JSON and return track-v1 payload.

    Detect-like behavior:
    - Always returns TrackResult(payload=track-v1 dict).
    - Writes nothing unless save_json_flag/save_frames/save_video is enabled.
    - If no artifacts enabled: no run folder is created.
    """
    device = resolve_device(device)

    # Resolve ReID weights early for consistent behavior across CLI + library.
    # We only support local paths (or names resolved under models_dir). Any failure is fatal
    # when the user explicitly provided --reid-weights.
    if reid_weights is not None:
        resolved = resolve_reid_weights(reid_weights, models_dir=models_dir)
        if resolved is None or not Path(resolved).exists():
            print(f"[gallery_track] ERROR: ReID weights not found/unresolved: {reid_weights}")
            raise SystemExit(2)
        reid_weights = resolved

    # For gallery trackers that require ReID, enforce required inputs.
    if tracker in ("gallery_hybrid", "gallery_only"):
        if reid_weights is None:
            # Hybrid can run without ReID, but will not do gallery matching. Only enforce
            # strictly for gallery_only.
            if tracker == "gallery_only":
                print("[gallery_track] ERROR: --reid-weights is required for tracker 'gallery_only'.")
                raise SystemExit(2)
        if gallery is None:
            # gallery_only requires a gallery; hybrid requires it only for gallery matching.
            if tracker == "gallery_only":
                print("[gallery_track] ERROR: --gallery is required for tracker 'gallery_only'.")
                raise SystemExit(2)

    det_payload_raw = load_json(dets_json)
    det_payload = normalize_detection_payload(det_payload_raw)

    # Plan outputs (only created if saving anything)
    plan: OutputPlan = plan_outputs(
        video_path=video,
        out_dir=out_dir,
        run_name=run_name,
        save_json_flag=save_json_flag,
        save_frames=save_frames,
        save_video_name=save_video,
    )
    if plan.run_dir is not None:
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        if plan.frames_dir is not None:
            plan.frames_dir.mkdir(parents=True, exist_ok=True)

    # Video setup
    vr = VideoReader(video)
    fps_out = save_fps if save_fps is not None else vr.fps
    vw: Optional[VideoWriter] = None
    if plan.video_path is not None:
        vw = VideoWriter(plan.video_path, fps=fps_out, size=vr.size, fourcc=fourcc)

    # Instantiate tracker backend
    # Note: we resolve ReID weights centrally here (path/name/url) for consistent behavior.
    tracker_kwargs: Dict[str, Any] = {}
    if tracker == "gallery_hybrid":
        tracker_kwargs = dict(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            frame_rate=frame_rate,
            per_class=per_class,
            max_obs=max_obs,
            reid_weights=(str(reid_weights) if reid_weights is not None else None),
            gallery_path=str(gallery) if gallery is not None else None,
            reid_frequency=reid_frequency,
            gallery_match_threshold=gallery_match_threshold,
            device=device,
            half=half,
            models_dir=models_dir,
        )
    elif tracker == "gallery_only":
        tracker_kwargs = dict(
            reid_weights=(str(reid_weights) if reid_weights is not None else None),
            gallery_path=str(gallery) if gallery is not None else None,
            gallery_match_threshold=gallery_match_threshold,
            device=device,
            half=half,
            track_thresh=track_thresh,
            models_dir=models_dir,
        )
    else:
        raise ValueError(f"Unknown tracker '{tracker}'")

    tracker_obj = create_tracker(tracker, **tracker_kwargs)

    # Build output payload header
    tracker_meta = compact_tracker_meta(
        # tracking knobs
        track_thresh=track_thresh,
        match_thresh=match_thresh if tracker == "gallery_hybrid" else None,
        track_buffer=track_buffer if tracker == "gallery_hybrid" else None,
        frame_rate=frame_rate if tracker == "gallery_hybrid" else None,
        per_class=per_class if tracker == "gallery_hybrid" else None,
        max_obs=max_obs if tracker == "gallery_hybrid" else None,
        # reid knobs
        reid_weights=str(reid_weights) if reid_weights is not None else None,
        gallery=str(gallery) if gallery is not None else None,
        reid_frequency=reid_frequency if tracker == "gallery_hybrid" else None,
        gallery_match_threshold=gallery_match_threshold,
        device=device,
        half=half,
    )
    out_payload = make_track_v1_header(
        det_payload,
        tracker_name=tracker,
        track_classes=classes,
        filter_gallery_for_tracked_classes=filter_gallery,
        tracker_meta=tracker_meta,
    )

    frames_in = det_payload.get("frames", [])
    n_frames_json = len(frames_in)

    processed = 0
    t0 = cv2.getTickCount()

    pbar = tqdm(total=n_frames_json, disable=no_progress, desc="Tracking", unit="frame")

    i = 0
    while True:
        ok, frame = vr.read()
        if not ok:
            break
        if i >= n_frames_json:
            break

        fr_in: Dict[str, Any] = frames_in[i]
        tracked_dets, passthrough_dets = split_tracked_vs_passthrough(fr_in, classes)

        # Build det array for tracked detections only: [x1,y1,x2,y2, score, class_id]
        if tracked_dets:
            arr = np.zeros((len(tracked_dets), 6), dtype=np.float32)
            for k, d in enumerate(tracked_dets):
                x1, y1, x2, y2 = d["bbox"][:4]
                score = float(d.get("score", 1.0))
                cls_id = d.get("class_id", 0)
                arr[k, :4] = [x1, y1, x2, y2]
                arr[k, 4] = score
                try:
                    arr[k, 5] = float(int(cls_id))
                except Exception:
                    arr[k, 5] = 0.0
        else:
            arr = np.zeros((0, 6), dtype=np.float32)

        outs = tracker_obj.update(arr, frame)

        # outs convention:
        #   [frame, x1, y1, x2, y2, track_id, score/conf, cls, gallery_id, det_ind]
        # where det_ind refers to index within *tracked_dets* in this frame.
        tracked_out: List[Dict[str, Any]] = [dict(d) for d in tracked_dets]

        if outs is not None and len(outs) > 0:
            for row in outs:
                try:
                    det_ind = int(row[-1])
                except Exception:
                    continue
                if not (0 <= det_ind < len(tracked_out)):
                    continue

                d = tracked_out[det_ind]

                # Update bbox with tracker-smoothed bbox
                d["bbox"] = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]

                # Always store numeric track id as string
                d["track_id"] = str(row[5])

                # Gallery id is optional (only present when tracker supports it)
                gid = row[8] if len(row) >= 10 else None
                if gid is not None and str(gid) != "":
                    d["gallery_id"] = str(gid)
                else:
                    d.pop("gallery_id", None)

        # If filter_gallery is enabled, drop ONLY the tracked-class detections
        # that did not receive a gallery_id. Passthrough detections remain intact.
        if filter_gallery and tracked_out:
            tracked_out = [d for d in tracked_out if d.get("gallery_id") is not None]

        # Compose final detections list: passthrough unchanged + tracked (possibly filtered)
        dets_out = passthrough_dets + tracked_out

        # Build output frame entry
        fr_out: Dict[str, Any] = {
            "frame": int(fr_in.get("frame", i)),
            "detections": dets_out,
        }

        # Preserve optional per-frame keys if present upstream (e.g., file)
        for k in ("file",):
            if k in fr_in and k not in fr_out:
                fr_out[k] = fr_in[k]

        out_payload["frames"].append(fr_out)

        # Optional visualization + artifacts
        if display or vw is not None or plan.frames_dir is not None:
            frame_vis = frame.copy()
            draw_frame(frame_vis, dets_out, class_names=class_names)

            if display:
                cv2.imshow("gallery_track", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    display = False
                    cv2.destroyWindow("gallery_track")

            if vw is not None:
                vw.write(frame_vis)

            if plan.frames_dir is not None:
                cv2.imwrite(str(plan.frames_dir / f"{i:06d}.jpg"), frame_vis)

        i += 1
        processed += 1
        pbar.update(1)

    pbar.close()

    t1 = cv2.getTickCount()
    elapsed = (t1 - t0) / cv2.getTickFrequency()
    fps_proc = processed / elapsed if elapsed > 0 else 0.0

    # Save JSON only if enabled
    if plan.json_path is not None:
        save_json(out_payload, plan.json_path)

    vr.release()
    if vw is not None:
        vw.release()

    paths = TrackPaths(
        run_dir=plan.run_dir,
        json_path=plan.json_path,
        video_path=plan.video_path,
        frames_dir=plan.frames_dir,
    )

    return TrackResult(
        payload=out_payload,
        paths=paths,
        stats={"frames": float(processed), "fps": float(fps_proc)},
    )