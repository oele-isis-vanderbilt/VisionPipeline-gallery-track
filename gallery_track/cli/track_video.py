from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .. import track_video
from ..trackers import available_trackers

# Optional BoxMOT registries (graceful fallback if unavailable)
# BoxMOT reorganized ReID modules across versions; support both layouts.
try:  # BoxMOT newer (v16+)
    from boxmot.reid.core.factory import MODEL_FACTORY  # {arch_name: ctor}
    from boxmot.reid.core.config import TRAINED_URLS  # {weight_name: url}
except Exception:  # pragma: no cover
    try:  # BoxMOT older
        from boxmot.appearance.reid.factory import MODEL_FACTORY  # type: ignore
        from boxmot.appearance.reid.config import TRAINED_URLS  # type: ignore
    except Exception:  # pragma: no cover
        MODEL_FACTORY = {}
        TRAINED_URLS = {}

# Keep a stable baseline list for help output even if BoxMOT registries are unavailable.
SUPPORTED_REID_MODELS = [
    "resnet50",
    "resnet101",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "hacnn",
    "mlfn",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
    "osnet_ain_x0_75",
    "osnet_ain_x0_5",
    "osnet_ain_x0_25",
    "lmbn_n",
    "clip",
]


def _parse_classes(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts]


def _load_class_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return None


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run tracking on a video using det-v1 detections JSON and emit track-v1 output.\n\n"
            "Input:\n"
            "  - --dets-json: det-v1 JSON from the `detect` package (VideoPipeline-detection / detect-lib).\n"
            "  - --video: the same source video used to generate the detections.\n\n"
            "Output:\n"
            "  - Always returns/prints a track-v1 payload in-memory (library) and a compact stats dict (CLI).\n"
            "  - Writes nothing unless you enable --json/--frames/--save-video.\n\n"
            "Tip: Use --list-trackers / --list-reid-models / --list-reid-weights for discovery."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Core inputs
    ap.add_argument(
        "--dets-json",
        type=Path,
        required=False,
        help="Path to det-v1 detections JSON (from detect). Required for tracking; must align with --video frame order.",
    )
    ap.add_argument(
        "--video",
        type=Path,
        required=False,
        help="Path to input video (same video used for detection). Used for frame reading + optional visualization.",
    )

    # Tracker selection
    ap.add_argument(
        "--tracker",
        type=str,
        default="gallery_hybrid",
        choices=available_trackers(),
        help="Tracking backend. 'gallery_hybrid' = temporal tracker + optional periodic gallery ReID; 'gallery_only' = per-frame gallery assignment (no temporal tracking).",
    )

    # Class selection + filtering semantics
    ap.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Class IDs to TRACK (comma/semicolon-separated). If omitted: all detections are tracked. When provided: only these classes get track_id/gallery matching; others are passed through unchanged.",
    )
    ap.add_argument(
        "--filter-gallery",
        action="store_true",
        help="If set: within the tracked classes, drop detections that did NOT get a gallery_id. Passthrough (non-tracked) detections are kept.",
    )

    # Optional label mapping
    ap.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="Optional text file with one class name per line (index = class_id). Used only for visualization labels.",
    )

    # Artifacts (all OFF by default, detect-like)
    ap.add_argument(
        "--json",
        action="store_true",
        help="Save track-v1 output as tracked.json under the run directory (opt-in; default: no files).",
    )
    ap.add_argument(
        "--frames",
        action="store_true",
        help="Save annotated frames as JPEGs under <run>/frames/ (opt-in; can be large).",
    )
    ap.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Save an annotated video under the run directory (e.g. annotated.mp4).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Output root used only when saving artifacts (default: out). No run folder is created if you don't enable saving.",
    )
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name inside out-dir. Defaults to the input video stem.",
    )
    ap.add_argument(
        "--display",
        action="store_true",
        help="Show a live window with annotations (press 'q' to stop display). Does not write files unless saving flags are set.",
    )
    ap.add_argument(
        "--save-fps",
        type=float,
        default=None,
        help="Override FPS for the saved video only (default: source FPS). Useful when source FPS metadata is wrong.",
    )
    ap.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="FourCC codec for --save-video (default: mp4v). Try avc1/H264 if your OpenCV build supports it.",
    )

    # Tracker knobs (hybrid)
    ap.add_argument(
        "--track-thresh",
        type=float,
        default=0.45,
        help="Primary confidence threshold for high-confidence detections (used for main matching + new track creation). Detections between 0.1 and track_thresh may still be used in second association of hybrid tracker.",
    )
    ap.add_argument(
        "--match-thresh",
        type=float,
        default=0.8,
        help="Association threshold for IoU-based matching (hybrid only). Higher = stricter matching (fewer wrong links, more fragmentation); lower = more aggressive linking (fewer fragments, more ID swaps).",
    )
    ap.add_argument(
        "--track-buffer",
        type=int,
        default=25,
        help="How long to keep 'lost' tracks alive before removing (hybrid only). Higher = more re-identification after occlusion; lower = faster cleanup but more new IDs.",
    )
    ap.add_argument(
        "--frame-rate",
        type=int,
        default=30,
        help="Reference FPS used to scale the internal buffer (hybrid only). Set this to your video's FPS if very different from 30 to keep occlusion time behavior consistent.",
    )
    ap.add_argument(
        "--per-class",
        action="store_true",
        help="If set: maintain separate trackers per class_id (hybrid only). Can reduce cross-class ID swaps but increases compute/state.",
    )
    ap.add_argument(
        "--max-obs",
        type=int,
        default=30,
        help="Max observation history stored per track (hybrid only). Higher can help smoothing/robustness; increases memory.",
    )

    # ReID / gallery knobs
    ap.add_argument(
        "--reid-weights",
        type=str,
        default=None,
        help=(
            "ReID weights path or name under --models-dir.\n"
            "- For gallery_only: required.\n"
            "- For gallery_hybrid: optional; if omitted, hybrid runs temporal tracking without gallery matching.\n"
            "Note: some exported backends only support batch=1; the tracker will warn and fall back if needed."
        ),
    )
    ap.add_argument(
        "--gallery",
        type=str,
        default=None,
        help="Gallery root directory: subfolders per identity, images inside each (used for embedding gallery). Required for gallery_only; optional for hybrid.",
    )
    ap.add_argument(
        "--reid-frequency",
        type=int,
        default=10,
        help="Run gallery matching every N frames (hybrid only). Higher = faster but identity updates are less frequent; lower = more stable IDs but slower.",
    )
    ap.add_argument(
        "--gallery-match-threshold",
        type=float,
        default=0.25,
        help="Cosine distance threshold for gallery assignment. Lower = stricter matches (fewer assignments, fewer false IDs); higher = more assignments but higher risk of wrong identity.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: auto/cpu/cuda/mps/0. 'auto' prefers cuda > mps > cpu.",
    )
    ap.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 for ReID where supported (GPU-friendly). Can be faster and use less VRAM; may reduce numeric stability on some backends.",
    )
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory where you store ReID weights for name-based resolution (no auto-download).",
    )

    # UX
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar output.",
    )
    ap.add_argument(
        "--list-trackers",
        action="store_true",
        help="Print available trackers and exit.",
    )

    # ReID registry listing (so users don't need to jump to export tool)
    ap.add_argument(
        "--list-reid-models",
        action="store_true",
        help="Print available ReID architectures (from BoxMOT if available) and exit.",
    )
    ap.add_argument(
        "--list-reid-weights",
        action="store_true",
        help="Print known pretrained ReID weight names (from BoxMOT TRAINED_URLS if available) and exit.",
    )

    ap.epilog = (
        "Examples:\n"
        "  # Hybrid temporal tracking (no gallery matching)\n"
        "  python -m gallery_track.cli.track_video --dets-json detections.json --video in.mp4 --tracker gallery_hybrid\n\n"
        "  # Hybrid + gallery matching (every 10 frames)\n"
        "  python -m gallery_track.cli.track_video --dets-json detections.json --video in.mp4 \\\n"
        "    --tracker gallery_hybrid --reid-weights osnet_x0_25_msmt17.pt --gallery galleries/ --reid-frequency 10\n\n"
        "  # Gallery-only (per-frame identity assignment; requires weights + gallery)\n"
        "  python -m gallery_track.cli.track_video --dets-json detections.json --video in.mp4 \\\n"
        "    --tracker gallery_only --reid-weights osnet_x0_25_msmt17.pt --gallery galleries/\n\n"
        "  # Save artifacts (creates out/<run-name>/...)\n"
        "  python -m gallery_track.cli.track_video --dets-json detections.json --video in.mp4 --json --save-video annotated.mp4\n"
    )

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if getattr(args, "list_trackers", False):
        print("Available trackers:")
        for n in available_trackers():
            print("  -", n)
        return

    if getattr(args, "list_reid_models", False):
        print("Available ReID architectures:")
        # Prefer BoxMOT registry if present, otherwise print our stable baseline list.
        names = sorted(MODEL_FACTORY.keys()) if MODEL_FACTORY else sorted(set(SUPPORTED_REID_MODELS))
        for n in names:
            print("  -", n)
        if not MODEL_FACTORY:
            print("\n(note: BoxMOT MODEL_FACTORY registry not found; list shown is a baseline set.)")
        return

    if getattr(args, "list_reid_weights", False):
        print("Known ReID weight names (if provided by your BoxMOT install):")
        if not TRAINED_URLS:
            print("  (none found; TRAINED_URLS registry not available in this BoxMOT install)")
        else:
            for n in sorted(TRAINED_URLS.keys()):
                print("  -", n)
        return

    if not (args.dets_json and args.video):
        ap.error("--dets-json and --video are required unless using a --list-* option")

    class_names = _load_class_names(args.class_names)
    class_filter = _parse_classes(args.classes)

    res = track_video(
        dets_json=args.dets_json,
        video=args.video,
        tracker=args.tracker,
        classes=class_filter,
        filter_gallery=args.filter_gallery,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        frame_rate=args.frame_rate,
        per_class=args.per_class,
        max_obs=args.max_obs,
        reid_weights=args.reid_weights,
        gallery=args.gallery,
        reid_frequency=args.reid_frequency,
        gallery_match_threshold=args.gallery_match_threshold,
        device=args.device,
        half=args.half,
        models_dir=args.models_dir,
        save_json_flag=bool(args.json),
        save_frames=bool(args.frames),
        save_video=args.save_video,
        out_dir=args.out_dir,
        run_name=args.run_name,
        display=args.display,
        save_fps=args.save_fps,
        fourcc=args.fourcc,
        class_names=class_names,
        no_progress=args.no_progress,
    )

    # Detect-like: print payload to stdout if not saving json? We'll always print a compact summary.
    # The payload is available via res.payload for library usage.
    stats = {k: (int(v) if k == "frames" else round(v, 2)) for k, v in res.stats.items()}
    print(stats)


if __name__ == "__main__":
    main()