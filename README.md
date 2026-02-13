

# track-lib

A modular **video tracking + gallery ReID** toolkit built on top of `det-v1` detection outputs.

This package is the **second stage** in the Vision Pipeline.

---

# Vision Pipeline

```
Original Video (.mp4)
        │
        ▼
  detect-lib
  (Detection Stage)
        │
        ├── in.mp4
        └── detections.json  (det-v1)
                │
                ▼
        track-lib
        (Tracking + ReID Stage)
                │
                └── tracked.json (track-v1)
```

## Stage 1 — Detection

Detection is performed using:

- PyPI: https://pypi.org/project/detect-lib/
- GitHub: https://github.com/Surya-Rayala/VideoPipeline-detection

`detect-lib` produces a canonical `det-v1` JSON file.

## Stage 2 — Tracking (this package)

`track-lib` consumes:

- The original video
- The `det-v1` JSON produced by detect-lib

And produces:

- `track-v1` JSON
- Adds `track_id`
- Adds optional `gallery_id` (ReID)
- Preserves all upstream detection metadata

---

# Core Design Philosophy

Just like `detect-lib`:

- **Nothing is written to disk by default**
- All artifacts are **opt-in**
- Clean, canonical JSON schema
- CLI and Python API parity
- Import-safe package design

---

# track-v1 Output Schema

`track-lib` always returns a canonical JSON payload:

```json
{
  "schema_version": "track-v1",
  "parent_schema_version": "det-v1",
  "video": {...},
  "detector": {...},
  "tracker": {...},
  "frames": [...]
}
```

Each detection may contain:

- `bbox`
- `score`
- `class_id`
- `track_id` (always string)
- `gallery_id` (optional)
- `keypoints` / `segments` (if upstream detection had them)

---

# Installation

## Install from PyPI

```bash
pip install track-lib
```

Requires Python ≥ 3.10.

---

# CLI Usage

Main command:

```bash
python -m gallery_track.cli.track_video -h
```

---

# Trackers

`track-lib` provides two tracking backends:

---

## 1️⃣ gallery_hybrid (Default)

Temporal tracker (ByteTrack-style) + optional periodic gallery ReID.

### Features

- Kalman filter motion model
- IoU-based association
- Track lifecycle management (new / tracked / lost / removed)
- Optional gallery-based identity assignment
- Robust to occlusions

### Basic Command

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_hybrid
```

### Hybrid + Gallery ReID

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_hybrid \
  --reid-weights osnet_x0_25_msmt17.pt \
  --gallery galleries/
```

---

## 2️⃣ gallery_only

Frame-wise identity assignment only.

No temporal memory.

### Use Cases

- Identity tagging
- Face/person recognition
- Offline analysis

### Command

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_only \
  --reid-weights osnet_x0_25_msmt17.pt \
  --gallery galleries/
```

---

# Class Filtering

```bash
--classes 0,1
--filter-gallery
```

### --classes
Restrict tracking to specified class IDs.

If omitted → all detections are tracked.

### --filter-gallery
Drops tracked detections that did not receive a `gallery_id`.

Passthrough detections remain unchanged.

---

# Tracking Parameters (gallery_hybrid)

## --track-thresh
Minimum detection confidence for high-confidence tracking.

- Increase → fewer false positives, stricter tracking
- Decrease → more tracks, potentially noisier

Detections between 0.1 and track_thresh may still be used in secondary association.

---

## --match-thresh
IoU association threshold.

- Higher → stricter linking
- Lower → more aggressive linking

---

## --track-buffer
How long to keep lost tracks alive.

- Higher → better occlusion recovery
- Lower → faster cleanup

---

## --per-class
Maintain separate tracker per class.

Reduces cross-class ID switches.

---

## --max-obs
Maximum history per track.

Higher → smoother motion but more memory.

---

# ReID / Gallery Parameters

## --reid-weights
Path to ReID weights.

Required for:
- gallery_only

Optional for:
- gallery_hybrid

---

## --gallery
Directory structured as:

```
galleries/
   identity_A/
   identity_B/
```

---

## --reid-frequency (hybrid only)
Run gallery matching every N frames.

Lower → more stable identity
Higher → faster runtime

---

## --gallery-match-threshold
Cosine distance threshold.

Lower → stricter matching
Higher → more assignments

---

## --device

```
auto | cpu | cuda | mps | 0
```

Auto resolves: cuda > mps > cpu

---

## --half
Enable FP16 inference where supported.

---

# Artifact Saving (All Optional)

```bash
--json
--frames
--save-video annotated.mp4
--out-dir out
--run-name exp1
--display
```

If none are enabled → no output folder is created.

---

# Complete Command Examples

## Hybrid Full Example

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_hybrid \
  --track-thresh 0.5 \
  --match-thresh 0.8 \
  --track-buffer 30 \
  --reid-weights osnet_x0_25_msmt17.pt \
  --gallery galleries/ \
  --json \
  --save-video annotated.mp4
```

---

## Gallery Only Example

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_only \
  --reid-weights osnet_x0_25_msmt17.pt \
  --gallery galleries/ \
  --json
```

---

# Python API Usage

```python
from gallery_track import track_video

res = track_video(
    dets_json="detections.json",
    video="in.mp4",
    tracker="gallery_hybrid",
)

print(res.payload["schema_version"])
print(res.stats)
```

Returns:

```python
TrackResult(
    payload: dict,
    paths: TrackPaths,
    stats: dict
)
```

---

# Gallery Builder GUI Tool

Launch:

```bash
python -m gallery_track.tools.build_gallery
```

Features:

- Load video
- Create identities
- Draw bounding boxes
- Save crops
- Compatible with both trackers

---

# ReID Export Tool

```bash
python -m gallery_track.tools.reid_export \
  --weights osnet_x0_25_msmt17.pt \
  --include onnx
```

Exports model formats and generates `export_meta.json`.

---

# Development (uv)

```bash
git clone https://github.com/Surya-Rayala/VideoPipeline-gallery-track.git
cd VideoPipeline-gallery-track
uv sync
```

Run CLI:

```bash
uv run python -m gallery_track.cli.track_video -h
```

---

# Schema Compatibility

| Stage | Package | Schema |
|--------|----------|--------|
| Detection | detect-lib | det-v1 |
| Tracking | track-lib | track-v1 |

---

# License

This project is licensed under the **AGPL-3.0 License**.

See `LICENSE` for full details.