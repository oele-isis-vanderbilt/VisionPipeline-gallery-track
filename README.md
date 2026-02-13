# track-lib

A modular **video object tracking + gallery ReID** toolkit with a clean **track-v1** JSON schema, pluggable trackers, and optional tooling.

This is the **second stage** of the Vision Pipeline.

Trackers included:
- **gallery_hybrid**: temporal tracking (ByteTrack-style) + optional periodic gallery ReID
- **gallery_only**: gallery assignment only (no temporal association)

> By default, `track-lib` **does not write any files**. You opt-in to saving JSON, frames, or annotated video via flags.

---

## Vision Pipeline

```
Original Video (.mp4)
        │
        ▼
  detect-lib
  (Detection Stage)
        │
        └── detections.json  (det-v1)
                │
                ▼
        track-lib
        (Tracking + ReID Stage)
                │
                └── tracked.json (track-v1)
```

Stage 1 (Detection):
- PyPI: https://pypi.org/project/detect-lib/
- GitHub: https://github.com/Surya-Rayala/VideoPipeline-detection

---

## track-v1 output (returned + optionally saved)

`track-lib` always produces a canonical JSON payload in-memory with:

- `schema_version`: always **"track-v1"**
- `parent_schema_version`: upstream schema (typically **"det-v1"**)
- `video`: carried from det payload when present
- `detector`: carried from det payload when present
- `tracker`: tracker settings used for the run (name + config)
- `frames`: per-frame detections
  - tracking: `track_id` (string)
  - gallery ReID (optional): `gallery_id` (string)

### Minimal schema example

```json
{
  "schema_version": "track-v1",
  "parent_schema_version": "det-v1",
  "video": {
    "path": "in.mp4",
    "fps": 30.0,
    "frame_count": 120,
    "width": 1920,
    "height": 1080
  },
  "detector": {
    "name": "yolo_bbox",
    "weights": "yolo26n",
    "classes": null,
    "conf_thresh": 0.25,
    "imgsz": 640,
    "device": "cpu",
    "half": false
  },
  "tracker": {
    "name": "gallery_hybrid",
    "class_filter": {
      "track_classes": null,
      "filter_gallery_for_tracked_classes": false
    },
    "config": {
      "track_thresh": 0.45,
      "match_thresh": 0.8,
      "track_buffer": 25,
      "frame_rate": 30,
      "per_class": false,
      "max_obs": 30,
      "reid_weights": null,
      "gallery": null,
      "reid_frequency": 10,
      "gallery_match_threshold": 0.25,
      "device": "cpu",
      "half": false
    }
  },
  "frames": [
    {
      "frame": 0,
      "detections": [
        {
          "bbox": [100.0, 50.0, 320.0, 240.0],
          "score": 0.91,
          "class_id": 0,
          "class_name": "person",
          "track_id": "3",
          "gallery_id": "person_A"
        }
      ]
    }
  ]
}
```

### Returned vs saved

- **Returned (always):** the full track-v1 payload is available as `TrackResult.payload` (Python) and is always produced in-memory.
- **Saved (opt-in):** nothing is written unless you enable artifacts:
  - `--json` saves `tracked.json`
  - `--frames` saves annotated frames under `frames/`
  - `--save-video` saves an annotated video

When no artifacts are enabled, no output directory/run folder is created.

---

## Install with `pip` (PyPI)

> Use this if you want to install and use the tool without cloning the repo.

### Install

```bash
pip install track-lib
```

---

## CLI usage (pip)

Global help:

```bash
python -m gallery_track.cli.track_video -h
python -m gallery_track.tools.reid_export -h
python -m gallery_track.tools.build_gallery -h
```

Package version:

```bash
python -m gallery_track.cli --version
```

List trackers:

```bash
python -m gallery_track.cli.track_video --list-trackers
python -c "import gallery_track; print(gallery_track.available_trackers())"
```

List ReID architectures / known weight names (from your BoxMOT install):

```bash
python -m gallery_track.cli.track_video --list-reid-models
python -m gallery_track.cli.track_video --list-reid-weights
```

---

## Tracking CLI: `gallery_track.cli.track_video`

### Quick start (hybrid)

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_hybrid
```

### Quick start (gallery-only)

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_only \
  --reid-weights osnet_x0_25_msmt17.pt \
  --gallery galleries/
```

### Save artifacts (opt-in)

```bash
python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_hybrid \
  --json \
  --frames \
  --save-video annotated.mp4 \
  --out-dir out --run-name demo
```

### Tracker behavior overview

#### `gallery_hybrid`

Temporal tracker.

- Uses detection `score` to split detections into:
  - **high confidence**: `score > track_thresh` (main association + new track creation)
  - **low confidence**: `0.1 < score < track_thresh` (secondary association)
- Optionally assigns `gallery_id` by computing ReID embeddings for active tracks every `reid_frequency` frames and matching them to a gallery.

When you provide `--reid-weights` and `--gallery`, the tracker will attempt identity assignment.

#### `gallery_only`

No temporal association.

- For each frame, runs ReID on detections and assigns `gallery_id` by matching against the gallery.
- `track_id` becomes a per-frame detection identifier (stringified `det_ind`).

---

## CLI arguments

### Required

- `--dets-json <path>`: Path to det-v1 detections JSON (from detect-lib).
- `--video <path>`: Path to the original input video (must match the detections).

### Tracker selection

- `--tracker <name>`: `gallery_hybrid` (default) or `gallery_only`.

### Class filtering

- `--classes <ids>`: Track only these class IDs (comma/semicolon-separated). If omitted, all detections are tracked.
- `--filter-gallery`: When enabled, only tracked classes are filtered to those with a `gallery_id`. Non-tracked classes remain intact.

### Optional labels

- `--class-names <file>`: Text file with one class name per line (used only for drawing labels).

### Artifact saving (opt-in)

- `--json`: Save `tracked.json` under the run directory.
- `--frames`: Save annotated frames under the run directory (`frames/`).
- `--save-video <name.mp4>`: Save annotated video under the run directory.
- `--out-dir <dir>`: Output root directory used only if saving artifacts (default `out`).
- `--run-name <name>`: Run folder name inside out-dir (defaults to video stem).
- `--display`: Show live annotated frames (press `q` to stop display).
- `--save-fps <float>`: Override output video FPS (defaults to source FPS).
- `--fourcc <fourcc>`: FourCC for saved video (default `mp4v`).

### Hybrid tracker knobs (`gallery_hybrid`)

- `--track-thresh <float>`: High-confidence threshold for tracking.
  - Increase → fewer tracks, cleaner but may miss weak detections.
  - Decrease → more tracks, more noise.
  - Note: detections in `0.1 < score < track_thresh` can still contribute in a secondary association step.

- `--match-thresh <float>`: IoU association threshold.
  - Increase → stricter linking (fewer wrong matches) but more fragmentation.
  - Decrease → more aggressive linking but more ID switches.

- `--track-buffer <int>`: How long to keep lost tracks alive (in frames, scaled by `--frame-rate`).
  - Increase → better occlusion recovery.
  - Decrease → faster cleanup.

- `--frame-rate <int>`: Reference FPS used to scale the internal buffer (default 30).

- `--per-class`: Run independent tracking per class.
  - Helps avoid cross-class linking.
  - Adds small overhead.

- `--max-obs <int>`: Max observation history per track (used for state/history).

### ReID / gallery knobs

- `--reid-weights <path|name>`: ReID weights file path (recommended) or a name expected under `--models-dir`.
  - Required for `gallery_only`.
  - Optional for `gallery_hybrid` (if omitted, hybrid runs temporal tracking only).

- `--gallery <dir>`: Gallery root directory:

  ```
  galleries/
    person_A/
      *.jpg
    person_B/
      *.jpg
  ```

- `--reid-frequency <int>`: How often to run ReID matching (hybrid only).
  - Lower → more identity updates (slower).
  - Higher → fewer identity updates (faster).

- `--gallery-match-threshold <float>`: Cosine distance threshold.
  - Lower → stricter matches.
  - Higher → more assignments (higher risk of wrong IDs).

- `--device <str>`: Device selector: `auto`, `cpu`, `cuda`, `mps`, `0`, etc.
- `--half`: Enable FP16 where supported.
- `--models-dir <dir>`: Directory where weights are stored (default `models`).

### UX

- `--no-progress`: Disable progress bar output.

---

## Python usage (import)

You can use `track-lib` as a library after installing it with pip.

### Quick sanity check

```bash
python -c "import gallery_track; print(gallery_track.available_trackers())"
```

### Python API reference (keywords)

#### `gallery_track.track_video(...)`

**Required**
- `dets_json` (`str | Path`): Path to det-v1 JSON output from detect-lib.
- `video` (`str | Path`): Input video path.
- `tracker` (`str`): Tracker backend (`gallery_hybrid` or `gallery_only`).

**Class filtering**
- `classes` (`Sequence[int] | None`): Track only these class IDs.
- `filter_gallery` (`bool`): If True, drop tracked-class detections without a gallery identity.

**Hybrid tracker knobs (`gallery_hybrid`)**
- `track_thresh` (`float`): High-confidence threshold for tracking.
- `match_thresh` (`float`): IoU association threshold.
- `track_buffer` (`int`): Lost-track buffer length.
- `frame_rate` (`int`): Reference FPS used for buffer scaling.
- `per_class` (`bool`): Run tracker per class.
- `max_obs` (`int`): Observation history size.

**ReID / gallery knobs**
- `reid_weights` (`str | Path | None`): ReID weights path/name.
- `gallery` (`str | Path | None`): Gallery directory.
- `reid_frequency` (`int`): Match frequency (hybrid only).
- `gallery_match_threshold` (`float`): Cosine distance threshold.
- `device` (`str`): Device selector (`auto/cpu/cuda/mps/0`).
- `half` (`bool`): FP16.
- `models_dir` (`str | Path`): Weights directory.

**Artifacts (all off by default)**
- `save_json_flag` (`bool`): Save `tracked.json`.
- `save_frames` (`bool`): Save annotated frames under `frames/`.
- `save_video` (`str | None`): Filename for annotated video.
- `out_dir` (`str | Path`): Output root (used only if saving artifacts).
- `run_name` (`str | None`): Run folder name.
- `display` (`bool`): Live display window.
- `save_fps` (`float | None`): Output video FPS override.
- `fourcc` (`str`): Video fourcc (default `mp4v`).
- `class_names` (`Sequence[str] | None`): Class name mapping for visualization.
- `no_progress` (`bool`): Disable progress.

Returns a `TrackResult` with `payload` (track-v1 JSON), `paths` (only populated when saving), and `stats`.

### Run tracking from a Python file

Create `run_track.py`:

```python
from gallery_track import track_video

res = track_video(
    dets_json="detections.json",
    video="in.mp4",
    tracker="gallery_hybrid",
)

payload = res.payload
print(payload["schema_version"], len(payload["frames"]))
print(res.paths)  # populated only if you enable saving artifacts
print(res.stats)
```

Run:

```bash
python run_track.py
```

### Run tracking with gallery matching (Python)

```python
from gallery_track import track_video

res = track_video(
    dets_json="detections.json",
    video="in.mp4",
    tracker="gallery_hybrid",
    reid_weights="models/osnet_x0_25_msmt17.pt",
    gallery="galleries/",
    reid_frequency=10,
    gallery_match_threshold=0.25,
    device="auto",
)

print(res.payload["tracker"]["name"], res.stats)
```

---

## Install from GitHub (uv)

Use this if you are developing locally or want reproducible project environments.

Install uv:
https://docs.astral.sh/uv/getting-started/installation/#standalone-installer

Verify:

```bash
uv --version
```

### Install dependencies

```bash
git clone https://github.com/Surya-Rayala/VideoPipeline-gallery-track.git
cd VideoPipeline-gallery-track
uv sync
```

---

## CLI usage (uv)

Global help:

```bash
uv run python -m gallery_track.cli.track_video -h
uv run python -m gallery_track.tools.reid_export -h
uv run python -m gallery_track.tools.build_gallery -h
```

List trackers:

```bash
uv run python -m gallery_track.cli.track_video --list-trackers
```

Basic command (hybrid tracking):

```bash
uv run python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_hybrid
```

Basic command (gallery-only):

```bash
uv run python -m gallery_track.cli.track_video \
  --dets-json detections.json \
  --video in.mp4 \
  --tracker gallery_only \
  --reid-weights osnet_x0_25_msmt17.pt \
  --gallery galleries/
```

---

# ReID export tool

`track-lib` includes a thin wrapper around BoxMOT’s ReID export pipeline.

This tool:
- exports your ReID weights into one or more formats
- collects artifacts under a run folder (`--out-dir` / `--run-name`)
- writes `export_meta.json` with settings and final output paths

## CLI usage (pip)

Global help:

```bash
python -m gallery_track.tools.reid_export -h
```

List supported formats:

```bash
python -m gallery_track.tools.reid_export --list-formats
```

List ReID architectures / downloadable weight names (from BoxMOT registries, if available):

```bash
python -m gallery_track.tools.reid_export --list-reid-models
python -m gallery_track.tools.reid_export --list-reid-weights
```

Export TorchScript:

```bash
python -m gallery_track.tools.reid_export \
  --weights osnet_x0_25_msmt17.pt \
  --include torchscript \
  --out-dir models/exports --run-name osnet_ts
```

Export ONNX:

```bash
python -m gallery_track.tools.reid_export \
  --weights osnet_x0_25_msmt17.pt \
  --include onnx \
  --opset 18 --simplify \
  --out-dir models/exports --run-name osnet_onnx
```

## CLI usage (uv)

```bash
uv run python -m gallery_track.tools.reid_export -h
```

Example (uv + onnx):

```bash
uv run python -m gallery_track.tools.reid_export \
  --weights osnet_x0_25_msmt17.pt \
  --include onnx \
  --out-dir models/exports --run-name osnet_onnx
```

---

# Gallery builder tool

`track-lib` includes an optional GUI tool to build a gallery directory by drawing crops on a video.

It creates a directory structure compatible with both trackers:

```
galleries/
  identity_A/
    identity_A_00000.jpg
    identity_A_00001.jpg
  identity_B/
    identity_B_00000.jpg
```

## CLI usage (pip)

Help:

```bash
python -m gallery_track.tools.build_gallery -h
```

Launch:

```bash
python -m gallery_track.tools.build_gallery
```

Launch with initial paths:

```bash
python -m gallery_track.tools.build_gallery \
  --gallery-root galleries \
  --video in.mp4
```

## CLI usage (uv)

```bash
uv run python -m gallery_track.tools.build_gallery --gallery-root galleries --video in.mp4
```

---

# License

This project is licensed under the **AGPL-3.0 License**. See `LICENSE`.