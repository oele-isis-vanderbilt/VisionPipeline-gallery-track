# gallery-track-lib

**Minimum Python:** `>=3.10`

**gallery-track-lib** is a modular **video object tracking + gallery ReID** toolkit with a clean **track-v1** JSON schema, pluggable trackers, and optional tooling.

This is the **second stage** of the Vision Pipeline.

Trackers included:
- **gallery_hybrid**: temporal tracking (ByteTrack-style) + optional periodic gallery ReID
- **gallery_only**: gallery assignment only (no temporal association)

> By default, `gallery-track-lib` **does not write any files**. You opt-in to saving JSON, frames, or annotated video via flags.

---

## Vision Pipeline

```
Original Video (.mp4) ───────────────┐
        │                            │
        ▼                            │
  detect-lib                         │
  (Detection Stage)                  │
        │                            │
        └── detections.json (det-v1) │
                     │               │
                     └──────┐        │
                            ▼        ▼
                         track-lib (Tracking + ReID Stage)
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
> Requires **Python >= 3.10**.

### Install

```bash
pip install gallery-track-lib
```

---

## CLI usage (pip)

Global help:

```bash
python -m gallery_track.cli.track_video -h
python -m gallery_track.tools.reid_export -h
python -m gallery_track.tools.build_gallery -h
```

> Note: the PyPI package name is `gallery-track-lib`, but the Python module/import name remains `gallery_track`.

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

- `--dets-json <path>`: Path to the det-v1 detections JSON produced by detect-lib (must correspond to the same video passed via `--video`).
- `--video <path>`: Path to the source video used for detection. The tracker reads frames from this file for timing/visualization and expects frame order to match `--dets-json`.

### Tracker selection

- `--tracker <name>`: Tracking backend to use. `gallery_hybrid` (default) performs temporal association; `gallery_only` assigns identities per frame with no temporal linking.

### Class filtering

- `--classes <ids>`: Comma/semicolon-separated class IDs to *track*. If omitted, all classes are tracked. If provided, only these classes receive `track_id` / gallery matching; other classes are passed through unchanged.
- `--filter-gallery`: For tracked classes only, drop detections that did not receive a `gallery_id`. Detections from non-tracked classes are always kept.

### Optional labels

- `--class-names <file>`: Optional newline-delimited class-name file where line index = `class_id`. Used only for on-frame labels (does not affect tracking logic).

### Artifact saving (opt-in)

- `--json`: Write the track-v1 payload to `<run>/tracked.json`.
- `--frames`: Save annotated frames as JPEGs under `<run>/frames/` (can be large).
- `--save-video <name.mp4>`: Save an annotated video as `<run>/<name.mp4>`.
- `--out-dir <dir>`: Output root used only when saving artifacts (default: `out`). No run folder is created unless a saving flag is enabled.
- `--run-name <name>`: Name of the run folder under `--out-dir`. Defaults to the input video stem.
- `--display`: Show a live annotated window while processing (press `q` to quit). Does not write files unless saving flags are set.
- `--save-fps <float>`: Override FPS for the saved video only (default: source FPS). Useful if the source FPS metadata is incorrect.
- `--fourcc <fourcc>`: FourCC codec for `--save-video` (default: `mp4v`). Try `avc1`/`H264` if supported by your OpenCV build.

### Hybrid tracker knobs (`gallery_hybrid`)

- `--track-thresh <float>` (hybrid only): Confidence threshold for primary association and new track creation. Detections in `0.1 < score < track_thresh` may still be used in a secondary association step.
  - Increase → fewer tracks, cleaner but may miss weak detections.
  - Decrease → more tracks, more noise.

- `--match-thresh <float>` (hybrid only): IoU matching threshold for associating detections to existing tracks.
  - Increase → stricter linking (fewer wrong matches) but more fragmentation.
  - Decrease → more aggressive linking but more ID switches.

- `--track-buffer <int>` (hybrid only): Max number of frames to keep a lost track before it is removed.
  - Increase → better occlusion recovery.
  - Decrease → faster cleanup.

- `--frame-rate <int>` (hybrid only): Reference FPS used to scale time-based behavior in the tracker (default 30). Set to your video FPS if it differs significantly.

- `--per-class` (hybrid only): Maintain independent tracking state per `class_id` (reduces cross-class ID swaps at the cost of more state/compute).
  - Helps avoid cross-class linking.
  - Adds small overhead.

- `--max-obs <int>` (hybrid only): Max observation history stored per track for internal smoothing/state.

### ReID / gallery knobs

- `--reid-weights <path|name>`: ReID weights to use for embedding extraction.
  - Provide either an explicit file path **or** a filename that exists under `--models-dir`.
  - **Custom weights naming note:** If you pass a *name* (not a full path), make sure the weight filename starts with one of the **model names** printed by `--list-reid-models` (this improves architecture auto-detection / compatibility). Example: `osnet_*`, `lmbn_*`, etc.
  - Required for `gallery_only`.
  - Optional for `gallery_hybrid` (if omitted, hybrid runs temporal tracking with no gallery assignment).

- `--gallery <dir>`: Gallery root directory containing one subfolder per identity (images inside each).

  ```
  galleries/
    person_A/
      *.jpg
    person_B/
      *.jpg
  ```

- `--reid-frequency <int>` (hybrid only): Run gallery matching every N frames (lower = more frequent updates, higher = faster).
- `--gallery-match-threshold <float>`: Cosine-distance threshold for assigning a `gallery_id` (lower = stricter, higher = more assignments but more risk of false IDs).
- `--device <str>`: Compute device for ReID: `auto`, `cpu`, `cuda`, `mps`, or a CUDA device index like `0`.
- `--half`: Enable FP16 for ReID when supported (typically GPU-only).
- `--models-dir <dir>`: Directory used for resolving weight *names* passed to `--reid-weights` (default: `models`).

### UX

- `--no-progress`: Disable the tqdm progress bar (useful for clean logs).

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
- `dets_json`: Path to det-v1 detections JSON (must correspond to the same video passed via `video`).
- `video`: Path to the source video used for detection.
- `tracker`: Tracker backend (`gallery_hybrid` or `gallery_only`).

**Class filtering**
- `classes`: Class IDs to track. If provided, only these classes receive track IDs / gallery matching; other classes pass through unchanged.
- `filter_gallery`: For tracked classes only, drop detections that did not receive a `gallery_id`.

**Hybrid tracker knobs (`gallery_hybrid`)**
- `track_thresh` (hybrid only): Confidence threshold for primary association and new track creation.
- `match_thresh` (hybrid only): IoU matching threshold for associating detections to existing tracks.
- `track_buffer` (hybrid only): Max number of frames to keep a lost track before removal.
- `frame_rate` (hybrid only): Reference FPS used to scale time-based behavior (default 30).
- `per_class` (hybrid only): Maintain independent tracking state per class_id.
- `max_obs` (hybrid only): Max observation history stored per track.

**ReID / gallery knobs**
- `reid_weights`: Weights to use for ReID embedding extraction (path or name under `models_dir`).
- `gallery`: Gallery root directory (subfolder per identity, images inside).
- `reid_frequency` (hybrid only): Run gallery matching every N frames.
- `gallery_match_threshold`: Cosine-distance threshold for assigning a gallery_id.
- `device`: Compute device for ReID (`auto/cpu/cuda/mps/0`).
- `half`: Enable FP16 for ReID when supported.
- `models_dir`: Directory used to resolve weight names.

**Artifacts (all off by default)**
- `save_json_flag`: Write `<run>/tracked.json`.
- `save_frames`: Write annotated JPEG frames under `<run>/frames/`.
- `save_video`: Filename for annotated video under the run folder (e.g., `annotated.mp4`).
- `out_dir`: Output root used only when saving artifacts.
- `run_name`: Run folder name (defaults to video stem).
- `display`: Show a live annotated window during processing.
- `save_fps`: Override FPS for saved video only.
- `fourcc`: FourCC codec for saved video (default `mp4v`).
- `class_names`: Optional class-name mapping for visualization labels.
- `no_progress`: Disable tqdm progress bar.

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
git clone https://github.com/Surya-Rayala/VisionPipeline-gallery-track.git
cd VisionPipeline-gallery-track
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

`gallery-track-lib` includes a thin wrapper around BoxMOT’s ReID export pipeline.

This tool:
- exports your ReID weights into one or more formats
- collects artifacts under a run folder (`--out-dir` / `--run-name`)
- writes `export_meta.json` with settings and final output paths

## CLI arguments

All exports are collected under a run folder:
- `<out-dir>/<run-name>/...`
- `export_meta.json` is written alongside exported artifacts with settings + final output paths.

### Output organization

- `--out-dir <dir>`: Root folder where export runs are written (default: `models/exports`).
- `--run-name <name>`: Run subfolder name under `--out-dir`. Defaults to the weights file stem.

### Core export arguments

- `--weights <path>`: Path to the source ReID `.pt` weights to export. Required unless using a `--list-*` option.
  - **Custom weights naming note:** If your `weighta` file is a custom weight, prefer naming it so the filename starts with a model name from `--list-reid-models` (e.g., `osnet_custom.pt`).
-- `--include <formats...>`: One or more export formats to generate (default: `torchscript`). Supported: `torchscript`, `onnx`, `openvino`, `engine`, `tflite`.

> **TensorRT (`engine`) export note:** If you include `engine`, you must install **both**:
> 1) a **TensorRT** build that is **compatible with your CUDA toolkit**, and  
> 2) NVIDIA’s **`nvidia-tensorrt`** package.
>
> (CUDA/TensorRT mismatches are the most common cause of export/runtime errors.)
>
> - pip:
>   - `pip install <compatible-tensorrt>`
>   - `pip install nvidia-tensorrt`
> - uv: run `uv sync` first, then:
>   - `uv add <compatible-tensorrt>`
>   - `uv add nvidia-tensorrt`


- `--device <str>`: Device used for export and dummy inference (`cpu`, `mps`, or a CUDA index like `0`) (Note: there is no auto backend support here and have to be manually selected and the exact device to be specified during inferencing).
- `--half`: Enable FP16 where supported (typically GPU exporters).
- `--batch-size <int>`: Dummy batch size used during export (default: `1`). Some backends only support `1`.

### Per-format knobs

- `--optimize`: Optimize TorchScript for mobile (CPU-only).
- `--dynamic`: Enable dynamic shapes where supported (commonly affects ONNX/TensorRT).
- `--simplify`: Run ONNX graph simplification after export.
- `--opset <int>`: ONNX opset version (default: `18`).
- `--verbose`: Enable verbose logging for TensorRT export.

### Listing / discovery

- `--list-formats`: Print supported export formats and exit.
- `--list-reid-models`: Print available ReID architectures (from your BoxMOT install) and exit.
- `--list-reid-weights`: Print known pretrained weight names (from BoxMOT registries, if available) and exit.

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

`gallery-track-lib` includes an optional GUI tool to build a gallery directory by drawing crops on a video.

It creates a directory structure compatible with both trackers:

```
galleries/
  identity_A/
    identity_A_00000.jpg
    identity_A_00001.jpg
  identity_B/
    identity_B_00000.jpg
```

## What the UI does (in plain English)

The gallery builder is a small desktop window that lets you:
- open a video
- pause on frames where your target identity is visible
- draw a bounding box around the person/object you want to add to the gallery
- save that crop into a folder named after the identity

Over time you build a small set of example images per identity (a “gallery”). The trackers then match detections against this gallery to produce `gallery_id`.

### What you’ll see

- A video preview panel (current frame)
- A simple toolbar / buttons for selecting:
  - the input **video**
  - the output **gallery root folder**
- Identity controls:
  - a text field to type an **identity name** (folder name, e.g., `person_A`)
  - a selector/list of existing identities found under the gallery root
  - a button to **add/create** the identity folder if it doesn’t exist
- A counter/indicator showing how many crops have been saved for the selected identity

(Exact layout may vary slightly by OS.)

### Typical workflow

1) Launch the tool.
2) Select the **gallery root** folder (where galleries will be written).
3) Select the **video** file.
4) Type an **identity name** (e.g., `person_A`).
5) Scrub / step through the video to find good frames.
6) Draw a box around the identity and save the crop.
7) Repeat for multiple frames and multiple identities.

### Adding and selecting identities

- **Identity = folder name.** Each identity you create becomes a subfolder under the gallery root.
- To add a new identity:
  1) Type a new name (for example: `person_A`).
  2) Click the **Add/Create** identity button.
  3) The tool creates `<gallery_root>/person_A/` if it doesn’t already exist.
- To switch to an existing identity:
  - Select it from the identity list/dropdown. New crops will be saved into that identity’s folder.

### Drawing boxes (click + drag)

- Pause or scrub to a frame where the identity is clearly visible.
- **Click and hold** on the video frame, **drag** to form a rectangle around the identity, then **release** to finish the box.
- After the box is drawn, click the **Save** / **Save crop** button to write the cropped JPEG into the selected identity folder.

If you draw the wrong box, simply draw a new one and save again (you can delete unwanted images from the identity folder later).

Tips:
- Use a variety of views (front/side), lighting, and distances.
- Avoid heavy blur/occlusion; clean crops work best.

The tool will create the identity subfolders if they don’t exist. Each saved crop is written as a JPEG into the selected identity folder under the gallery root. If you reopen the tool later and select the same gallery root, it will automatically pick up the existing identity folders and let you continue adding more crops to them.

## Troubleshooting: OpenCV crash (common on some macOS/Linux setups)

If the **gallery builder UI** crashes on launch (often due to an OpenCV / Qt / GUI backend conflict), try removing any existing OpenCV wheels and reinstalling the **headless** build.

### pip

```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless
```

### uv

```bash
uv remove opencv-python opencv-contrib-python
uv add opencv-python-headless
```

> Note: The headless build disables OpenCV GUI backends. If you need native OpenCV windows elsewhere, reinstall `opencv-python` instead.


## Paths and defaults

You can start the UI in two ways:

- **Without arguments**: the UI will prompt you to pick the **video** and **gallery root** inside the window.
- **With arguments**: you can pre-fill the paths from the command line.

If you launch without `--gallery-root` and/or `--video`, you must choose them in the UI before you can save crops.

When you choose a gallery root that already contains identity subfolders, the UI will load and list them automatically so you can keep adding crops.

## CLI usage (pip)

Help:

```bash
python -m gallery_track.tools.build_gallery -h
```

Launch with no pre-selected paths (you will choose video + gallery root in the UI):
```bash
python -m gallery_track.tools.build_gallery
```

Launch with initial paths:

```bash
python -m gallery_track.tools.build_gallery \
  --gallery-root galleries \
  --video in.mp4
```
When you pass these flags, the UI starts with the fields pre-filled, but you can still change them inside the app.

## CLI usage (uv)

```bash
uv run python -m gallery_track.tools.build_gallery --gallery-root galleries --video in.mp4
```

---

# License

This project is licensed under the **AGPL-3.0 License**. See `LICENSE`.
