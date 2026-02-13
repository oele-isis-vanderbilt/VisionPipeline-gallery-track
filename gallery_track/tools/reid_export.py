from __future__ import annotations

"""gallery_track.tools.reid_export

Thin wrapper around BoxMOT's ReID export pipeline.

Adds:
  - --out-dir / --run-name to collect exports under a run folder
  - export_meta.json with settings + output paths

This tool delegates exports to BoxMOT's ReID export pipeline.

CLI:
    python -m gallery_track.tools.reid_export ...
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

SUPPORTED_EXPORT_FORMATS = ["torchscript", "onnx", "openvino", "engine", "tflite"]

# --- BoxMOT export pipeline (current BoxMOT layout; fallback to legacy) ---
try:
    from boxmot.engine.export import (
        setup_model as bm_setup_model,
        create_export_tasks as bm_create_export_tasks,
        perform_exports as bm_perform_exports,
    )
    from boxmot.reid.core.registry import ReIDModelRegistry
except Exception:  # pragma: no cover
    # Legacy BoxMOT (< reid.core/* layout)
    from boxmot.appearance.reid.export import (  # type: ignore
        setup_model as bm_setup_model,
        create_export_tasks as bm_create_export_tasks,
        perform_exports as bm_perform_exports,
    )
    from boxmot.appearance.reid.registry import ReIDModelRegistry  # type: ignore

# Optional registries for listing (may not exist in all BoxMOT versions)
try:  # newer
    from boxmot.reid.core.factory import MODEL_FACTORY  # type: ignore
except Exception:  # pragma: no cover
    try:
        from boxmot.appearance.reid.factory import MODEL_FACTORY  # type: ignore
    except Exception:
        MODEL_FACTORY = {}

try:  # newer
    from boxmot.reid.config import TRAINED_URLS  # type: ignore
except Exception:  # pragma: no cover
    try:
        from boxmot.appearance.reid.config import TRAINED_URLS  # type: ignore
    except Exception:
        TRAINED_URLS = {}

from boxmot.utils import logger as LOGGER


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export a ReID model via BoxMOT, with run-folder & metadata",
    )

    # Output organization
    ap.add_argument("--out-dir", type=Path, default=Path("models/exports"), help="Root folder for exports")
    ap.add_argument("--run-name", type=str, default=None, help="Subfolder under out-dir (defaults to weights stem)")

    # Core export args (aligned to BoxMOT engine/export.py expectations)
    ap.add_argument("--weights", type=str, required=False, help="ReID weights path")
    ap.add_argument(
        "--include",
        nargs="+",
        default=["torchscript"],
        help=f"Which formats to export: {', '.join(SUPPORTED_EXPORT_FORMATS)}",
    )
    ap.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps/0)")
    ap.add_argument("--half", action="store_true", help="Enable FP16 where supported")
    ap.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used for export")

    # Per-format knobs used by BoxMOT exporters
    ap.add_argument("--optimize", action="store_true", help="Optimize TorchScript for mobile")
    ap.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes where supported")
    ap.add_argument("--simplify", action="store_true", help="Simplify ONNX")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging for TensorRT")

    # Listing helpers
    ap.add_argument("--list-reid-models", action="store_true", help="List available ReID architectures and exit")
    ap.add_argument("--list-reid-weights", action="store_true", help="List known pretrained ReID weight names and exit")
    ap.add_argument("--list-formats", action="store_true", help="List supported export formats and exit")

    return ap.parse_args()


from ..core.reid_registry import resolve_reid_weights


def list_reid_architectures() -> List[str]:
    try:
        return sorted(getattr(MODEL_FACTORY, "keys", lambda: [])())
    except Exception:
        try:
            return sorted(list(MODEL_FACTORY.keys()))
        except Exception:
            return []


def list_downloadable_reid_weights() -> List[str]:
    try:
        return sorted(list(TRAINED_URLS.keys()))
    except Exception:
        return []


def _gather_outputs(outputs: Dict[str, Path], run_dir: Path) -> Dict[str, Path]:
    """Collect exporter results into run_dir; return final paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    moved: Dict[str, Path] = {}
    for fmt, src in outputs.items():
        if src is None:
            continue
        src = Path(src)
        if not src.exists():
            continue
        dst = run_dir / src.name
        if src.resolve() == dst.resolve():
            moved[fmt] = dst
            continue
        # Prefer atomic rename; fallback to copy
        try:
            src.replace(dst)
        except Exception:
            # directory or cross-device: do a manual copy
            if src.is_dir():
                # minimal recursive copy
                if dst.exists():
                    # best-effort cleanup
                    for p in sorted(dst.glob("**/*"), reverse=True):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                dst.mkdir(parents=True, exist_ok=True)
                for p in src.glob("**/*"):
                    if p.is_dir():
                        (dst / p.relative_to(src)).mkdir(parents=True, exist_ok=True)
                    else:
                        (dst / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
                        (dst / p.relative_to(src)).write_bytes(p.read_bytes())
            else:
                dst.write_bytes(src.read_bytes())
        moved[fmt] = dst
    return moved


def main() -> None:
    args = _parse_args()

    # Early list exits
    if getattr(args, "list_formats", False):
        print("Supported export formats:")
        for f in SUPPORTED_EXPORT_FORMATS:
            print("  -", f)
        return
    if getattr(args, "list_reid_models", False):
        print("Available ReID architectures:")
        for n in list_reid_architectures():
            print("  -", n)
        return
    if getattr(args, "list_reid_weights", False):
        print("Downloadable ReID .pt weights:")
        for n in list_downloadable_reid_weights():
            print("  -", n)
        return
    if args.weights is None:
        print(
            "reid_export.py: error: --weights is required unless using --list-reid-models, --list-reid-weights, or --list-formats"
        )
        raise SystemExit(2)

    t0 = time.time()

    # Resolve weights
    resolved = resolve_reid_weights(args.weights, models_dir="models", fatal=True)
    if resolved is None or not Path(resolved).exists():
        print(f"reid_export.py: error: Weights not found or unresolved: {args.weights}")
        raise SystemExit(2)
    args.weights = Path(resolved)

    # Delegate to BoxMOT's setup/export pipeline
    model, dummy = bm_setup_model(args)

    with torch.no_grad():
        out = model(dummy)
    out_tensor = out[0] if isinstance(out, tuple) else out
    size_mb = args.weights.stat().st_size / (1024.0**2)
    LOGGER.info(f"\nStarting from {args.weights} with output shape {tuple(out_tensor.shape)} ({size_mb:.1f} MB)")

    export_tasks = bm_create_export_tasks(args, model, dummy)
    raw_outputs = bm_perform_exports(export_tasks)  # dict[str, str|Path]

    raw_outputs = raw_outputs or {}
    out_map: Dict[str, Path] = {}
    for k, v in raw_outputs.items():
        if v:
            out_map[k] = Path(v)
    run_name = args.run_name or args.weights.stem
    run_dir = Path(args.out_dir) / run_name
    final_outputs = _gather_outputs(out_map, run_dir)

    # Save export metadata
    meta: Dict[str, Any] = {
        "schema_version": "export-v1",
        "source_weights": str(args.weights),
        "model_name": ReIDModelRegistry.get_model_name(args.weights),
        "nr_classes": ReIDModelRegistry.get_nr_classes(args.weights),
        "device": str(getattr(args, "device", "")),
        "half": bool(getattr(args, "half", False)),
        "dynamic": bool(getattr(args, "dynamic", False)),
        "simplify": bool(getattr(args, "simplify", False)),
        "opset": int(getattr(args, "opset", 0)),
        "imgsz": [],  # removed from args, so empty list
        "batch_size": int(getattr(args, "batch_size", 1)),
        "include": list(getattr(args, "include", [])),
        "elapsed_sec": float(max(0.0, time.time() - t0)),
        "outputs": {k: str(v) for k, v in final_outputs.items()},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "export_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if final_outputs:
        LOGGER.info(
            f"\nExport complete. Results saved to {run_dir.resolve()}\n"
            f"Formats: {', '.join(final_outputs.keys())}\n"
            f"Metadata: {str(meta_path.resolve())}\n"
            f"Visualize: https://netron.app"
        )


if __name__ == "__main__":
    main()