from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
from typing import List, Optional, Tuple

import os
import shutil
import tempfile
from urllib.request import urlopen, Request

# BoxMOT registries (graceful fallback across BoxMOT versions)
try:
    # BoxMOT newer
    from boxmot.reid.config import TRAINED_URLS  # {weight_filename: url}
    from boxmot.reid.factory import MODEL_FACTORY  # {arch_name: ctor}
except Exception:  # pragma: no cover
    try:
        # BoxMOT older
        from boxmot.appearance.reid.config import TRAINED_URLS  # type: ignore
        from boxmot.appearance.reid.factory import MODEL_FACTORY  # type: ignore
    except Exception:  # pragma: no cover
        TRAINED_URLS = {}
        MODEL_FACTORY = {}


# Keep this list in sync with BoxMOT's accepted suffixes for auto-backend.
# We do *not* enforce suffix validation here; BoxMOT may still load other formats.
_ACCEPT_SUFFIXES: Tuple[str, ...] = (
    ".pt",
    ".torchscript",
    ".onnx",
    "_openvino_model",
    ".engine",
    ".tflite",
)


def supported_reid_architectures() -> List[str]:
    """List supported ReID model architectures (backbones) reported by BoxMOT."""
    try:
        return sorted(MODEL_FACTORY.keys())
    except Exception:
        return []


def downloadable_reid_weights() -> List[str]:
    """List known downloadable ReID weight filenames from BoxMOT.

    The list comes from BoxMOT's `TRAINED_URLS` registry (filename -> URL).
    """
    try:
        return sorted(TRAINED_URLS.keys())
    except Exception:
        return []


def accepted_weight_suffixes() -> Tuple[str, ...]:
    """Expose accepted suffixes for CLI help text."""
    return _ACCEPT_SUFFIXES


def _download_weight_from_registry(*, filename: str, models_dir: Path) -> Path:
    """Download a known weight file into models_dir using BoxMOT's TRAINED_URLS.

    Only downloads when `filename` exists as a key in TRAINED_URLS.
    """
    url = None
    try:
        url = TRAINED_URLS.get(filename)  # type: ignore[attr-defined]
    except Exception:
        url = None

    if not url:
        raise FileNotFoundError(f"[gallery_track] No downloadable entry for: '{filename}'")

    models_dir.mkdir(parents=True, exist_ok=True)
    dst = (models_dir / filename).resolve()
    if dst.exists() and dst.is_file():
        return dst

    # Download to a temp file first, then move into place atomically.
    fd, tmp_path = tempfile.mkstemp(prefix=f"{filename}.", suffix=".tmp", dir=str(models_dir))
    os.close(fd)
    tmp = Path(tmp_path)

    try:
        req = Request(url, headers={"User-Agent": "gallery-track-lib"})
        with urlopen(req, timeout=60) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dst)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    return dst


def _normalize_weight_name(name: str) -> str:
    """Normalize a weights identifier to a likely filename.

    - If user passes a filename with a suffix: unchanged
    - If user passes an OpenVINO token containing `_openvino_model`: unchanged
    - Else: append `.pt`

    Examples:
      - 'osnet_x0_25_msmt17' -> 'osnet_x0_25_msmt17.pt'
      - 'osnet_x0_25_msmt17.pt' -> unchanged
      - 'something_openvino_model' -> unchanged
    """
    s = name.strip()
    if not s:
        return s
    if "_openvino_model" in s:
        return s
    p = Path(s)
    if p.suffix:
        return p.name
    return f"{p.name}.pt"


def _unknown_name_message(name: str, *, models_dir: str | Path) -> str:
    keys = downloadable_reid_weights()
    hints = get_close_matches(name, keys, n=6, cutoff=0.4) if keys else []

    md = str(Path(models_dir))
    msg = (
        f"[gallery_track] ReID weights not found: '{name}'.\n"
        f"Tried:\n"
        f"  - as a local file path\n"
        f"  - under models-dir: {md}/{name}\n"
        f"\n"
        f"Tips:\n"
        f"  - Pass an explicit weights file path.\n"
        f"  - Or place the file under --models-dir and pass its name.\n"
        f"  - If your BoxMOT install exposes TRAINED_URLS, passing a known weight name will auto-download into --models-dir.\n"
        f"  - To see known BoxMOT downloadable names (if available):\n"
        f"      python -m gallery_track.cli.track_video --list-reid-weights\n"
    )
    if hints:
        msg += "  - Closest names:\n" + "\n".join([f"      * {h}" for h in hints]) + "\n"
    return msg


def resolve_reid_weights(
    weights_or_name: Optional[str | Path],
    *,
    models_dir: str | Path = "models",
) -> Optional[Path]:
    """Resolve ReID weights.

    Accepts:
      - an existing local file path
      - a weights *name* that is expected to exist under `models_dir`

    If the requested filename exists in BoxMOT's `TRAINED_URLS` registry, this
    function will auto-download it into `models_dir` and return the local path.

    For safety, arbitrary URLs are not accepted here: only registry-known names.
    """
    if not weights_or_name:
        return None

    raw = str(weights_or_name).strip()
    if not raw:
        return None

    p = Path(raw).expanduser()

    # 1) Direct local path
    if p.exists() and p.is_file():
        return p.resolve()

    # 2) Name or relative path under models_dir
    #    - normalize filename (adds .pt when no suffix)
    #    - also allow user-provided relative paths (subfolders)
    md = Path(models_dir).expanduser()

    # If the user passed a relative path like "reid/osnet_x0_25_msmt17.pt",
    # try that as-is under models_dir.
    cand_rel = (md / p).resolve()
    if cand_rel.exists() and cand_rel.is_file():
        return cand_rel

    # Try normalized filename under models_dir.
    fname = _normalize_weight_name(p.name)
    cand_name = (md / fname).resolve()
    if cand_name.exists() and cand_name.is_file():
        return cand_name

    # 3) Auto-download if this filename is known to BoxMOT.
    try:
        if fname in downloadable_reid_weights():
            return _download_weight_from_registry(filename=fname, models_dir=md)
    except Exception:
        # Fall through to a helpful error message.
        pass

    # Not found
    raise FileNotFoundError(_unknown_name_message(fname, models_dir=models_dir))


def infer_reid_arch_from_weights(weights_path_or_name: Optional[str | Path]) -> Optional[str]:
    """Best-effort inference of model architecture from weights filename.

    Matches any MODEL_FACTORY key that appears as a prefix in the filename.
    Prefers the longest match (more specific).
    """
    if not weights_path_or_name:
        return None

    name = Path(str(weights_path_or_name)).name.lower()
    arches = supported_reid_architectures()
    if not arches:
        return None

    for arch in sorted(arches, key=len, reverse=True):
        a = arch.lower()
        if name.startswith(a) or f"{a}_" in name:
            return arch

    return None