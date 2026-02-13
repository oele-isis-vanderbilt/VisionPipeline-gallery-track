from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from scipy.spatial.distance import cdist

import logging

logger = logging.getLogger(__name__)


# Lazy import helper for BoxMOT's ReidAutoBackend
def _get_reid_auto_backend_cls():
    """Return BoxMOT's ReidAutoBackend class across BoxMOT versions."""
    try:
        from boxmot.reid.core.auto_backend import ReidAutoBackend  # BoxMOT newer
        return ReidAutoBackend
    except Exception:  # pragma: no cover
        from boxmot.appearance.reid.auto_backend import ReidAutoBackend  # BoxMOT older
        return ReidAutoBackend




class GalleryOnlyTracker:
    """Frame-wise *gallery-only* tracker (no temporal association).

    Output rows match:
      [frame, x1, y1, x2, y2, track_id, score, cls, gallery_id, det_ind]
    where track_id is a numeric per-detection identifier (as a string).
    """

    def __init__(
        self,
        reid_weights: str | Path | None,
        gallery_path: str | Path | None,
        gallery_match_threshold: float = 0.25,
        device: str = "cuda",
        half: bool = False,
        track_thresh: float = 0.0,
        models_dir: str | Path = "models",
    ) -> None:
        if reid_weights is None:
            raise ValueError("gallery_only tracker requires `reid_weights`.")
        if gallery_path is None:
            raise ValueError("gallery_only tracker requires `gallery_path` (gallery directory).")

        resolved = Path(str(reid_weights)).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        resolved = resolved.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"ReID weights file not found: {resolved}")

        self.device: str = device
        self.gallery_match_threshold: float = float(gallery_match_threshold)
        self.track_thresh: float = float(track_thresh)
        self.frame_count: int = 0
        self._warned_batch1_fallback: bool = False

        ReidAutoBackend = _get_reid_auto_backend_cls()
        self.model = ReidAutoBackend(weights=resolved, device=device, half=half).model

        self.gallery_path = str(Path(gallery_path).expanduser().resolve())
        self._gal_X: Optional[np.ndarray] = None
        self._gal_ids: Optional[List[str]] = None
        self._gal_slices: Optional[List[slice]] = None
        self._gal_X_cuda = None  # optional cached GPU copy

        self._load_gallery_embeddings()

    def _load_gallery_embeddings(self) -> None:
        root = Path(self.gallery_path)
        if not root.exists():
            raise FileNotFoundError(f"Gallery path does not exist: {root}")

        gallery_dict: dict[str, List[np.ndarray]] = {}
        for subfolder in sorted(root.iterdir()):
            if not subfolder.is_dir():
                continue
            identity = subfolder.name
            feats: List[np.ndarray] = []
            for img_path in sorted(subfolder.glob("*.*")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w, _ = img.shape
                bbox = np.ascontiguousarray(np.array([[0, 0, w, h]], dtype=np.float32))
                img = np.ascontiguousarray(img)
                f = self.model.get_features(bbox, img)
                if f is not None and len(f) > 0:
                    feats.append(f[0])
            if feats:
                gallery_dict[identity] = feats

        if not gallery_dict:
            self._gal_X = None
            self._gal_ids = None
            self._gal_slices = None
            self._gal_X_cuda = None
            return

        gal_X: List[np.ndarray] = []
        gal_ids: List[str] = []
        gal_slices: List[slice] = []
        offset = 0
        for identity, feats in gallery_dict.items():
            n = len(feats)
            gal_ids.append(identity)
            gal_X.extend(feats)
            gal_slices.append(slice(offset, offset + n))
            offset += n

        X = np.vstack(gal_X).astype(np.float32, copy=False)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self._gal_X = X
        self._gal_ids = gal_ids
        self._gal_slices = gal_slices

        # Optional GPU cache
        self._gal_X_cuda = None
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    self._gal_X_cuda = torch.from_numpy(self._gal_X).to("cuda", non_blocking=True)
            except Exception:
                self._gal_X_cuda = None
                
    def _get_features_safe(self, boxes: np.ndarray, img_bgr: np.ndarray):
        """
        Try batched get_features(boxes, img) first.
        If backend/export only supports batch=1 (common for fixed ONNX), fall back to per-box calls.
        Returns a list aligned to boxes rows (may contain None).
        """
        try:
            feats = self.model.get_features(boxes, img_bgr)
            if feats is None:
                return [None] * len(boxes)
            # normalize to list
            return [feats[i] if i < len(feats) else None for i in range(len(boxes))]
        except Exception as e:
            # Fallback: per-box inference (batch=1)
            # Emit a visible warning once per tracker instance so users understand the perf hit.
            if not getattr(self, "_warned_batch1_fallback", False):
                self._warned_batch1_fallback = True
                logger.warning(
                    "[gallery_track] ReID batched feature extraction failed; falling back to per-box (batch=1). "
                    "This is slower and may indicate a fixed-batch export (e.g., ONNX). Error: %s",
                    e,
                )

            out = []
            for i in range(len(boxes)):
                b = np.ascontiguousarray(boxes[i : i + 1], dtype=np.float32)
                try:
                    f = self.model.get_features(b, img_bgr)
                    out.append(f[0] if f is not None and len(f) > 0 else None)
                except Exception:
                    out.append(None)
            return out
        
    def _extract_features_for_detections(self, img_bgr, dets_array):
        if dets_array is None or len(dets_array) == 0:
            return []
        boxes = dets_array[:, :4].astype(np.float32, copy=False).reshape(-1, 4)
        boxes = np.ascontiguousarray(boxes, dtype=np.float32)
        img_bgr = np.ascontiguousarray(img_bgr)
        return self._get_features_safe(boxes, img_bgr)

    def update(self, dets: np.ndarray, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("GalleryOnlyTracker.update requires the current frame image `img`.")

        self.frame_count += 1

        if dets is None or dets.size == 0:
            return np.zeros((0, 10), dtype=object)

        dets = np.asarray(dets)
        if dets.ndim != 2 or dets.shape[1] < 6:
            raise ValueError("gallery_only expects dets shape (N, >=6) with [x1,y1,x2,y2,score,cls].")

        # Append det_ind
        n, m = dets.shape
        dets_ext = np.empty((n, m + 1), dtype=dets.dtype)
        dets_ext[:, :m] = dets
        dets_ext[:, m] = np.arange(n, dtype=dets.dtype)
        dets = dets_ext

        # score filtering
        scores = dets[:, 4].astype(float)
        keep = scores >= self.track_thresh
        dets = dets[keep]
        if dets.size == 0:
            return np.zeros((0, 10), dtype=object)

        # No gallery available => no matches
        if self._gal_X is None or not self._gal_ids:
            outputs = np.empty((len(dets), 10), dtype=object)
            for i, d in enumerate(dets):
                x1, y1, x2, y2, score, cls, det_ind = d[:7]
                outputs[i] = [
                    self.frame_count - 1,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    str(int(det_ind)),  # track_id
                    float(score),
                    float(cls),
                    None,               # gallery_id
                    int(det_ind),
                ]
            return outputs

        # Extract features
        det_features = self._extract_features_for_detections(img, dets)

        num_dets = len(dets)
        assigned_ids: List[Optional[str]] = [None] * num_dets

        valid_idx = [i for i, f in enumerate(det_features) if f is not None]
        if valid_idx:
            T = np.vstack([det_features[i] for i in valid_idx]).astype(np.float32, copy=False)
            T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)

            # cosine distance to all gallery vectors, then reduce per identity via min slice
            use_gpu = self.device == "cuda"
            try:
                import torch  # type: ignore[import]
                use_gpu = use_gpu and torch.cuda.is_available()
            except Exception:
                use_gpu = False

            if use_gpu:
                import torch  # type: ignore[import]

                t = torch.from_numpy(T).to("cuda", non_blocking=True)
                g = self._gal_X_cuda
                if g is None or g.shape[0] != self._gal_X.shape[0] or g.device.type != "cuda":
                    g = torch.from_numpy(self._gal_X).to("cuda", non_blocking=True)
                    self._gal_X_cuda = g
                dist_all = (1.0 - (t @ g.T)).float()

                Tn = dist_all.shape[0]
                G = len(self._gal_ids)
                INF = np.float32(1e6)
                reduced_t = torch.full((Tn, G), INF, dtype=torch.float32, device=dist_all.device)
                for j, sl in enumerate(self._gal_slices):
                    if sl.stop > sl.start:
                        reduced_t[:, j] = dist_all[:, sl].min(dim=1).values
                reduced = reduced_t.cpu().numpy().astype(np.float32, copy=False)
            else:
                dist_all = cdist(T, self._gal_X, metric="cosine").astype(np.float32, copy=False)
                Tn = dist_all.shape[0]
                G = len(self._gal_ids)
                INF = np.float32(1e6)
                reduced = np.full((Tn, G), INF, dtype=np.float32)
                for j, sl in enumerate(self._gal_slices):
                    if sl.stop > sl.start:
                        reduced[:, j] = dist_all[:, sl].min(axis=1)

            thr = np.float32(self.gallery_match_threshold)
            INF = np.float32(1e6)
            reduced[reduced >= thr] = INF

            row_ind, col_ind = linear_sum_assignment(reduced)
            for r, c in zip(row_ind, col_ind):
                if reduced[r, c] < thr:
                    det_i = valid_idx[r]
                    assigned_ids[det_i] = self._gal_ids[c]

        outputs = np.empty((num_dets, 10), dtype=object)
        for i, d in enumerate(dets):
            x1, y1, x2, y2, score, cls, det_ind = d[:7]
            outputs[i] = [
                self.frame_count - 1,
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                str(int(det_ind)),        # track_id
                float(score),
                float(cls),
                assigned_ids[i],          # gallery_id
                int(det_ind),
            ]
        return outputs