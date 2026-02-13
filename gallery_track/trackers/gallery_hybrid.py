from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from scipy.spatial.distance import cdist

import logging

logger = logging.getLogger(__name__)

def _get_reid_auto_backend_cls():
    """Import BoxMOT's ReidAutoBackend across versions.

    We import lazily to avoid hard failures at package import time when BoxMOT
    internal module paths change.
    """
    try:  # BoxMOT newer
        from boxmot.reid.core.auto_backend import ReidAutoBackend  # type: ignore

        return ReidAutoBackend
    except Exception:  # pragma: no cover
        try:  # BoxMOT older
            from boxmot.appearance.reid.auto_backend import ReidAutoBackend  # type: ignore

            return ReidAutoBackend
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Could not import BoxMOT ReidAutoBackend. Install a compatible BoxMOT version."
            ) from e
    
from boxmot.motion.kalman_filters.aabb.xyah_kf import KalmanFilterXYAH
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh, xywh2tlwh, tlwh2xyah
from boxmot.trackers.basetracker import BaseTracker

from .base import BaseTrack, TrackState


class STrack(BaseTrack):
    """Single-object track for GalleryHybridTracker (ByteTrack-style + KF + optional gallery_id)."""

    shared_kalman = KalmanFilterXYAH()

    @property
    def id(self) -> int:
        return getattr(self, "track_id", 0)

    @id.setter
    def id(self, value) -> None:
        self.track_id = int(value)

    def __init__(self, det: np.ndarray, max_obs: int):
        super().__init__()
        self._xywh = xyxy2xywh(det[0:4]).astype(np.float32)
        self.conf = float(det[4])
        self.cls = float(det[5])
        self.det_ind = int(det[6])
        self.max_obs = int(max_obs)

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
        self.history_observations = deque([], maxlen=self.max_obs)

        self.gallery_id: Optional[str] = None
        self.prev_gallery_id: Optional[str] = None

        self._xyxy_cache = None

    def predict(self) -> None:
        mean_state = None if self.mean is None else self.mean.copy()
        if mean_state is None or self.covariance is None:
            return
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        kf = self.kalman_filter or STrack.shared_kalman
        self.mean, self.covariance = kf.predict(mean_state, self.covariance)
        self._xyxy_cache = None

    @staticmethod
    def multi_predict(stracks: List["STrack"]) -> None:
        if not stracks:
            return
        idx = [i for i, st in enumerate(stracks) if (st.mean is not None and st.covariance is not None)]
        if not idx:
            return
        multi_mean = np.asarray([stracks[i].mean.copy() for i in idx])
        multi_cov = np.asarray([stracks[i].covariance for i in idx])
        for j, i in enumerate(idx):
            if stracks[i].state != TrackState.Tracked:
                multi_mean[j][7] = 0
        multi_mean, multi_cov = STrack.shared_kalman.multi_predict(multi_mean, multi_cov)
        for j, i in enumerate(idx):
            stracks[i].mean = multi_mean[j]
            stracks[i].covariance = multi_cov[j]
            stracks[i]._xyxy_cache = None

    def activate(self, kalman_filter, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyah)
        self._xyxy_cache = None
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xyah)
        self._xyxy_cache = None
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    def update(self, new_track: "STrack", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.history_observations.append(self.xyxy)
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xyah)
        self._xyxy_cache = None
        self.state = TrackState.Tracked
        self.is_activated = True
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    @property
    def xywh(self) -> np.ndarray:
        if self.mean is None:
            return self._xywh.copy()
        xc, yc, a, h = self.mean[:4]
        w = a * h
        return np.array([xc, yc, w, h], dtype=np.float32)

    @property
    def tlwh(self) -> np.ndarray:
        return xywh2tlwh(self.xywh)

    @property
    def xyah(self) -> np.ndarray:
        return tlwh2xyah(self.tlwh)

    @property
    def xyxy(self) -> np.ndarray:
        if self._xyxy_cache is not None:
            return self._xyxy_cache.copy()
        xyxy = xywh2xyxy(self.xywh.copy())
        self._xyxy_cache = xyxy
        return xyxy.copy()


class GalleryHybridTracker(BaseTracker):
    """Temporal tracker (ByteTrack-style) with optional periodic gallery ReID assignment.

    Output rows:
      [frame, x1, y1, x2, y2, track_id, score, cls, gallery_id, det_ind]
    """

    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 25,
        frame_rate: int = 30,
        per_class: bool = False,
        max_obs: int = 30,
        reid_weights: Optional[str | Path] = None,
        gallery_path: Optional[str] = None,
        reid_frequency: int = 10,
        gallery_match_threshold: float = 0.25,
        device: str = "cuda",
        half: bool = False,
        models_dir: str | Path = "models",  # kept for CLI compatibility; unused in hybrid tracker
    ):
        super().__init__(per_class=per_class)
        self.active_tracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.track_buffer = int(track_buffer)
        self.per_class = bool(per_class)
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.det_thresh = float(track_thresh)

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYAH()
        self.max_obs = int(max_obs)

        # ReID config
        self.reid_frequency = int(reid_frequency)
        self.gallery_match_threshold = float(gallery_match_threshold)
        self.device = str(device)

        self.reid_model = None
        self.gallery_path = gallery_path
        self._gal_X = None
        self._gal_ids = None
        self._gal_slices = None
        self._gal_X_cuda = None
        self._warned_batch1_fallback = False

        # Load ReID + gallery if provided (hybrid works without ReID)
        if reid_weights:
            w = Path(str(reid_weights)).expanduser()
            if w.exists():
                ReidAutoBackend = _get_reid_auto_backend_cls()
                self.reid_model = ReidAutoBackend(weights=w, device=device, half=half).model
            else:
                self.reid_model = None

        if self.reid_model is not None and self.gallery_path:
            self._load_gallery_embeddings()

    def _load_gallery_embeddings(self) -> None:
        gallery_root = Path(self.gallery_path)
        if not gallery_root.exists():
            self._gal_X = None
            self._gal_ids = None
            self._gal_slices = None
            self._gal_X_cuda = None
            return

        gal_X: List[np.ndarray] = []
        gal_ids: List[str] = []
        gal_slices: List[slice] = []
        offset = 0

        for subfolder in sorted(gallery_root.iterdir()):
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
                f = self.reid_model.get_features(bbox, img)
                if f is not None and len(f) > 0:
                    feats.append(f[0])

            if feats:
                n = len(feats)
                gal_ids.append(identity)
                gal_X.extend(feats)
                gal_slices.append(slice(offset, offset + n))
                offset += n

        if not gal_X:
            self._gal_X = None
            self._gal_ids = None
            self._gal_slices = None
            self._gal_X_cuda = None
            return

        X = np.vstack(gal_X).astype(np.float32, copy=False)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        self._gal_X = X
        self._gal_ids = gal_ids
        self._gal_slices = gal_slices

        self._gal_X_cuda = None
        if self.device == "cuda":
            try:
                import torch

                if torch.cuda.is_available():
                    self._gal_X_cuda = torch.from_numpy(self._gal_X).to("cuda", non_blocking=True)
            except Exception:
                self._gal_X_cuda = None

    def _extract_features_batched(self, img_bgr: np.ndarray, tracks: List[STrack]) -> List[Optional[np.ndarray]]:
        if not tracks:
            return []

        img_bgr = np.ascontiguousarray(img_bgr)
        # (N, 4) xyxy boxes
        boxes = np.array([t.xyxy for t in tracks], dtype=np.float32).reshape(-1, 4)
        boxes = np.ascontiguousarray(boxes, dtype=np.float32)

        # Try a single batched call first (best performance on PyTorch backend)
        try:
            feats = self.reid_model.get_features(boxes, img_bgr)
            if feats is None:
                return [None] * len(tracks)
            out: List[Optional[np.ndarray]] = []
            for i in range(len(tracks)):
                out.append(feats[i] if i < len(feats) else None)
            return out
        except Exception as e:
            # Fallback: some backends/exports (notably ONNX with fixed batch) may reject
            # variable batch sizes. In that case, run per-track (batch=1) extraction.
            #
            # Emit a visible warning once per tracker instance so users understand the perf hit.
            if not getattr(self, "_warned_batch1_fallback", False):
                self._warned_batch1_fallback = True
                logger.warning(
                    "[gallery_track] ReID batched feature extraction failed; falling back to per-box (batch=1). "
                    "This is slower and may indicate a fixed-batch export (e.g., ONNX). Error: %s",
                    e,
                )

            out: List[Optional[np.ndarray]] = []
            for t in tracks:
                box = np.ascontiguousarray(np.array(t.xyxy, dtype=np.float32).reshape(1, 4), dtype=np.float32)
                try:
                    f = self.reid_model.get_features(box, img_bgr)
                except Exception:
                    f = None
                out.append(f[0] if f is not None and len(f) > 0 else None)
            return out

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None) -> np.ndarray:  # type: ignore[override]
        self.check_inputs(dets, img)

        # Append det_ind
        n, m = dets.shape
        dets_ext = np.empty((n, m + 1), dtype=dets.dtype)
        dets_ext[:, :m] = dets
        dets_ext[:, m] = np.arange(n, dtype=dets.dtype)
        dets = dets_ext

        self.frame_count += 1

        activated: List[STrack] = []
        refind: List[STrack] = []
        lost: List[STrack] = []
        removed: List[STrack] = []

        scores = dets[:, 4]
        remain = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        second = np.logical_and(inds_low, inds_high)

        dets_second = dets[second]
        dets = dets[remain]

        detections = [STrack(det, max_obs=self.max_obs) for det in dets]

        unconfirmed = [t for t in self.active_tracks if not t.is_activated]
        tracked = [t for t in self.active_tracks if t.is_activated]

        # pool = tracked + lost
        pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(pool)

        if len(pool) == 0 or len(detections) == 0:
            matches, u_track, u_det = [], list(range(len(pool))), list(range(len(detections)))
        else:
            dists = iou_distance(pool, detections)
            dists = fuse_score(dists, detections)
            matches, u_track, u_det = linear_assignment(dists, thresh=self.match_thresh)

        for it, idet in matches:
            tr = pool[it]
            det = detections[idet]
            if tr.state == TrackState.Tracked:
                tr.update(det, self.frame_count)
                activated.append(tr)
            else:
                tr.re_activate(det, self.frame_count, new_id=False)
                refind.append(tr)

        # second association (low conf)
        detections_second = [STrack(det2, max_obs=self.max_obs) for det2 in dets_second]
        r_tracked = [pool[i] for i in u_track if pool[i].state == TrackState.Tracked]

        if len(r_tracked) == 0 or len(detections_second) == 0:
            matches2, u_track2, u_det2 = [], list(range(len(r_tracked))), list(range(len(detections_second)))
        else:
            d2 = iou_distance(r_tracked, detections_second)
            matches2, u_track2, u_det2 = linear_assignment(d2, thresh=0.5)

        for it, idet in matches2:
            tr = r_tracked[it]
            det = detections_second[idet]
            if tr.state == TrackState.Tracked:
                tr.update(det, self.frame_count)
                activated.append(tr)
            else:
                tr.re_activate(det, self.frame_count, new_id=False)
                refind.append(tr)

        for it in u_track2:
            tr = r_tracked[it]
            if tr.state != TrackState.Lost:
                tr.mark_lost()
                lost.append(tr)

        # unconfirmed handling
        dets_tmp = [detections[i] for i in u_det]
        if len(unconfirmed) == 0 or len(dets_tmp) == 0:
            matches3, u_unconf, u_det3 = [], list(range(len(unconfirmed))), list(range(len(dets_tmp)))
        else:
            d3 = iou_distance(unconfirmed, dets_tmp)
            d3 = fuse_score(d3, dets_tmp)
            matches3, u_unconf, u_det3 = linear_assignment(d3, thresh=0.7)

        for it, idet in matches3:
            unconfirmed[it].update(dets_tmp[idet], self.frame_count)
            activated.append(unconfirmed[it])

        for it in u_unconf:
            tr = unconfirmed[it]
            tr.mark_removed()
            removed.append(tr)

        # init new tracks
        for inew in u_det3:
            tr = dets_tmp[inew]
            if tr.conf < self.det_thresh:
                continue
            tr.activate(self.kalman_filter, self.frame_count)
            activated.append(tr)

        for tr in self.lost_stracks:
            if self.frame_count - tr.end_frame > self.max_time_lost:
                tr.mark_removed()
                removed.append(tr)

        self.active_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]
        self.active_tracks = joint_stracks(self.active_tracks, activated)
        self.active_tracks = joint_stracks(self.active_tracks, refind)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(self.active_tracks, self.lost_stracks)

        # Periodic gallery matching
        if img is not None and self.reid_model is not None and self._gal_X is not None and self._gal_ids:
            if self.frame_count % self.reid_frequency == 0:
                tracks = list(self.active_tracks)
                feats = self._extract_features_batched(img, tracks)

                valid_idx = [i for i, f in enumerate(feats) if f is not None]
                assigned_gallery = set()

                if valid_idx:
                    T = np.vstack([feats[i] for i in valid_idx]).astype(np.float32, copy=False)
                    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)

                    use_gpu = self.device == "cuda"
                    try:
                        import torch
                        use_gpu = use_gpu and torch.cuda.is_available()
                    except Exception:
                        use_gpu = False

                    if use_gpu:
                        import torch
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
                    new_assign = {}
                    for r, c in zip(row_ind, col_ind):
                        if reduced[r, c] < thr:
                            ti = valid_idx[r]
                            tr = tracks[ti]
                            identity = self._gal_ids[c]
                            new_assign[tr] = identity
                            assigned_gallery.add(identity)

                    for tr in self.active_tracks:
                        if tr in new_assign:
                            tr.gallery_id = new_assign[tr]
                            tr.prev_gallery_id = new_assign[tr]
                        else:
                            if tr.prev_gallery_id is not None and tr.prev_gallery_id not in assigned_gallery:
                                tr.gallery_id = tr.prev_gallery_id
                                assigned_gallery.add(tr.gallery_id)
                            else:
                                tr.gallery_id = None

        # Build outputs (object dtype for mixed types)
        output_tracks = [t for t in self.active_tracks if t.is_activated]
        k = len(output_tracks)

        outputs = np.empty((k, 9), dtype=object)
        for i, t in enumerate(output_tracks):
            bb = t.xyxy
            outputs[i, 0] = self.frame_count - 1
            outputs[i, 1] = float(bb[0])
            outputs[i, 2] = float(bb[1])
            outputs[i, 3] = float(bb[2])
            outputs[i, 4] = float(bb[3])
            outputs[i, 5] = str(t.id)     # numeric track id as string
            outputs[i, 6] = float(t.conf) # score/conf
            outputs[i, 7] = float(t.cls)  # cls
            outputs[i, 8] = t.gallery_id  # may be None

        det_inds = np.fromiter((t.det_ind for t in output_tracks), dtype=object, count=k).reshape(k, 1)
        return np.concatenate([outputs, det_inds], axis=1)


def joint_stracks(a: List[STrack], b: List[STrack]) -> List[STrack]:
    exists = {}
    res: List[STrack] = []
    for t in a:
        exists[t.id] = 1
        res.append(t)
    for t in b:
        if not exists.get(t.id, 0):
            exists[t.id] = 1
            res.append(t)
    return res


def sub_stracks(a: List[STrack], b: List[STrack]) -> List[STrack]:
    m = {t.id: t for t in a}
    for t in b:
        if t.id in m:
            del m[t.id]
    return list(m.values())


def remove_duplicate_stracks(a: List[STrack], b: List[STrack]):
    if not a or not b:
        return a, b
    pdist = iou_distance(a, b)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = a[p].frame_id - a[p].start_frame
        timeq = b[q].frame_id - b[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(a) if i not in dupa]
    resb = [t for i, t in enumerate(b) if i not in dupb]
    return resa, resb