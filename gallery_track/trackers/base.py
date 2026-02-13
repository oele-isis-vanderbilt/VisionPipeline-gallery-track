from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np


class TrackState:
    """Enumeration of possible tracking states."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for tracking objects (instance-safe).

    Provides a consistent API for track lifecycle management and ID assignment.
    """

    _count: int = 0

    def __init__(self) -> None:
        self.track_id: int = 0
        self.is_activated: bool = False
        self.state: int = TrackState.New

        self.history: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.features: List[np.ndarray] = []
        self.curr_feature: Optional[np.ndarray] = None

        self.conf: float = 0.0
        self.start_frame: int = 0
        self.frame_id: int = 0
        self.time_since_update: int = 0

        # multi-camera location placeholder
        self.location: Tuple[float, float] = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    @staticmethod
    def clear_count() -> None:
        BaseTrack._count = 0