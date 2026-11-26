"""Data types passed in and out of the person tracker."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """One detected person in one frame. Box is x1, y1, x2, y2 in pixels."""

    box: Box
    score: float
    keypoints: np.ndarray | None = None  # K x 3: x, y, confidence


@dataclass(frozen=True)
class TrackedPerson:
    """A detection with a temporary track number.

    The number is valid only within one camera and one run. It is never stored.
    """

    track_id: int
    box: Box
    score: float
    keypoints: np.ndarray | None
