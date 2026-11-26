"""Debug view: draws temporary track numbers on a frame copy. Never writes to disk."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .types import TrackedPerson

_COLOR = (0, 255, 0)


def draw_tracks(frame: np.ndarray, people: Sequence[TrackedPerson]) -> np.ndarray:
    """Return a copy of ``frame`` with a box and ``#<track_id>`` for each person."""
    out = frame.copy()
    for p in people:
        x1, y1, x2, y2 = (int(round(v)) for v in p.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR, 2)
        cv2.putText(out, f"#{p.track_id}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR, 1, cv2.LINE_AA)
    return out
