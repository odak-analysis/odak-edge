"""Person tracker: a thin adapter around the unchanged vendored OC-SORT.

Tracking uses position and motion only. No appearance or re-identification model.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import TrackerConfig
from .ocsort import OCSort
from .types import Detection, TrackedPerson


class PersonTracker:
    _open = False  # OC-SORT keeps its track counter on a class, so only one tracker may be open

    def __init__(self, config: TrackerConfig = TrackerConfig()) -> None:
        if PersonTracker._open:
            raise RuntimeError(
                "one PersonTracker per process: OC-SORT keeps its track counter on the class"
            )
        self._ocsort = OCSort(
            det_thresh=config.det_thresh,
            max_age=config.max_age,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold,
            delta_t=config.delta_t,
            asso_func=config.asso_func,
            inertia=config.inertia,
            use_byte=config.use_byte,
        )
        PersonTracker._open = True
        self._closed = False

    def update(self, detections: Sequence[Detection]) -> list[TrackedPerson]:
        """Track one frame. Call it for every frame, also when nothing was detected."""
        valid = [d for d in detections if _is_valid(d)]
        if valid:
            # new float array: the vendored update() divides it in place
            rows = np.array([(*d.box, d.score) for d in valid], dtype=np.float64)
        else:
            rows = np.empty((0, 5))
        tracked = self._ocsort.update(rows, (1, 1), (1, 1))

        used: set[int] = set()
        people: list[TrackedPerson] = []
        for x1, y1, x2, y2, tid in tracked:
            idx = _first_unused_match(valid, (x1, y1, x2, y2), used)
            if idx is None:
                continue
            used.add(idx)
            d = valid[idx]
            people.append(TrackedPerson(
                track_id=int(tid),
                box=(float(x1), float(y1), float(x2), float(y2)),
                score=d.score,
                keypoints=d.keypoints,
            ))
        return people

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            PersonTracker._open = False

    def __enter__(self) -> "PersonTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _is_valid(d: Detection) -> bool:
    x1, y1, x2, y2 = d.box
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2, d.score)):
        return False
    return x2 > x1 and y2 > y1


def _first_unused_match(valid, box, used):
    for i, d in enumerate(valid):
        if i not in used and np.allclose(d.box, box, atol=1e-6):
            return i
    return None
