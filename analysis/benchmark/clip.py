"""Read a test clip (or one segment of it) frame by frame, in memory only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class ClipInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def _open(path: Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open video: {path}")
    return cap


def read_clip_info(path: Path) -> ClipInfo:
    cap = _open(path)
    try:
        return ClipInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def segment_frames(path: Path, start_s: float = 0.0, end_s: float | None = None) -> Iterator[np.ndarray]:
    """Frames from ``start_s`` up to (not including) ``end_s`` seconds; the whole clip by default.

    Arguments are checked when called, not when the first frame is read.
    """
    if start_s < 0:
        raise ValueError(f"segment start must be >= 0, got {start_s}")
    if end_s is not None and end_s <= start_s:
        raise ValueError(f"segment end ({end_s}) must be after its start ({start_s})")
    info = read_clip_info(path)
    first = round(start_s * info.fps)
    if first >= info.frame_count:
        raise ValueError(f"segment start {start_s}s is outside the clip ({info.frame_count} frames)")
    stop = None if end_s is None else round(end_s * info.fps)
    return _frames(Path(path), first, stop)


def _frames(path: Path, first: int, stop: int | None) -> Iterator[np.ndarray]:
    cap = _open(path)
    try:
        if first:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        index = first
        while stop is None or index < stop:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
            index += 1
    finally:
        cap.release()
