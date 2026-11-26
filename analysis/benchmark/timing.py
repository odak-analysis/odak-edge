"""Per-frame timing loop. Reading (decoding) a frame counts as part of its time."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Measurement:
    frame_seconds: list[float]
    passes: int
    started_at: datetime
    ended_at: datetime


def measure(
    make_frames: Callable[[], Iterable[T]],
    process: Callable[[T], object],
    min_seconds: float,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], datetime] = datetime.now,
) -> Measurement:
    """Run ``process`` over the frames and time every frame, decoding included.

    The source is read at least once completely. After that it is read again from
    the start until ``min_seconds`` have passed; the last pass may end early.
    The process-return value is dropped: nothing about the frames is kept.
    """
    frame_seconds: list[float] = []
    passes = 0
    start = clock()
    started_at = now()
    while True:
        frames = iter(make_frames())  # reopening the source is not part of any frame
        previous = clock()
        got_frame = False
        for frame in frames:
            got_frame = True
            process(frame)
            end = clock()
            frame_seconds.append(end - previous)
            previous = end
            if passes >= 1 and end - start >= min_seconds:
                break
        if not got_frame:
            raise ValueError("source gave no frames")
        passes += 1
        if clock() - start >= min_seconds:
            break
    return Measurement(frame_seconds, passes, started_at, now())
