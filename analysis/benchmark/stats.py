"""Speed statistics: median frames per second and the comparison with objective O5."""
from __future__ import annotations

import statistics
from typing import Sequence

WARMUP_FRAMES = 30  # NFR-01: the first 30 frames are discarded
O5_MINIMUM_FPS = 5.0
O5_TARGET_FPS = 10.0


def median_fps(frame_seconds: Sequence[float], warmup: int = WARMUP_FRAMES) -> float:
    """Median of the per-frame FPS (1 / seconds) after the first ``warmup`` frames."""
    if any(not s > 0 for s in frame_seconds):
        raise ValueError("frame durations must be positive")
    kept = frame_seconds[warmup:]
    if not kept:
        raise ValueError(f"need at least one frame after the first {warmup}")
    return statistics.median(1.0 / s for s in kept)


def o5_verdict(fps: float) -> str:
    """One phrase that compares a median FPS with the O5 minimum and target."""
    if fps < O5_MINIMUM_FPS:
        return "below the O5 minimum of 5 FPS"
    if fps < O5_TARGET_FPS:
        return "meets the O5 minimum of 5 FPS but not the target of 10 FPS"
    return "meets the O5 target of 10 FPS"
