import pytest


class FakeClock:
    """A clock that only moves when a test advances it."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def tiny_clip(tmp_path):
    """A synthetic 64x48, 10 FPS, 50-frame clip; every pixel of frame i has value i * 5."""
    import cv2
    import numpy as np

    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(50):
        writer.write(np.full((48, 64, 3), i * 5, np.uint8))
    writer.release()
    return path
