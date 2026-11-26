import cv2
import numpy as np
import pytest

from analysis.tracking import TrackedPerson
from analysis.tracking.debug_view import draw_tracks


def test_draw_tracks_marks_box_and_keeps_input():
    frame = np.zeros((200, 200, 3), np.uint8)
    out = draw_tracks(frame, [TrackedPerson(7, (50, 60, 100, 160), 0.9, None)])
    assert frame.sum() == 0
    assert out[60, 50:101].any() and out[60:161, 50].any()


def test_draw_tracks_writes_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: pytest.fail("imwrite called"))
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a, **k: pytest.fail("VideoWriter called"))
    draw_tracks(np.zeros((100, 100, 3), np.uint8), [TrackedPerson(1, (10, 10, 40, 90), 0.9, None)])
    assert list(tmp_path.iterdir()) == []
