import numpy as np
from analysis.tracking.ocsort import OCSort


def test_vendored_ocsort_returns_track_rows():
    t = OCSort(det_thresh=0.6)
    det = np.array([[10.0, 10.0, 50.0, 110.0, 0.9]])
    out = None
    for _ in range(3):
        out = t.update(det.copy(), (1, 1), (1, 1))
    assert out.shape == (1, 5)
    assert out[0, 4] >= 1


def test_vendored_ocsort_accepts_empty_frame():
    t = OCSort(det_thresh=0.6)
    assert t.update(np.empty((0, 5)), (1, 1), (1, 1)).shape == (0, 5)
