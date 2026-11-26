import numpy as np
import pytest

from analysis.tracking import Detection, PersonTracker, TrackerConfig


def det(x, y=100, w=40, h=100, s=0.9, kp=None):
    return Detection(box=(x, y, x + w, y + h), score=s, keypoints=kp)


def test_overlapping_detections_keep_same_track_id():
    with PersonTracker() as t:
        ids = [[p.track_id for p in t.update([det(100 + 5 * i)])] for i in range(10)]
    seen = {i for frame in ids for i in frame}
    assert len(seen) == 1


def test_two_people_crossing_keep_their_ids():
    # A moves right, B moves left; they cross with a 30 px vertical offset (partial overlap)
    with PersonTracker() as t:
        first = last = None
        for i in range(30):
            out = t.update([det(10 * i, y=100), det(290 - 10 * i, y=130)])
            by_y = {round(p.box[1]): p.track_id for p in out}
            if len(by_y) == 2:
                first = first or by_y
                last = by_y
    assert first == last


def test_track_survives_short_gap():
    with PersonTracker() as t:
        ids = []
        for i in range(25):
            visible = not (10 <= i < 15)
            ids += [p.track_id for p in t.update([det(100 + 5 * i)] if visible else [])]
    assert len(set(ids)) == 1


def test_new_id_after_gap_longer_than_max_age():
    with PersonTracker(TrackerConfig(max_age=5)) as t:
        before = after = None
        for i in range(25):
            visible = not (5 <= i < 15)
            out = t.update([det(100 + 2 * i)] if visible else [])
            if out and i < 5:
                before = out[0].track_id
            if out and i >= 15:
                after = out[0].track_id
    assert before is not None and after is not None and after != before


def test_low_score_detection_never_tracked():
    with PersonTracker() as t:
        assert all(t.update([det(0, s=0.4)]) == [] for _ in range(5))


def test_empty_frame_returns_empty_list():
    with PersonTracker() as t:
        assert t.update([]) == []


def test_keypoints_follow_their_detection():
    kp_a, kp_b = np.full((17, 3), 1.0), np.full((17, 3), 2.0)
    with PersonTracker() as t:
        for i in range(4):
            out = t.update([det(0 + 5 * i, kp=kp_a), det(300 - 5 * i, kp=kp_b)])
    by_x = sorted(out, key=lambda p: p.box[0])
    assert np.array_equal(by_x[0].keypoints, kp_a) and np.array_equal(by_x[1].keypoints, kp_b)


# --- Review Focus ---

def test_second_open_tracker_in_same_process_raises():
    with PersonTracker():
        with pytest.raises(RuntimeError, match="one PersonTracker per process"):
            PersonTracker()
    PersonTracker().close()  # can be opened again after the first one is closed


def test_integer_boxes_accepted_and_input_not_mutated():
    d = Detection(box=(10, 10, 50, 110), score=0.9)
    with PersonTracker() as t:
        for _ in range(3):
            out = t.update([d])
    assert d.box == (10, 10, 50, 110) and len(out) == 1


def test_duplicate_boxes_get_distinct_detections():
    kp1, kp2 = np.zeros((17, 3)), np.ones((17, 3))
    with PersonTracker() as t:
        for _ in range(4):
            out = t.update([det(100, kp=kp1), det(100, kp=kp2)])
    assert len({id(p.keypoints) for p in out}) == len(out)


@pytest.mark.parametrize("box", [(50, 10, 50, 110), (60, 10, 50, 110),
                                 (float("nan"), 10, 50, 110), (0, 0, float("inf"), 100)])
def test_degenerate_boxes_are_dropped(box):
    with PersonTracker() as t:
        for i in range(4):
            out = t.update([Detection(box=box, score=0.9), det(200 + 5 * i)])
    assert len(out) == 1 and out[0].box[0] >= 200


def test_detection_without_keypoints():
    with PersonTracker() as t:
        for i in range(3):
            out = t.update([det(5 * i)])
    assert out[0].keypoints is None


# --- Review fixes ---

def test_person_near_top_left_corner_stays_in_output():
    # box coordinates plus score are negative: OC-SORT then returns its own estimate, not the box
    with PersonTracker() as t:
        counts = [len(t.update([Detection(box=(-80 + i, -120 + i, -10 + i, 10 + i), score=0.9)]))
                  for i in range(8)]
    assert counts == [1] * 8


def test_close_boxes_keep_their_own_keypoints_and_ids():
    # 0.01 px apart at x = 1900: the default relative tolerance of np.allclose treats them as equal
    kpa, kpb = np.full((17, 3), 1.0), np.full((17, 3), 2.0)
    owner = {}
    with PersonTracker() as t:
        for _ in range(10):
            a = Detection(box=(1900.0, 100.0, 1940.0, 200.0), score=0.9, keypoints=kpa)
            b = Detection(box=(1900.01, 100.0, 1940.01, 200.0), score=0.9, keypoints=kpb)
            for p in t.update([a, b]):
                want = kpa if abs(p.box[0] - 1900.0) < 0.005 else kpb
                assert p.keypoints is want
                assert owner.setdefault(p.track_id, id(p.keypoints)) == id(p.keypoints)
