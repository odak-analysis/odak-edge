import pytest

from analysis.benchmark.stats import WARMUP_FRAMES, median_fps, o5_verdict


def test_warmup_is_30_frames():
    assert WARMUP_FRAMES == 30


def test_first_30_frames_are_discarded():
    assert median_fps([1.0] * 30 + [0.1] * 5) == pytest.approx(10.0)


def test_median_of_per_frame_fps():
    assert median_fps([1.0] * 30 + [0.1, 0.2, 0.25]) == pytest.approx(5.0)  # 10, 5, 4 FPS


def test_even_count_averages_middle_two():
    assert median_fps([1.0] * 30 + [0.1, 0.2]) == pytest.approx(7.5)


def test_needs_frames_after_warmup():
    with pytest.raises(ValueError, match="after the first 30"):
        median_fps([0.1] * 30)


def test_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="positive"):
        median_fps([0.1] * 30 + [0.1, 0.0])


@pytest.mark.parametrize("fps, text", [
    (4.99, "below the O5 minimum of 5 FPS"),
    (5.0, "meets the O5 minimum of 5 FPS but not the target of 10 FPS"),
    (9.99, "meets the O5 minimum of 5 FPS but not the target of 10 FPS"),
    (10.0, "meets the O5 target of 10 FPS"),
])
def test_o5_verdict(fps, text):
    assert o5_verdict(fps) == text
