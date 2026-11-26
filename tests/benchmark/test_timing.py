from datetime import datetime

import pytest

from analysis.benchmark.timing import measure


def frames_costing(clk, n, decode_s):
    def make():
        for i in range(n):
            clk.advance(decode_s)
            yield i
    return make


def test_frame_time_includes_reading_and_processing(fake_clock):
    m = measure(frames_costing(fake_clock, 3, 0.01), lambda f: fake_clock.advance(0.04),
                min_seconds=0, clock=fake_clock)
    assert m.frame_seconds == pytest.approx([0.05, 0.05, 0.05])


def test_one_full_pass_even_when_min_seconds_is_zero(fake_clock):
    m = measure(frames_costing(fake_clock, 3, 0.0), lambda f: fake_clock.advance(1.0),
                min_seconds=0, clock=fake_clock)
    assert (len(m.frame_seconds), m.passes) == (3, 1)


def test_loops_source_until_min_seconds(fake_clock):
    m = measure(frames_costing(fake_clock, 3, 0.0), lambda f: fake_clock.advance(1.0),
                min_seconds=5, clock=fake_clock)
    assert (len(m.frame_seconds), m.passes) == (5, 2)


def test_empty_source_raises(fake_clock):
    with pytest.raises(ValueError, match="no frames"):
        measure(lambda: iter([]), lambda f: None, min_seconds=0, clock=fake_clock)


def test_empty_second_pass_raises(fake_clock):
    sources = iter([[1, 2], []])
    with pytest.raises(ValueError, match="no frames"):
        measure(lambda: next(sources), lambda f: fake_clock.advance(1.0),
                min_seconds=10, clock=fake_clock)


def test_records_wall_clock_window(fake_clock):
    stamps = iter([datetime(2026, 11, 28, 10, 0, 0), datetime(2026, 11, 28, 10, 10, 0)])
    m = measure(lambda: iter([1]), lambda f: None, min_seconds=0,
                clock=fake_clock, now=lambda: next(stamps))
    assert (m.started_at, m.ended_at) == (datetime(2026, 11, 28, 10, 0, 0),
                                          datetime(2026, 11, 28, 10, 10, 0))
