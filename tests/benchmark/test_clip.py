import pytest

from analysis.benchmark.clip import read_clip_info, segment_frames


def test_clip_info_reads_size_fps_and_frame_count(tiny_clip):
    i = read_clip_info(tiny_clip)
    assert (i.width, i.height, i.fps, i.frame_count) == (64, 48, 10.0, 50)


def test_whole_clip_by_default(tiny_clip):
    assert sum(1 for _ in segment_frames(tiny_clip)) == 50


def test_segment_yields_only_frames_in_range(tiny_clip):
    frames = list(segment_frames(tiny_clip, start_s=2.0, end_s=3.0))
    assert len(frames) == 10
    assert abs(float(frames[0].mean()) - 100) <= 3  # frame 20 -> 20 * 5, MJPG is lossy


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_clip_info(tmp_path / "missing.mp4")


@pytest.mark.parametrize("start, end", [(6.0, None), (3.0, 2.0), (-1.0, 2.0)])
def test_segment_outside_clip_raises(tiny_clip, start, end):
    with pytest.raises(ValueError):
        list(segment_frames(tiny_clip, start, end))
