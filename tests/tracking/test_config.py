import pytest

from analysis.tracking.config import TrackerConfig, load_tracker_config


def test_defaults_match_ocsort_defaults():
    c = TrackerConfig()
    assert (c.det_thresh, c.max_age, c.min_hits, c.iou_threshold,
            c.delta_t, c.asso_func, c.inertia, c.use_byte) == (0.6, 30, 3, 0.3, 3, "iou", 0.2, False)


def test_load_reads_tracker_table(tmp_path):
    p = tmp_path / "edge.toml"
    p.write_text("[tracker]\nmax_age = 15\nmin_hits = 2\n", encoding="utf-8")
    c = load_tracker_config(p)
    assert (c.max_age, c.min_hits, c.det_thresh) == (15, 2, 0.6)


def test_load_without_tracker_table_gives_defaults(tmp_path):
    p = tmp_path / "edge.toml"
    p.write_text("[video]\npath='x'\n", encoding="utf-8")
    assert load_tracker_config(p) == TrackerConfig()


@pytest.mark.parametrize("bad", [{"det_thresh": 0}, {"det_thresh": 1.2}, {"max_age": 0},
                                 {"iou_threshold": 1.0}, {"inertia": -0.1}, {"delta_t": 0},
                                 {"asso_func": "reid"}])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        TrackerConfig.from_mapping(bad)


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="with_reid"):
        TrackerConfig.from_mapping({"with_reid": True})
