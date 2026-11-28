import pytest

from analysis.benchmark import cli
from analysis.benchmark.cli import main
from analysis.benchmark.pipeline import PipelineInfo
from analysis.benchmark.results import read_records


class FakePipeline:
    info = PipelineInfo(("fake-pose.onnx", "OC-SORT"), "FakeRuntime", (640, 384))

    def __init__(self):
        self.closed = False
        self.frames = 0

    def process(self, frame):
        self.frames += 1

    def close(self):
        self.closed = True


@pytest.fixture
def fake_pipeline(monkeypatch):
    p = FakePipeline()
    monkeypatch.setattr(cli, "load_pipeline", lambda config: p)
    return p


def run_args(clip, out, *extra):
    return ["run", "--config", "edge.toml", "--clip", str(clip), "--input-type", "rendered",
            "--crowd", "1-3", "--min-minutes", "0", "--machine", "ref laptop",
            "--power-plan", "Best performance", "--out", str(out), *extra]


def test_run_appends_one_record_with_conditions(tiny_clip, tmp_path, fake_pipeline):
    out = tmp_path / "s.jsonl"
    assert main(run_args(tiny_clip, out)) == 0
    [r] = read_records(out)
    assert (r["kind"], r["crowd_level"], r["resolution"], r["clip_fps"]) == ("run", "1-3", [64, 48], 10.0)
    assert (r["warmup_frames"], r["frames_measured"], r["passes"]) == (30, 20, 1)
    assert r["models"] == ["fake-pose.onnx", "OC-SORT"] and r["runtime"] == "FakeRuntime"
    assert r["processed_size"] == [640, 384] and r["input_type"] == "rendered"
    assert r["stages_measured"] == ["read", "pose", "tracking"]
    assert r["stages_not_measured"] == ["floor mapping", "role separation", "aggregation"]
    assert r["clip"] == "tiny.avi" and r["issue"] == "ODK-45"
    assert r["o5"] and r["median_fps"] > 0 and r["avg_power_w"] is None
    assert fake_pipeline.closed and fake_pipeline.frames == 50


def test_run_with_segment_records_it(tiny_clip, tmp_path, fake_pipeline):
    out = tmp_path / "s.jsonl"
    main(run_args(tiny_clip, out, "--segment", "0", "4"))
    assert read_records(out)[0]["segment_s"] == [0.0, 4.0]
    assert fake_pipeline.frames == 40


def test_run_writes_nothing_but_the_results_file(tiny_clip, tmp_path, fake_pipeline, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    out = tmp_path / "s.jsonl"
    main(run_args(tiny_clip, out))
    assert list(work.iterdir()) == []
    assert not any("track" in k for k in read_records(out)[0])


def test_run_too_short_for_warmup_fails_without_a_record(tiny_clip, tmp_path, fake_pipeline, capsys):
    out = tmp_path / "s.jsonl"
    assert main(run_args(tiny_clip, out, "--segment", "0", "2")) == 2
    assert not out.exists() and "after the first 30" in capsys.readouterr().err
    assert fake_pipeline.closed


def test_run_missing_clip_fails_before_loading_the_pipeline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_pipeline", lambda config: pytest.fail("pipeline loaded"))
    assert main(run_args(tmp_path / "nope.mp4", tmp_path / "s.jsonl")) == 2
    assert "nope.mp4" in capsys.readouterr().err


def test_idle_records_window(tmp_path):
    out = tmp_path / "s.jsonl"
    assert main(["idle", "--minutes", "0", "--machine", "m", "--power-plan", "p", "--out", str(out)]) == 0
    [r] = read_records(out)
    assert r["kind"] == "idle" and r["started_at"] <= r["ended_at"] and r["avg_power_w"] is None


def test_power_command_fills_records(tmp_path, capsys):
    from datetime import datetime
    from analysis.benchmark.results import append_record
    out = tmp_path / "s.jsonl"
    append_record(out, {"kind": "idle", "started_at": datetime(2026, 11, 28, 10, 0, 0).isoformat(),
                        "ended_at": datetime(2026, 11, 28, 10, 0, 10).isoformat(), "avg_power_w": None})
    rows = ['"Date","Time","CPU Package Power [W]","GPU Power [W]",']
    rows += [f'"28.11.2026","10:00:{s:02d}.000","10.0","5.0",' for s in range(0, 12)]
    log = tmp_path / "log.csv"
    log.write_text("\n".join(rows) + "\n", encoding="latin-1")
    assert main(["power", "--hwinfo-log", str(log), "--results", str(out)]) == 0
    assert read_records(out)[0]["avg_power_w"]["total_w"] == 15.0
