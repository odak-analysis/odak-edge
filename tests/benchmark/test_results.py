from datetime import datetime

from analysis.benchmark.power import CPU_COLUMN, GPU_COLUMN, PowerSample
from analysis.benchmark.results import (STAGES_MEASURED, STAGES_NOT_MEASURED,
                                        append_record, fill_power, read_records)


def test_existing_records_are_kept(tmp_path):
    p = tmp_path / "r" / "s.jsonl"
    append_record(p, {"kind": "run", "n": 1})
    append_record(p, {"kind": "run", "n": 2})
    assert [r["n"] for r in read_records(p)] == [1, 2]


def test_power_fills_only_covered_records_without_power(tmp_path):
    p = tmp_path / "s.jsonl"

    def w(m):
        return datetime(2026, 11, 28, 10, m).isoformat()

    append_record(p, {"kind": "run", "started_at": w(0), "ended_at": w(1), "avg_power_w": None})
    append_record(p, {"kind": "run", "started_at": w(0), "ended_at": w(1), "avg_power_w": {"total_w": 1}})
    append_record(p, {"kind": "run", "started_at": w(30), "ended_at": w(31), "avg_power_w": None})
    log = [PowerSample(datetime(2026, 11, 28, 10, 0, s), {CPU_COLUMN: 20.0, GPU_COLUMN: 30.0})
           for s in range(60)]
    log.append(PowerSample(datetime(2026, 11, 28, 10, 1, 0), {CPU_COLUMN: 20.0, GPU_COLUMN: 30.0}))
    assert fill_power(p, log) == [0]
    r = read_records(p)
    assert r[0]["avg_power_w"] == {"cpu_package_w": 20.0, "gpu_w": 30.0, "total_w": 50.0}
    assert r[1]["avg_power_w"] == {"total_w": 1} and r[2]["avg_power_w"] is None


def test_stage_lists_match_the_plan():
    assert STAGES_MEASURED == ("read", "pose", "tracking")
    assert STAGES_NOT_MEASURED == ("floor mapping", "role separation", "aggregation")
