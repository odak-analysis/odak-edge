from datetime import datetime
from pathlib import Path

import pytest

from analysis.benchmark.power import (CPU_COLUMN, GPU_COLUMN, PowerSample,
                                      mean_power, read_hwinfo_log)

HEADER = f'"Date","Time","Core Temp [°C]","{CPU_COLUMN}","{GPU_COLUMN}",'


def write_log(tmp_path, rows):
    p = tmp_path / "log.csv"
    p.write_text("\n".join([HEADER, *rows, HEADER, '"","","CPU","CPU","GPU",']) + "\n",
                 encoding="latin-1")
    return p


def test_reads_requested_columns(tmp_path):
    p = write_log(tmp_path, ['"28.11.2026","10:00:00.000","55","20.5","30.0",'])
    [s] = read_hwinfo_log(p)
    assert s.at == datetime(2026, 11, 28, 10, 0, 0)
    assert s.watts == {CPU_COLUMN: 20.5, GPU_COLUMN: 30.0}


def test_skips_trailing_header_rows_and_non_numeric(tmp_path):
    p = write_log(tmp_path, ['"28.11.2026","10:00:00.000","55","20.5","30.0",',
                             '"28.11.2026","10:00:01.000","55","N/A","30.0",'])
    assert len(read_hwinfo_log(p)) == 1


def test_decimal_comma_is_read(tmp_path):
    p = write_log(tmp_path, ['"28.11.2026","10:00:00.000","55","20,5","30,0",'])
    assert read_hwinfo_log(p)[0].watts == {CPU_COLUMN: 20.5, GPU_COLUMN: 30.0}


def test_missing_column_raises(tmp_path):
    p = write_log(tmp_path, ['"28.11.2026","10:00:00.000","55","20.5","30.0",'])
    with pytest.raises(ValueError, match="Total System Power"):
        read_hwinfo_log(p, columns=("Total System Power [W]",))


def test_log_without_readable_rows_raises(tmp_path):
    p = write_log(tmp_path, ['"11/28/2026","10:00:00.000","55","20.5","30.0",'])
    with pytest.raises(ValueError, match="no readable samples"):
        read_hwinfo_log(p)


def samples(*secs_and_watts):
    return [PowerSample(datetime(2026, 11, 28, 10, 0, s), {CPU_COLUMN: w}) for s, w in secs_and_watts]


def test_mean_power_uses_only_samples_inside_window():
    s = samples((0, 100.0), (10, 20.0), (20, 40.0), (30, 100.0))
    assert mean_power(s, datetime(2026, 11, 28, 10, 0, 10), datetime(2026, 11, 28, 10, 0, 20),
                      CPU_COLUMN) == pytest.approx(30.0)


def test_mean_power_rejects_window_not_covered_by_log():
    s = samples((30, 20.0), (40, 20.0))
    with pytest.raises(ValueError, match="does not cover"):
        mean_power(s, datetime(2026, 11, 28, 10, 0, 0), datetime(2026, 11, 28, 10, 0, 40),
                   CPU_COLUMN)


def test_sample_log_file_parses():
    got = read_hwinfo_log(Path(__file__).parent / "data" / "hwinfo_sample.csv")
    assert len(got) == 3 and all(v > 0 for s in got for v in s.watts.values())
