"""Power draw from a HWiNFO CSV log (free tool): per-sample readings and window averages.

HWiNFO reports the CPU package and the GPU, not the power taken from the wall, so the
numbers are a lower bound of what the laptop draws.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

CPU_COLUMN = "CPU Package Power [W]"
GPU_COLUMN = "GPU Power [W]"

_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d")
_TIME_FORMATS = ("%H:%M:%S.%f", "%H:%M:%S")


@dataclass(frozen=True)
class PowerSample:
    at: datetime
    watts: dict[str, float]


def _parse(text: str, formats: Sequence[str]) -> datetime:
    for fmt in formats:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(text)


def read_hwinfo_log(path: Path, columns: Sequence[str] = (CPU_COLUMN, GPU_COLUMN)) -> list[PowerSample]:
    """Read the requested columns. Rows that do not parse (HWiNFO repeats the header
    at the end of the file) and non-numeric cells are skipped."""
    with open(path, newline="", encoding="latin-1") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    wanted = {}
    for name in ("Date", "Time", *columns):
        if name not in header:
            raise ValueError(f"column not found in HWiNFO log: {name}")
        wanted[name] = header.index(name)

    samples: list[PowerSample] = []
    for row in rows[1:]:
        try:
            day = _parse(row[wanted["Date"]], _DATE_FORMATS)
            clock = _parse(row[wanted["Time"]], _TIME_FORMATS)
            watts = {c: float(row[wanted[c]].replace(",", ".")) for c in columns}
        except (ValueError, IndexError):
            continue
        samples.append(PowerSample(datetime.combine(day.date(), clock.time()), watts))
    if not samples:
        raise ValueError(f"no readable samples in {path}")
    return samples


def mean_power(samples: Sequence[PowerSample], start: datetime, end: datetime,
               column: str, tolerance_s: float = 5.0) -> float:
    """Mean of ``column`` over the samples inside [start, end].

    Raises if the log does not reach both ends of the window (within ``tolerance_s``),
    so a log that was started late or stopped early never gives a partial average.
    """
    inside = [s for s in samples if start <= s.at <= end]
    tol = timedelta(seconds=tolerance_s)
    if not inside or inside[0].at > start + tol or inside[-1].at < end - tol:
        raise ValueError(f"log does not cover the window {start.isoformat()} to {end.isoformat()}")
    return sum(s.watts[column] for s in inside) / len(inside)
