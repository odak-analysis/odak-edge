"""The results file: one JSON object per line. Records are only ever appended; the
only later change is filling in ``avg_power_w`` once the HWiNFO log is available."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .power import CPU_COLUMN, GPU_COLUMN, PowerSample, mean_power

STAGES_MEASURED = ("read", "pose", "tracking")
STAGES_NOT_MEASURED = ("floor mapping", "role separation", "aggregation")


def append_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fill_power(path: Path, samples: Sequence[PowerSample]) -> list[int]:
    """Write average power into records that have none and whose window the log covers.

    Returns the indexes of the records that were filled. All other records stay as they are.
    """
    records = read_records(path)
    filled: list[int] = []
    for i, r in enumerate(records):
        if r.get("avg_power_w") is not None:
            continue
        start, end = datetime.fromisoformat(r["started_at"]), datetime.fromisoformat(r["ended_at"])
        try:
            cpu = mean_power(samples, start, end, CPU_COLUMN)
            gpu = mean_power(samples, start, end, GPU_COLUMN)
        except ValueError:
            continue
        r["avg_power_w"] = {"cpu_package_w": round(cpu, 2), "gpu_w": round(gpu, 2),
                            "total_w": round(cpu + gpu, 2)}
        filled.append(i)
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)
    return filled
