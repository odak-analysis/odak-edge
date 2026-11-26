"""Tracker parameters: defaults are OC-SORT's own, validated, loadable from TOML."""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ASSO_FUNCS = frozenset({"iou", "giou", "diou", "ciou", "ct_dist"})


@dataclass(frozen=True)
class TrackerConfig:
    det_thresh: float = 0.6
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    delta_t: int = 3
    asso_func: str = "iou"
    inertia: float = 0.2
    use_byte: bool = False

    def __post_init__(self) -> None:
        for name in ("det_thresh", "iou_threshold", "inertia"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{name} must be a number, got {v!r}")
        for name in ("max_age", "min_hits", "delta_t"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{name} must be a whole number, got {v!r}")
        if not isinstance(self.use_byte, bool):
            raise ValueError(f"use_byte must be true or false, got {self.use_byte!r}")
        if not isinstance(self.asso_func, str):
            raise ValueError(f"asso_func must be a string, got {self.asso_func!r}")
        if not 0 < self.det_thresh < 1:
            raise ValueError(f"det_thresh must be in (0, 1), got {self.det_thresh}")
        if not 0 < self.iou_threshold < 1:
            raise ValueError(f"iou_threshold must be in (0, 1), got {self.iou_threshold}")
        if not 0 <= self.inertia <= 1:
            raise ValueError(f"inertia must be in [0, 1], got {self.inertia}")
        if self.max_age < 1:
            raise ValueError(f"max_age must be >= 1, got {self.max_age}")
        if self.min_hits < 0:
            raise ValueError(f"min_hits must be >= 0, got {self.min_hits}")
        if self.delta_t < 1:
            raise ValueError(f"delta_t must be >= 1, got {self.delta_t}")
        if self.asso_func not in ASSO_FUNCS:
            raise ValueError(
                f"asso_func must be one of {sorted(ASSO_FUNCS)}, got {self.asso_func!r}"
            )

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "TrackerConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(m) - known)
        if unknown:
            raise ValueError(f"unknown tracker setting(s): {', '.join(unknown)}")
        return cls(**m)


def load_tracker_config(path: Path) -> TrackerConfig:
    """Read the [tracker] table of a TOML file; defaults if the table is absent."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return TrackerConfig.from_mapping(data.get("tracker", {}))
