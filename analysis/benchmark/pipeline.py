"""What the benchmark needs from a pipeline: a name for the conditions and a per-frame call."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PipelineInfo:
    models: tuple[str, ...]
    runtime: str
    processed_size: tuple[int, int]


class FramePipeline(Protocol):
    info: PipelineInfo

    def process(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


def load_pipeline(config_path: Path) -> FramePipeline:
    raise NotImplementedError("needs ODK-42 detection stage")
