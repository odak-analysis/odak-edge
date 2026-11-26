"""NFR-17 / PC-1: tracking uses position and motion only; track numbers are temporary."""
import dataclasses
import subprocess
import sys
from pathlib import Path

from analysis.tracking import Detection, PersonTracker, TrackerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN = {"torch", "torchvision", "torchreid", "fastreid", "boxmot",
             "deep_sort_realtime", "ultralytics", "onnxruntime", "tensorflow", "cv2"}


def det(x, y=100, w=40, h=100, s=0.9):
    return Detection(box=(x, y, x + w, y + h), score=s)


def test_tracker_run_loads_no_appearance_or_model_library():
    code = ("import sys\n"
            "from analysis.tracking import PersonTracker, Detection\n"
            "with PersonTracker() as t:\n"
            "    for i in range(20): t.update([Detection((5*i,0,40+5*i,100),0.9)])\n"
            "print(','.join(sorted({m.split('.')[0] for m in sys.modules})))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=REPO_ROOT).stdout.strip().split(",")
    assert FORBIDDEN.isdisjoint(out)


def test_config_has_no_appearance_or_reid_field():
    names = {f.name for f in dataclasses.fields(TrackerConfig)}
    assert names == {"det_thresh", "max_age", "min_hits", "iou_threshold",
                     "delta_t", "asso_func", "inertia", "use_byte"}


def test_track_ids_are_per_run():
    runs = []
    for _ in range(2):
        with PersonTracker() as t:
            for i in range(4):
                out = t.update([det(5 * i)])
        runs.append(out[0].track_id)
    assert runs[0] == runs[1] == 1
