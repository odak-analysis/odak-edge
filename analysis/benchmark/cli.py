"""Command line: python -m analysis.benchmark run | idle | power."""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .clip import read_clip_info, segment_frames
from .pipeline import load_pipeline
from .power import read_hwinfo_log
from .results import STAGES_MEASURED, STAGES_NOT_MEASURED, append_record, fill_power
from .stats import WARMUP_FRAMES, median_fps, o5_verdict
from .timing import measure


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _run(args: argparse.Namespace) -> int:
    clip = Path(args.clip)
    info = read_clip_info(clip)
    segment = tuple(args.segment) if args.segment else ()
    segment_frames(clip, *segment)  # fail on a bad segment before the pipeline is loaded

    pipeline = load_pipeline(Path(args.config))
    try:
        m = measure(lambda: segment_frames(clip, *segment), pipeline.process,
                    min_seconds=args.min_minutes * 60)
        fps = median_fps(m.frame_seconds)
        record = {
            "kind": "run", "issue": args.issue, "git_commit": _git_commit(),
            "machine": args.machine, "os": platform.platform(), "power_plan": args.power_plan,
            "clip": clip.name, "input_type": args.input_type, "resolution": [info.width, info.height],
            "clip_fps": info.fps, "segment_s": list(segment) if segment else None,
            "crowd_level": args.crowd,
            "processed_size": list(pipeline.info.processed_size), "models": list(pipeline.info.models),
            "runtime": pipeline.info.runtime, "stages_measured": list(STAGES_MEASURED),
            "stages_not_measured": list(STAGES_NOT_MEASURED),
            "warmup_frames": WARMUP_FRAMES, "frames_measured": len(m.frame_seconds) - WARMUP_FRAMES,
            "measured_seconds": round(sum(m.frame_seconds), 2), "passes": m.passes,
            "median_fps": round(fps, 2), "o5": o5_verdict(fps),
            "started_at": m.started_at.isoformat(), "ended_at": m.ended_at.isoformat(),
            "avg_power_w": None,
        }
    finally:
        pipeline.close()
    append_record(Path(args.out), record)
    print(f"median {record['median_fps']} FPS over {record['frames_measured']} frames "
          f"({record['measured_seconds']} s): {record['o5']}")
    return 0


def _idle(args: argparse.Namespace) -> int:
    started = datetime.now()
    time.sleep(args.minutes * 60)
    append_record(Path(args.out), {
        "kind": "idle", "issue": args.issue, "machine": args.machine, "os": platform.platform(),
        "power_plan": args.power_plan, "started_at": started.isoformat(),
        "ended_at": datetime.now().isoformat(), "avg_power_w": None,
    })
    return 0


def _power(args: argparse.Namespace) -> int:
    filled = fill_power(Path(args.results), read_hwinfo_log(Path(args.hwinfo_log)))
    print(f"power filled for records {filled}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis.benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    def common(s):
        s.add_argument("--machine", required=True)
        s.add_argument("--power-plan", required=True)
        s.add_argument("--issue", default="ODK-45")
        s.add_argument("--out", required=True)

    run = sub.add_parser("run", help="measure median FPS on a clip")
    run.add_argument("--config", required=True)
    run.add_argument("--clip", required=True)
    run.add_argument("--input-type", required=True, choices=["rendered", "real"])
    run.add_argument("--crowd", required=True, choices=["1-3", "8+"])
    run.add_argument("--segment", nargs=2, type=float, metavar=("START", "END"))
    run.add_argument("--min-minutes", type=float, default=10.0)
    common(run)
    run.set_defaults(func=_run)

    idle = sub.add_parser("idle", help="record an idle window for the power baseline")
    idle.add_argument("--minutes", type=float, default=5.0)
    common(idle)
    idle.set_defaults(func=_idle)

    power = sub.add_parser("power", help="fill average power from a HWiNFO log")
    power.add_argument("--hwinfo-log", required=True)
    power.add_argument("--results", required=True)
    power.set_defaults(func=_power)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
