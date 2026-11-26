# analysis

The analysis program. It reads the camera video inside the store and produces anonymous, aggregated numbers.

## Tracking

`analysis/tracking/` gives each detected person a temporary track number within one camera, so that the same person keeps the same number from frame to frame.

**Privacy.** No appearance or re-identification model is loaded. The tracker uses position and motion only. Track numbers are temporary: they are valid only within one camera and one run, start again from 1 in every run, and are not written to any output except the debug view.

**Source.** The tracking algorithm is OC-SORT (MIT license), copied unchanged from https://github.com/noahcao/OC_SORT at commit `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`. Origin, reason for use and license are recorded in [`tracking/ocsort/NOTICE.md`](tracking/ocsort/NOTICE.md). `PersonTracker` in `tracking/tracker.py` is a thin adapter on top of it.

**One tracker per process.** OC-SORT keeps its track counter on a class, so a second `PersonTracker` in the same process would reset the numbers of the first. Opening a second one while the first is open raises `RuntimeError`. Run each camera in its own process.

**Call it for every frame.** `PersonTracker.update(detections)` must be called for every frame, also when nothing was detected (pass an empty list). Otherwise missing frames are not counted and tracks live too long.

**Settings.** The `[tracker]` table in `config/edge.example.toml` sets these parameters. The defaults are OC-SORT's own.

| Setting | Meaning |
|---|---|
| `det_thresh` | Detections scoring at or below this are left out of the main matching. With `use_byte`, those scoring above 0.1 can still extend an existing track, so they can appear in the output. |
| `max_age` | Frames a person may go unseen before the track is dropped. |
| `min_hits` | Matched frames in a row before a track number is shown. In the first `min_hits` frames of a run every new track is shown at once. |
| `iou_threshold` | Minimum box overlap to match a detection to a track. |
| `delta_t` | Frames back used to estimate a person's direction of motion. |
| `asso_func` | Overlap measure for the later matching rounds (BYTE and re-matching of unmatched tracks): `iou`, `giou`, `diou`, `ciou` or `ct_dist`. The first round always uses IoU. |
| `inertia` | Weight of the motion direction when matching. |
| `use_byte` | Also match low-score detections (BYTE association). |

**Debug view.** `tracking/debug_view.py` draws `#<track_id>` and a box on a copy of a frame. It never writes anything to disk.
