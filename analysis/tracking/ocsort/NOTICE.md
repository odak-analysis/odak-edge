# Third-party notice: OC-SORT

- **Source:** https://github.com/noahcao/OC_SORT
- **Commit:** `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- **Copied from:** `trackers/ocsort_tracker/`: `ocsort.py`, `association.py`, `kalmanfilter.py`
- **Copied on:** 2026-11-24
- **Modifications:** none. The three files are unchanged (byte-for-byte).
- **License:** MIT, see `LICENSE` in this folder.

## Reason for use

Technology Stack, Table 5 and Table 20: OC-SORT is the chosen tracker algorithm and is reused without change. It uses position and motion only (no appearance model), which satisfies FR-03 and PC-1.

## Notes

- OC-SORT's Kalman filter needs the `filterpy` package, which is listed in `pyproject.toml`.
- `__init__.py` and this file are written by the ODAK team. All other files here are the original source.
