# analysis

The analysis program. It reads the camera video inside the store and produces anonymous, aggregated numbers.

## Run person detection

Run these commands in PowerShell from the repository root. If you already have a virtual environment with OpenCV and ONNX Runtime, skip setup.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install opencv-python onnxruntime
```

The detection model is **RTMO-s Body7 (640 × 640)**. Download the [OpenMMLab ONNX archive](https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip) and extract its `end2end.onnx` file. Keep the model and the [ODK-43 test clips](https://hasat.atlassian.net/browse/ODK-43?focusedCommentId=10172) outside the repository.

Preview one frame of the crowded clip at 15 seconds. Replace the paths if you saved the files elsewhere.

```powershell
.\.venv\Scripts\python.exe .\analysis\detect_people.py --input "..\test-data\odk-43\8-kisi-test.mp4" --model "..\models\rtmo-s\end2end.onnx" --start-seconds 15 --max-frames 1
```

Green boxes surround detected people. The label is the box confidence. Red dots show keypoints with confidence at least 0.5. Press any key to close a single-frame preview, or **Q** or **Esc** to stop a longer preview.

- To run the whole clip, omit `--start-seconds` and set `--max-frames 0`.
- To print one JSON record per frame without opening a window, add `--json`. Each record has a zero-based `frame_index` and a `people` list. Each person has a `box` in `[x1, y1, x2, y2]` order, a box `score`, and 17 `keypoints` in `[x, y, confidence]` order. Coordinates refer to the source frame and can extend past its edges.
- To adjust how much room the image can use within the model input, set `--frame-size WIDTH HEIGHT`. The image keeps its aspect ratio. The model still receives a 640 × 640 input with gray padding.
- `--refine-upper` is an optional experiment. It runs three extra scans across the upper part of the frame and draws additional detections in cyan. The regular preview uses one full-frame scan.

The script processes video frames in memory.

## Run the test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test checks the JSON output for a detected person and an empty frame. It uses simulated model output so it can run without video files or model weights.
