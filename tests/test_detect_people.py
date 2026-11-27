import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from analysis import detect_people


class DetectionOutputTest(unittest.TestCase):
    def test_json_reports_a_detection_and_an_empty_frame(self):
        video = MagicMock()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        video.read.side_effect = [(True, frame), (True, frame), (False, None)]

        model = MagicMock()
        model.get_inputs.return_value = [MagicMock(name="input")]
        model.run.side_effect = [
            (
                np.array([[[10, 20, 30, 40, 0.75]]], dtype=np.float32),
                np.tile(np.array([15, 25, 0.875], dtype=np.float32), (1, 1, 17, 1)),
            ),
            (np.empty((1, 0, 5)), np.empty((1, 0, 17, 3))),
        ]

        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "test.onnx"
            model_path.touch()
            argv = ["detect_people.py", "--model", str(model_path), "--max-frames", "0", "--json"]
            with (
                patch.object(detect_people.sys, "argv", argv),
                patch.object(detect_people.ort, "InferenceSession", return_value=model),
                patch.object(detect_people.cv2, "VideoCapture", return_value=video),
                patch.object(detect_people.cv2, "imshow") as show,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                detect_people.main()

        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(records, [
            {
                "frame_index": 0,
                "people": [{
                    "box": [30.0, 60.0, 90.0, 120.0],
                    "score": 0.75,
                    "keypoints": [[45.0, 75.0, 0.875]] * 17,
                }],
            },
            {"frame_index": 1, "people": []},
        ])
        self.assertEqual(stderr.getvalue().strip(), "2 kare islendi")
        show.assert_not_called()
        video.release.assert_called_once()
