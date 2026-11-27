# tests

Tests for the analysis program, the setup tool and the shared code.

No tests yet.

## Sprint 4 test clips

The clips come from [NVIDIA PhysicalAI-SmartSpaces](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces) (MTMC Tracking 2024), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They are rendered retail recordings from two fixed, angled cameras. Each original MP4 is 1920 × 1080 at 30 FPS and lasts 13:19.80.

Download the original MP4s from the linked source pages and keep them outside the repository. The timestamps below refer to the original files.

| People in view | Source clip | Selected intervals (MM:SS) | Selected duration |
| --- | --- | --- | ---: |
| 1–3 | [scene_072 / camera_0651](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/blob/main/MTMC_Tracking_2024/test/scene_072/camera_0651/video.mp4) | 00:00–00:46, 00:51–00:56, 00:59–01:25, 01:34–01:57, 02:07–02:28, 02:40–03:47, 03:58–04:48, 04:59–05:33, 05:50–06:34, 06:48–07:14, 07:51–08:10, 08:20–08:51, 08:58–09:49, 10:54–13:19 | 09:48 |
| 8+ | [scene_080 / camera_0782](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces/blob/main/MTMC_Tracking_2024/test/scene_080/camera_0782/video.mp4) | 00:00–00:35, 01:26–01:51, 02:22–02:52, 02:54–03:45, 04:21–05:28, 05:35–06:35, 07:34–08:08, 08:17–09:05 | 05:50 |

See [ODK-43](https://hasat.atlassian.net/browse/ODK-43?focusedCommentId=10172) for the clip selection and [ODK-45](https://hasat.atlassian.net/browse/ODK-45) for the performance measurement protocol.
