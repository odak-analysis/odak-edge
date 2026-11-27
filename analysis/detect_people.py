"""Preview RTMO-s detections on consecutive video frames."""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import onnxruntime as ort


def detect(frame, model, input_name, box_threshold, offset=(0, 0), frame_size=(640, 640)):
    height, width = frame.shape[:2]
    max_width, max_height = frame_size
    scale = min(max_width / width, max_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    small = cv2.resize(frame, (resized_width, resized_height))
    padded = cv2.copyMakeBorder(
        small, 0, 640 - resized_height, 0, 640 - resized_width,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    model_input = padded.transpose(2, 0, 1)[None].astype("float32")

    boxes, keypoints = model.run(None, {input_name: model_input})

    scale_x = width / resized_width
    scale_y = height / resized_height
    offset_x, offset_y = offset
    detections = []
    for box, person_points in zip(boxes[0], keypoints[0]):
        x1, y1, x2, y2, box_score = box
        if box_score < box_threshold:
            continue
        bounds = (
            x1 * scale_x + offset_x, y1 * scale_y + offset_y,
            x2 * scale_x + offset_x, y2 * scale_y + offset_y,
        )
        points = [
            (x * scale_x + offset_x, y * scale_y + offset_y, score)
            for x, y, score in person_points
        ]
        detections.append((bounds, box_score, points))
    return detections


def same_region(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
    smaller_area = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    # A crop can cut off part of someone already detected in the full frame.
    return smaller_area > 0 and overlap_width * overlap_height >= 0.6 * smaller_area


def draw_detections(frame, detections, color):
    for bounds, box_score, person_points in detections:
        x1, y1, x2, y2 = (round(value) for value in bounds)
        top_left = (x1, y1)
        cv2.rectangle(frame, top_left, (x2, y2), color, 4)
        label_position = (max(0, top_left[0]), max(20, top_left[1] - 5))
        cv2.putText(
            frame, f"{box_score:.2f}", label_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )

        for x, y, point_score in person_points:
            if point_score >= 0.5:
                center = (round(x), round(y))
                cv2.circle(frame, center, 5, (0, 0, 255), -1)


def main():
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=workspace / "test-data" / "odk-43" / "1-3 vid.mp4",
        help="Path to the video to preview.",
    )
    parser.add_argument(
        "--model", type=Path,
        default=workspace / "models" / "rtmo-s" / "end2end.onnx",
        help="Path to the local RTMO-s ONNX model.",
    )
    parser.add_argument(
        "--start-seconds", type=float, default=0, help="Time to start the preview."
    )
    parser.add_argument(
        "--box-score", type=float, default=0.5, help="Minimum person box confidence."
    )
    parser.add_argument(
        "--frame-size", type=int, nargs=2, default=(640, 640), metavar=("WIDTH", "HEIGHT"),
        help="Maximum image width and height inside the model's fixed 640x640 input.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=60,
        help="Number of frames to preview (default: 60, use 0 for the entire video).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print one JSON record per frame without a window.",
    )
    crop_options = parser.add_mutually_exclusive_group()
    crop_options.add_argument(
        "--crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Preview just this rectangle from the source video.",
    )
    crop_options.add_argument(
        "--refine-crop", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Preview the full frame and look for additional people in this rectangle.",
    )
    crop_options.add_argument(
        "--refine-upper", action="store_true",
        help="Rescan the upper part in three overlapping regions.",
    )
    args = parser.parse_args()
    if args.json and args.crop:
        parser.error("--json uses full-frame coordinates; use --refine-crop for a region scan")
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        parser.error("--start-seconds must be a finite, nonnegative number")
    if not 0 <= args.box_score <= 1:
        parser.error("--box-score must be between 0 and 1")
    if any(not 1 <= side <= 640 for side in args.frame_size):
        parser.error("Both --frame-size dimensions must be between 1 and 640")
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    crop = args.crop or args.refine_crop
    if crop and (min(crop[:2]) < 0 or min(crop[2:]) < 1):
        parser.error("Crop needs nonnegative X and Y and positive dimensions")
    video_path = args.input
    model_path = args.model
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")

    model = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = model.get_inputs()[0].name
    video = cv2.VideoCapture(str(video_path))
    processed_frames = 0
    start_frame = 0
    try:
        if args.start_seconds:
            fps = video.get(cv2.CAP_PROP_FPS)
            start_frame = round(args.start_seconds * fps)
            if fps <= 0 or not video.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                raise SystemExit(f"Could not seek to {args.start_seconds}s in {video_path}")
        while not args.max_frames or processed_frames < args.max_frames:
            ok, frame = video.read()
            if not ok:
                if not processed_frames:
                    raise SystemExit(f"Could not read a frame from {video_path}")
                break
            if crop:
                x, y, width, height = crop
                if x + width > frame.shape[1] or y + height > frame.shape[0]:
                    raise SystemExit("Crop extends beyond the frame")
                region = frame[y:y + height, x:x + width]
            if args.crop:
                frame = region.copy()
            detections = detect(
                frame, model, input_name, args.box_score, frame_size=args.frame_size
            )
            if args.refine_upper:
                height, width = frame.shape[:2]
                upper_height = max(1, round(height * 0.64))
                tile_width = max(1, round(width * 0.40))
                last_left = width - tile_width
                candidates = []
                for left in (0, last_left // 2, last_left):
                    tile = frame[:upper_height, left:left + tile_width]
                    candidates.extend(detect(
                        tile, model, input_name, args.box_score, (left, 0), args.frame_size
                    ))
            elif args.refine_crop:
                candidates = detect(
                    region, model, input_name, args.box_score, (x, y), args.frame_size
                )
            else:
                candidates = []
            extra = []
            for candidate in sorted(candidates, key=lambda person: person[1], reverse=True):
                if not any(same_region(candidate[0], person[0]) for person in detections + extra):
                    extra.append(candidate)
            frame_index = start_frame + processed_frames
            processed_frames += 1
            if args.json:
                people = [
                    {
                        "box": [float(value) for value in box],
                        "score": float(score),
                        "keypoints": [[float(value) for value in point] for point in points],
                    }
                    for box, score, points in detections + extra
                ]
                print(json.dumps({"frame_index": frame_index, "people": people}))
                continue
            draw_detections(frame, extra, (255, 255, 0))
            draw_detections(frame, detections, (0, 255, 0))
            cv2.imshow("Modelin buldugu kutular ve noktalar", frame)
            delay = 0 if args.max_frames == 1 else 1
            if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                break
        print(f"{processed_frames} kare islendi", file=sys.stderr if args.json else sys.stdout)
    finally:
        video.release()
        if not args.json:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
