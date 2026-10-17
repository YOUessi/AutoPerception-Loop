"""Inference-time YOLO + Ultralytics ByteTrack adapter.

Use a BDD-trained model; raw COCO class IDs are NOT aligned to our BDD 8-class labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_perception.io import write_jsonl
from auto_perception.schema import Box, Detection, FrameRecord, safe_id


def track_video(
    video: str | Path, weights: str | Path, frames_dir: str | Path,
    output_jsonl: str | Path, conf: float = 0.08, imgsz: int = 640,
    device: str = "0", source_split: str = "train", max_frames: int | None = None,
) -> dict[str, Any]:
    if source_split not in {"train", "val", "test"}:
        raise ValueError("source_split must be train, val, or test")
    if not 0.0 < conf < 1.0:
        raise ValueError("conf must be between zero and one")
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install optional vision deps: pip install -e ".[vision]"') from exc
    video = Path(video).resolve(strict=True)
    video_id = safe_id(video.stem)
    image_dir = Path(frames_dir) / video_id
    image_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {video}")
    model = YOLO(str(weights))
    # Tracker state intentionally lives for one video only. New call -> fresh model.
    records = []
    index = 0
    try:
        while True:
            success, frame = cap.read()
            if not success or (max_frames is not None and index >= max_frames):
                break
            height, width = frame.shape[:2]
            image_path = image_dir / f"{index:06d}.jpg"
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Cannot save frame: {image_path}")
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            boxes = result.boxes
            detections = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
                for coords, score, cls_id, track_id in zip(xyxy, confs, classes, ids):
                    detections.append(Detection(Box(*coords), int(cls_id), float(score),
                                                int(track_id) if track_id is not None else None))
            records.append(FrameRecord(video_id, index, str(image_path.resolve()),
                                       width, height, tuple(detections), source_split))
            index += 1
    finally:
        cap.release()
    write_jsonl(output_jsonl, (r.to_dict() for r in records))
    return {"video_id": video_id, "frames": len(records),
            "detections": sum(len(x.detections) for x in records),
            "output": str(output_jsonl)}
