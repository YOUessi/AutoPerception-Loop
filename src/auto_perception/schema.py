"""Minimal, explicit offline schema shared by tracking, selection, and evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        )

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        b = self.clip(width, height)
        if not b.valid:
            raise ValueError("Clipped box has no positive area")
        return (
            (b.x1 + b.x2) / (2 * width),
            (b.y1 + b.y2) / (2 * height),
            (b.x2 - b.x1) / width,
            (b.y2 - b.y1) / height,
        )

    def iou(self, other: "Box") -> float:
        intersect = Box(max(self.x1, other.x1), max(self.y1, other.y1),
                        min(self.x2, other.x2), min(self.y2, other.y2)).area
        union = self.area + other.area - intersect
        return intersect / union if union > 0 else 0.0


@dataclass(frozen=True)
class Detection:
    bbox: Box
    class_id: int
    confidence: float = 1.0
    track_id: int | None = None

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "Detection":
        box = obj["bbox"]
        if isinstance(box, dict):
            b = Box(float(box["x1"]), float(box["y1"]),
                    float(box["x2"]), float(box["y2"]))
        else:
            b = Box(*map(float, box))
        return cls(b, int(obj["class_id"]), float(obj.get("confidence", 1.0)),
                   int(obj["track_id"]) if obj.get("track_id") is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    frame_index: int
    image_path: str
    width: int
    height: int
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    source_split: str = "train"

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "FrameRecord":
        return cls(
            video_id=str(obj["video_id"]),
            frame_index=int(obj["frame_index"]),
            image_path=str(obj["image_path"]),
            width=int(obj["width"]), height=int(obj["height"]),
            detections=tuple(Detection.from_dict(d) for d in obj.get("detections", [])),
            source_split=str(obj.get("source_split", "train")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "frame_index": self.frame_index,
                "image_path": self.image_path, "width": self.width,
                "height": self.height,
                "source_split": self.source_split,
                "detections": [d.to_dict() for d in self.detections]}

    @property
    def key(self) -> tuple[str, int]:
        return self.video_id, self.frame_index

    def verify(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.frame_index < 0:
            raise ValueError(f"Invalid frame dimensions/index in {self.key}")
        if not self.video_id or self.video_id in (".", ".."):
            raise ValueError("Invalid video ID")
        for d in self.detections:
            if not d.bbox.valid or not (0.0 <= d.confidence <= 1.0):
                raise ValueError(f"Invalid detection in {self.key}")


def safe_id(value: str) -> str:
    """Safe component for paths produced by clips; reject rather than silently rewrite."""
    if not value or value in (".", "..") or not all(c.isalnum() or c in "_-" for c in value):
        raise ValueError(f"Unsafe video identifier: {value!r}")
    return value


def safe_name(value: str) -> str:
    if Path(value).name != value or value in ("", ".", "..") or "\\" in value:
        raise ValueError(f"Unsafe file name: {value!r}")
    return value
