"""Reproducible 3-way pseudo-label gate: accept / review / reject.

Research hypothesis: track consistency can help label selection; it is NOT
assumed to improve accuracy until held-out human ground truth is evaluated.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log, sqrt
from statistics import mean, pstdev
from typing import Any

from auto_perception.schema import Detection, FrameRecord


@dataclass(frozen=True)
class SelectionConfig:
    method: str = "temporal_uncertainty"
    accept_threshold: float = 0.78
    review_threshold: float = 0.48
    min_track_frames: int = 3
    high_conf_shortcut: float = 0.97
    critical_classes: tuple[int, ...] = (0, 1, 6, 7)
    weight_confidence: float = 0.45
    weight_support: float = 0.25
    weight_class_agreement: float = 0.15
    weight_area_stability: float = 0.15
    penalty_uncertainty: float = 0.10

    @classmethod
    def from_mapping(cls, params: dict[str, Any]) -> "SelectionConfig":
        params = dict(params)
        if "critical_classes" in params:
            params["critical_classes"] = tuple(map(int, params["critical_classes"]))
        config = cls(**params)
        config.verify()
        return config

    def verify(self) -> None:
        if self.method not in {"confidence", "temporal", "temporal_uncertainty"}:
            raise ValueError(f"Unknown scoring method: {self.method}")
        if not 0 <= self.review_threshold < self.accept_threshold <= 1:
            raise ValueError("0 <= review_threshold < accept_threshold <= 1 required")
        if not 0 < self.high_conf_shortcut <= 1 or self.min_track_frames < 1:
            raise ValueError("Invalid track length or shortcut threshold")
        if any(x < 0 for x in (self.weight_confidence, self.weight_support,
                                self.weight_class_agreement, self.weight_area_stability,
                                self.penalty_uncertainty)):
            raise ValueError("Weights and penalty must be non-negative")
        if self.method != "confidence" and abs(sum((self.weight_confidence,
            self.weight_support, self.weight_class_agreement,
            self.weight_area_stability)) - 1.0) > 1e-6:
            raise ValueError("Non-penalty weights must sum to one")


def _track_features(frames: list[FrameRecord], config: SelectionConfig) -> dict[tuple[str, int], dict[str, float]]:
    tracks: dict[tuple[str, int], list[tuple[int, Detection]]] = defaultdict(list)
    for frame in frames:
        for detection in frame.detections:
            if detection.track_id is not None:
                tracks[(frame.video_id, detection.track_id)].append((frame.frame_index, detection))
    features = {}
    for key, samples in tracks.items():
        samples.sort(key=lambda x: x[0])
        classes = Counter(item.class_id for _, item in samples)
        majority = classes.most_common(1)[0][1] / len(samples)
        start, end = samples[0][0], samples[-1][0]
        coverage = len({index for index, _ in samples}) / (end - start + 1)
        support = min(1.0, len(samples) / config.min_track_frames) * coverage
        areas = [max(1.0, d.bbox.area) for _, d in samples]
        changes = [abs(log(a / b)) for a, b in zip(areas[1:], areas[:-1])]
        stability = 1.0 / (1.0 + (mean(changes) if changes else 0.0))
        uncertainty = min(1.0, pstdev(d.confidence for _, d in samples) * 2)
        features[key] = {
            "support": support,
            "class_agreement": majority,
            "area_stability": stability,
            "uncertainty": uncertainty,
            "track_length": float(len(samples)),
        }
    return features


def select_frames(frames: list[FrameRecord], cfg: SelectionConfig) -> list[dict[str, Any]]:
    """Retain all frames; acceptance rate must be measured against ALL observations."""
    cfg.verify()
    features = _track_features(frames, cfg)
    results = []
    for frame in frames:
        row = {"video_id": frame.video_id, "frame_index": frame.frame_index,
               "image_path": frame.image_path, "width": frame.width,
               "height": frame.height, "source_split": frame.source_split,
               "decisions": []}
        for det in frame.detections:
            stat = features.get((frame.video_id, det.track_id), {}) if det.track_id is not None else {}
            nframes = int(stat.get("track_length", 1))
            conf = det.confidence
            if cfg.method == "confidence":
                score = conf
            else:
                score = (cfg.weight_confidence * conf
                         + cfg.weight_support * stat.get("support", 0.0)
                         + cfg.weight_class_agreement * stat.get("class_agreement", 0.0)
                         + cfg.weight_area_stability * stat.get("area_stability", 0.0))
                if cfg.method == "temporal_uncertainty":
                    score -= cfg.penalty_uncertainty * stat.get("uncertainty", 0.0)
                score = max(0.0, min(1.0, score))
            if score < cfg.review_threshold:
                decision, reason = "reject", "low_quality_score"
            elif score < cfg.accept_threshold:
                decision, reason = "review", "uncertain_quality"
            else:
                decision, reason = "accept", "quality_pass"
            # Avoid losing short-lived pedestrians/riders/cyclists: route them to a
            # human rather than treating low temporal support as proof of absence.
            if cfg.method != "confidence" and nframes < cfg.min_track_frames:
                if det.class_id in cfg.critical_classes:
                    decision, reason = "review", "safety_critical_transient"
                elif conf < cfg.high_conf_shortcut and decision == "accept":
                    decision, reason = "review", "short_track"
            if det.track_id is None and cfg.method != "confidence" and decision == "accept":
                decision, reason = "review", "untracked_detection"
            row["decisions"].append({**det.to_dict(), "quality_score": round(score, 6),
                                     "decision": decision, "reason": reason,
                                     "track_features": stat})
        results.append(row)
    return results
