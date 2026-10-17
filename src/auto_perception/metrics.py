"""Offline evaluation of pseudo-label accept/reject decisions against hidden GT.

Pseudo precision/recall is NOT COCO mAP. For model mAP use model.val().
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from auto_perception.io import read_frames, read_jsonl
from auto_perception.schema import Box, Detection


def evaluate_pseudo(
    decisions_jsonl: str | Path, ground_truth_jsonl: str | Path, iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    decisions = read_jsonl(decisions_jsonl)
    ground_truth = read_frames(ground_truth_jsonl)
    gt_index = {f.key: f for f in ground_truth}
    pred_index = {}
    for item in decisions:
        key = str(item["video_id"]), int(item["frame_index"])
        if key in pred_index:
            raise ValueError(f"Duplicate pseudo frame {key}")
        pred_index[key] = item
    if set(pred_index) != set(gt_index):
        raise ValueError("Evaluation requires identical frame keys, including empty frames")
    total = Counter()
    per_class: dict[int, Counter] = defaultdict(Counter)
    for key, truth in gt_index.items():
        preds = [Detection.from_dict(d) for d in pred_index[key]["decisions"]
                 if d["decision"] == "accept"]
        # Highest-confidence-first greedy matching, class + IoU must both match.
        gt = list(truth.detections)
        used = set()
        for pred in sorted(preds, key=lambda p: p.confidence, reverse=True):
            candidates = [(i, pred.bbox.iou(target.bbox)) for i, target in enumerate(gt)
                          if i not in used and target.class_id == pred.class_id]
            best = max(candidates, key=lambda x: x[1], default=None)
            if best is not None and best[1] >= iou_threshold:
                used.add(best[0])
                total["tp"] += 1
                per_class[pred.class_id]["tp"] += 1
            else:
                total["fp"] += 1
                per_class[pred.class_id]["fp"] += 1
        for i, target in enumerate(gt):
            if i not in used:
                total["fn"] += 1
                per_class[target.class_id]["fn"] += 1

    def stats(counts: Counter) -> dict[str, float | int]:
        tp, fp, fn = (counts[name] for name in ("tp", "fp", "fn"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(2 * precision * recall / (precision + recall), 6)
                if precision + recall else 0.0}

    return {"frames": len(gt_index), "iou_threshold": iou_threshold,
            "overall": stats(total),
            "per_class": {str(k): stats(v) for k, v in sorted(per_class.items())},
            "warning": "This evaluates acceptance quality, not detector COCO mAP."
            }
