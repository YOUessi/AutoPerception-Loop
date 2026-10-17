"""Pseudo-label export with conservative frame-level quarantine."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from auto_perception.io import link_or_copy, read_frames, read_jsonl, write_json, write_jsonl
from auto_perception.schema import Box, safe_id
from auto_perception.selection import SelectionConfig, select_frames


def export_pseudo(
    tracks_jsonl: str | Path, output_dir: str | Path, cfg: SelectionConfig,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    raw_frames = read_jsonl(tracks_jsonl)
    if any("source_split" not in item for item in raw_frames):
        raise ValueError("Every track record must explicitly declare source_split")
    frames = read_frames(tracks_jsonl)
    if any(frame.source_split != "train" for frame in frames):
        raise ValueError("Pseudo-training data must not include validation/test frames")
    decisions = select_frames(frames, cfg)
    output = Path(output_dir)
    write_jsonl(output / "decisions.jsonl", decisions)
    review_rows: list[dict[str, Any]] = []
    counts = Counter()
    exported = 0
    quarantined = 0
    empty = 0
    for item in decisions:
        category_counts = Counter(d["decision"] for d in item["decisions"])
        counts.update(category_counts)
        if not item["decisions"]:
            empty += 1
            continue  # never silently turn empty predictions into negative labels
        unsafe = [d for d in item["decisions"] if d["decision"] != "accept"]
        for det in unsafe:
            review_rows.append({"video_id": item["video_id"],
                                "frame_index": item["frame_index"],
                                "image_path": item["image_path"], **det})
        if unsafe:
            quarantined += 1
            continue  # no partial labels -> fewer false negatives during training
        video = safe_id(item["video_id"])
        source = Path(item["image_path"])
        if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Unsupported training image suffix: {source}")
        basename = f"{int(item['frame_index']):06d}"
        image_dest = output / "images" / "train" / video / (basename + source.suffix.lower())
        label_dest = output / "labels" / "train" / video / (basename + ".txt")
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        labels = []
        for det in item["decisions"]:
            normalized = Box(**det["bbox"]).to_yolo(item["width"], item["height"])
            labels.append(" ".join([str(det["class_id"]), *[f"{x:.8f}" for x in normalized]]))
        link_or_copy(source, image_dest, mode=link_mode)
        label_dest.write_text("\n".join(labels) + "\n", encoding="utf-8")
        exported += 1
    write_jsonl(output / "review_queue.jsonl", review_rows)
    report = {"source_split": "train", "input_frames": len(frames),
              "exported_complete_frames": exported,
              "quarantined_frames": quarantined, "empty_prediction_frames_skipped": empty,
              "detections_by_status": dict(counts), "selection_method": cfg.method,
              "selection_config": vars(cfg),
              "notes": ["All detections in an exported frame must be accepted.",
                        "No detections does not imply empty ground truth.",
                        "A human audit is mandatory before treating pseudo-labels as truth."]}
    write_json(output / "summary.json", report)
    return report
