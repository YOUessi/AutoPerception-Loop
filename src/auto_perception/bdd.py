"""BDD100K 2D detection annotation conversion (preserve official split)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from auto_perception.io import link_or_copy, write_json, write_jsonl, write_yaml
from auto_perception.schema import Box, safe_name

# Keep raw BDD names. Remapping to coarse classes can hide rider/bicycle distinctions.
BDD_CLASSES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
CLASS_MAP = {name: index for index, name in enumerate(BDD_CLASSES)}


def _format_yolo(box: Box, width: int, height: int, class_id: int) -> str | None:
    clipped = box.clip(width, height)
    if not clipped.valid:
        return None
    xywh = clipped.to_yolo(width, height)
    return " ".join((str(class_id), *(f"{v:.8f}" for v in xywh)))


def convert_bdd_det(
    annotations: str | Path, images_dir: str | Path, output_dir: str | Path,
    split: str, mode: str = "symlink", limit: int | None = None,
    fail_on_missing: bool = False,
) -> dict[str, Any]:
    if split not in {"train", "val"}:
        raise ValueError("Use official BDD train/val split; test has no ground-truth labels")
    rows = json.loads(Path(annotations).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("BDD detection annotations must be a list")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    output = Path(output_dir)
    images_root = Path(images_dir)
    image_dest = output / "images" / split
    label_dest = output / "labels" / split
    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = Counter()
    missing = []
    processed = 0
    for item in rows:
        if limit is not None and processed >= limit:
            break
        filename = safe_name(str(item["name"]))
        image = images_root / filename
        if not image.is_file():
            missing.append(filename)
            if fail_on_missing:
                raise FileNotFoundError(image)
            continue
        with Image.open(image) as im:
            width, height = im.size
        labels = []
        for obj in item.get("labels", []):
            category = str(obj.get("category", ""))
            if category not in CLASS_MAP or "box2d" not in obj:
                continue
            box = obj["box2d"]
            b = Box(float(box["x1"]), float(box["y1"]),
                    float(box["x2"]), float(box["y2"]))
            record = _format_yolo(b, width, height, CLASS_MAP[category])
            if record is not None:
                labels.append(record)
                counts[category] += 1
        label_path = label_dest / (Path(filename).stem + ".txt")
        if label_path.exists():
            raise FileExistsError(f"Label already exists: {label_path}")
        link_or_copy(image, image_dest / filename, mode)
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""),
                              encoding="utf-8")
        manifest.append({"name": filename, "width": width, "height": height,
                         "split": split, "attributes": item.get("attributes", {}),
                         "labeled_objects": len(labels)})
        processed += 1
    write_jsonl(output / f"manifest_{split}.jsonl", manifest)
    write_yaml(output / "data.yaml", {
        "path": str(output.resolve()), "train": "images/train", "val": "images/val",
        "names": dict(enumerate(BDD_CLASSES)), "nc": len(BDD_CLASSES),
    })
    summary = {"split": split, "images": processed, "objects": dict(counts),
               "missing_images": len(missing), "missing_examples": missing[:20],
               "label_root": str(label_dest), "image_root": str(image_dest)}
    write_json(output / f"summary_{split}.json", summary)
    return summary
