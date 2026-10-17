"""Create frozen, manifestable student training set using real + pseudo labels."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from auto_perception.bdd import BDD_CLASSES
from auto_perception.io import link_or_copy, write_json, write_jsonl, write_yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _add_split(src_images: Path, src_labels: Path, out: Path, split: str,
               prefix: str, mode: str, rows: list[dict[str, Any]]) -> int:
    count = 0
    for img in sorted(src_images.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = img.relative_to(src_images)
        label = src_labels / rel.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"Missing paired label for {img}: {label}")
        # Input frames may live in subdirs; use all path components for uniqueness.
        normalized = "__".join(rel.parts)
        name = f"{prefix}__{normalized}"
        destination_image = out / "images" / split / name
        destination_label = out / "labels" / split / (Path(name).stem + ".txt")
        link_or_copy(img, destination_image, mode)
        link_or_copy(label, destination_label, mode)
        rows.append({"split": split, "source": prefix, "source_image": str(img.resolve()),
                     "source_label": str(label.resolve()),
                     "output_image": str(destination_image)})
        count += 1
    return count


def assemble_student_dataset(
    labeled_dataset: str | Path, pseudo_dataset: str | Path, output_dir: str | Path,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    labeled, pseudo, output = map(Path, (labeled_dataset, pseudo_dataset, output_dir))
    if not (labeled / "images/train").exists() or not (labeled / "images/val").exists():
        raise ValueError("Labeled dataset must contain official train and val splits")
    if not (pseudo / "images/train").exists() or not (pseudo / "labels/train").exists():
        raise ValueError("Pseudo dataset must contain accepted train frames")
    # Pseudo data is always train-only. Validation comes exclusively from official GT.
    summary = __import__("json").loads((pseudo / "summary.json").read_text(encoding="utf-8"))
    if summary.get("source_split") != "train":
        raise ValueError("Pseudo dataset provenance is not train-only")
    # Prevent duplicate images (including symlinks) appearing as both GT and pseudo.
    human_images = {x.resolve() for x in (labeled / "images/train").rglob("*")
                    if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES}
    pseudo_images = {x.resolve() for x in (pseudo / "images/train").rglob("*")
                     if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES}
    overlap = human_images & pseudo_images
    if overlap:
        raise ValueError(f"Labeled/pseudo train images overlap ({len(overlap)}), e.g. {next(iter(overlap))}")
    rows: list[dict[str, Any]] = []
    counts = Counter()
    counts["labeled_train"] = _add_split(labeled / "images/train", labeled / "labels/train",
                                          output, "train", "human", link_mode, rows)
    counts["pseudo_train"] = _add_split(pseudo / "images/train", pseudo / "labels/train",
                                         output, "train", "pseudo", link_mode, rows)
    counts["labeled_val"] = _add_split(labeled / "images/val", labeled / "labels/val",
                                        output, "val", "human", link_mode, rows)
    write_yaml(output / "data.yaml", {"path": str(output.resolve()),
        "train": "images/train", "val": "images/val", "nc": len(BDD_CLASSES),
        "names": dict(enumerate(BDD_CLASSES))})
    write_jsonl(output / "provenance.jsonl", rows)
    report = dict(counts)
    write_json(output / "summary.json", report)
    return report
