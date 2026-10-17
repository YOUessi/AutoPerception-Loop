import json
from pathlib import Path

import pytest

from auto_perception.io import write_jsonl
from auto_perception.metrics import evaluate_pseudo
from auto_perception.mix import assemble_student_dataset
from auto_perception.pseudo import export_pseudo
from auto_perception.schema import Box, Detection, FrameRecord
from auto_perception.selection import SelectionConfig


def test_frame_quarantine_and_pseudo_audit(tmp_path, make_image):
    img1 = make_image("video01/000000.jpg")
    img2 = make_image("video01/000001.jpg")
    gt1 = Detection(Box(1, 1, 21, 21), 2)
    gt2 = Detection(Box(15, 2, 36, 25), 0)
    frames = [
        FrameRecord("video01", 0, str(img1), 100, 50,
                    (Detection(gt1.bbox, 2, 0.99, 1),)),
        FrameRecord("video01", 1, str(img2), 100, 50,
                    (Detection(gt2.bbox, 0, 0.66, 2),)),
    ]
    frames_file = tmp_path / "frames.jsonl"
    write_jsonl(frames_file, (f.to_dict() for f in frames))
    pseudo = tmp_path / "pseudo"
    cfg = SelectionConfig(method="confidence", accept_threshold=0.78,
                          review_threshold=0.48)
    report = export_pseudo(frames_file, pseudo, cfg)
    assert report["exported_complete_frames"] == 1
    assert report["quarantined_frames"] == 1
    assert (pseudo / "labels/train/video01/000000.txt").is_file()
    assert not (pseudo / "labels/train/video01/000001.txt").exists()
    review = (pseudo / "review_queue.jsonl").read_text()
    assert "uncertain_quality" in review
    gt = tmp_path / "gt.jsonl"
    write_jsonl(gt, (FrameRecord(f.video_id, f.frame_index, f.image_path,
                                 f.width, f.height, (g,), f.source_split).to_dict()
                      for f, g in zip(frames, (gt1, gt2))))
    audit = evaluate_pseudo(pseudo / "decisions.jsonl", gt)
    assert audit["overall"]["tp"] == 1
    assert audit["overall"]["fn"] == 1
    assert audit["overall"]["fp"] == 0
    assert audit["overall"]["recall"] == 0.5


def test_no_pseudolabel_from_val(tmp_path, make_image):
    image = make_image("frame.jpg")
    frame = FrameRecord("clip", 0, str(image), 100, 50, (), "val")
    frames_file = tmp_path / "val.jsonl"
    write_jsonl(frames_file, [frame.to_dict()])
    with pytest.raises(ValueError, match="validation/test"):
        export_pseudo(frames_file, tmp_path / "bad", SelectionConfig())


def test_student_mixing_only_human_val(tmp_path, make_image):
    labeled = tmp_path / "labeled"
    pseudo = tmp_path / "pseudo"
    for folder, name in [(labeled / "images/train", "a"),
                         (labeled / "images/val", "b"),
                         (pseudo / "images/train/subclip", "c")]:
        folder.mkdir(parents=True)
        make_image(str(folder.relative_to(tmp_path) / (name + ".jpg")))
        label_folder = Path(str(folder).replace("/images/", "/labels/"))
        label_folder.mkdir(parents=True)
        (label_folder / (name + ".txt")).write_text("2 0.5 0.5 0.2 0.2\n")
    (pseudo / "summary.json").write_text(json.dumps({"source_split": "train"}))
    result = assemble_student_dataset(labeled, pseudo, tmp_path / "student")
    assert result == {"labeled_train": 1, "pseudo_train": 1, "labeled_val": 1}
    assert len(list((tmp_path / "student/images/train").iterdir())) == 2
    assert len(list((tmp_path / "student/images/val").iterdir())) == 1


def test_student_rejects_overlap(tmp_path, make_image):
    labeled = tmp_path / "labeled"
    pseudo = tmp_path / "pseudo"
    img = make_image("actual.jpg")
    for root, split, stem in [(labeled, "train", "human"),
                              (labeled, "val", "humanval"),
                              (pseudo, "train", "pseudo")]:
        outimg = root / "images" / split / f"{stem}.jpg"
        outimg.parent.mkdir(parents=True, exist_ok=True)
        outimg.symlink_to(img)
        outlabel = root / "labels" / split / f"{stem}.txt"
        outlabel.parent.mkdir(parents=True, exist_ok=True)
        outlabel.write_text("2 0.5 0.5 0.2 0.2\n")
    (pseudo / "summary.json").write_text(json.dumps({"source_split": "train"}))
    with pytest.raises(ValueError, match="overlap"):
        assemble_student_dataset(labeled, pseudo, tmp_path / "mixed")
