import json
from pathlib import Path

from auto_perception.bdd import BDD_CLASSES, convert_bdd_det


def test_bdd_conversion_preserves_labels_and_empty_frames(tmp_path, make_image):
    image_dir = tmp_path / "bdd_images"
    image_dir.mkdir()
    make_image("bdd_images/one.jpg", (100, 50))
    make_image("bdd_images/two.jpg", (100, 50))
    ann = tmp_path / "labels.json"
    ann.write_text(json.dumps([
        {"name": "one.jpg", "labels": [
            {"category": "car", "box2d": {"x1": 10, "y1": 5, "x2": 50, "y2": 45}},
            {"category": "traffic light", "box2d": {"x1": 3, "y1": 3,
                                                     "x2": 10, "y2": 12}}],
         "attributes": {"weather": "clear"}},
        {"name": "two.jpg", "labels": []},
    ]), encoding="utf-8")
    out = tmp_path / "converted"
    result = convert_bdd_det(ann, image_dir, out, "train")
    assert result["images"] == 2
    assert result["objects"] == {"car": 1}
    assert (out / "images/train/one.jpg").is_symlink()
    label = (out / "labels/train/one.txt").read_text().strip().split()
    assert int(label[0]) == BDD_CLASSES.index("car")
    assert [float(x) for x in label[1:]] == [0.3, 0.5, 0.4, 0.8]
    assert (out / "labels/train/two.txt").read_text() == ""
    val = convert_bdd_det(ann, image_dir, out, "val")
    assert val["images"] == 2
    assert (out / "data.yaml").exists()
