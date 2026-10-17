import pytest

from auto_perception.schema import Box, Detection, FrameRecord, safe_id, safe_name


def test_yolo_normalization_and_clip():
    xywh = Box(-10, 5, 50, 45).to_yolo(100, 50)
    assert xywh == pytest.approx((0.25, 0.5, 0.5, 0.8))
    assert Box(0, 0, 10, 10).iou(Box(5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_boxes_and_frame_validation():
    with pytest.raises(ValueError):
        Box(2, 2, 2, 3).to_yolo(10, 10)
    det = Detection(Box(0, 0, 10, 10), 2, 0.8, 5)
    frame = FrameRecord("video01", 4, "/tmp/img.jpg", 100, 50, (det,))
    frame.verify()
    assert FrameRecord.from_dict(frame.to_dict()) == frame
    with pytest.raises(ValueError):
        FrameRecord("x", 0, "foo", 0, 50).verify()


@pytest.mark.parametrize("candidate", ["../escape", "a/b", "..", "", "a\\b", "a b"])
def test_safe_id_rejects_unsafe(candidate):
    with pytest.raises(ValueError):
        safe_id(candidate)


@pytest.mark.parametrize("candidate", ["../x.jpg", "/tmp/a.jpg", "a\\b.jpg", ""])
def test_name_rejects_path_traversal(candidate):
    with pytest.raises(ValueError):
        safe_name(candidate)
