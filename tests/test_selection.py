from pathlib import Path

from auto_perception.schema import Box, Detection, FrameRecord
from auto_perception.selection import SelectionConfig, select_frames


def build(seq: int, cls: int = 2, tid: int | None = 12, conf: float = 0.96):
    return FrameRecord("clipA", seq, "/tmp/unused.jpg", 100, 60,
                       (Detection(Box(10 + seq, 10, 30 + seq, 30), cls, conf, tid),))


def test_temporal_track_agreement_and_critical_transients():
    frames = [build(0), build(1), build(2), build(3, cls=0, tid=44)]
    decisions = select_frames(frames, SelectionConfig())
    assert all(decisions[i]["decisions"][0]["decision"] == "accept" for i in range(3))
    critical = decisions[3]["decisions"][0]
    assert critical["decision"] == "review"
    assert critical["reason"] == "safety_critical_transient"


def test_confidence_ablation_and_invalid_configs():
    decision = select_frames([build(0, conf=0.9)],
                             SelectionConfig(method="confidence"))[0]["decisions"][0]
    assert decision["decision"] == "accept"
    from pytest import raises
    with raises(ValueError):
        SelectionConfig(accept_threshold=0.3, review_threshold=0.5).verify()
    with raises(ValueError):
        SelectionConfig(method="unknown").verify()
