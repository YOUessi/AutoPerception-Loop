"""Single public entry point for deterministic pipelines and GPU jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from auto_perception.bdd import convert_bdd_det
from auto_perception.io import load_yaml, write_json
from auto_perception.metrics import evaluate_pseudo
from auto_perception.mix import assemble_student_dataset
from auto_perception.pseudo import export_pseudo
from auto_perception.selection import SelectionConfig
from auto_perception.track import track_video
from auto_perception.training import evaluate_detector, train_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoperception",
        description="Reproducible 2D driving perception + temporal pseudo-label lab")
    cmds = parser.add_subparsers(dest="cmd", required=True)

    det = cmds.add_parser("prepare-bdd", help="Convert BDD100K detection JSON -> YOLO labels")
    det.add_argument("--annotations", required=True)
    det.add_argument("--images-dir", required=True)
    det.add_argument("--out", required=True)
    det.add_argument("--split", required=True, choices=("train", "val"))
    det.add_argument("--limit", type=int)
    det.add_argument("--fail-on-missing", action="store_true")
    det.add_argument("--link-mode", default="symlink", choices=("symlink", "copy", "hardlink"))

    train = cmds.add_parser("train", help="Ultralytics YOLO detector baseline or student")
    train.add_argument("--data", required=True)
    train.add_argument("--weights", default="yolov8n.pt")
    train.add_argument("--out", default="runs")
    train.add_argument("--name", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=4)

    eva = cmds.add_parser("eval", help="Compute YOLO COCO-style detection mAP on val split")
    eva.add_argument("--data", required=True)
    eva.add_argument("--weights", required=True)
    eva.add_argument("--out", default="runs/eval")
    eva.add_argument("--imgsz", type=int, default=640)
    eva.add_argument("--device", default="0")

    track = cmds.add_parser("track-video", help="YOLO + ByteTrack inference -> JSONL")
    track.add_argument("--video", required=True)
    track.add_argument("--weights", required=True)
    track.add_argument("--frames-dir", required=True)
    track.add_argument("--out", required=True)
    track.add_argument("--device", default="0")
    track.add_argument("--conf", type=float, default=0.08)
    track.add_argument("--imgsz", type=int, default=640)
    track.add_argument("--source-split", default="train", choices=("train", "val", "test"))
    track.add_argument("--max-frames", type=int)

    select = cmds.add_parser("select-pseudo", help="Confidence/temporal 3-way pseudo-label gate")
    select.add_argument("--tracks", required=True)
    select.add_argument("--config", default="configs/pseudo.yaml")
    select.add_argument("--out", required=True)
    select.add_argument("--link-mode", default="symlink", choices=("symlink", "copy", "hardlink"))

    audit = cmds.add_parser("audit-pseudo", help="Compare pseudo selection vs hidden ground truth")
    audit.add_argument("--decisions", required=True)
    audit.add_argument("--ground-truth", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--iou", type=float, default=0.5)

    mix = cmds.add_parser("make-student", help="Mix official labeled train and accepted train-only pseudo")
    mix.add_argument("--labeled", required=True)
    mix.add_argument("--pseudo", required=True)
    mix.add_argument("--out", required=True)
    mix.add_argument("--link-mode", default="symlink", choices=("symlink", "copy", "hardlink"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "prepare-bdd":
        output = convert_bdd_det(args.annotations, args.images_dir, args.out, args.split,
                                 args.link_mode, args.limit, args.fail_on_missing)
    elif args.cmd == "train":
        output = train_detector(args.data, args.weights, args.out, args.name,
                                args.epochs, args.imgsz, args.batch,
                                args.device, args.seed, args.workers)
    elif args.cmd == "eval":
        output = evaluate_detector(args.data, args.weights, args.out,
                                   imgsz=args.imgsz, device=args.device)
    elif args.cmd == "track-video":
        output = track_video(args.video, args.weights, args.frames_dir,
                             args.out, args.conf, args.imgsz, args.device,
                             args.source_split, args.max_frames)
    elif args.cmd == "select-pseudo":
        cfg = SelectionConfig.from_mapping(load_yaml(args.config))
        output = export_pseudo(args.tracks, args.out, cfg, args.link_mode)
    elif args.cmd == "audit-pseudo":
        output = evaluate_pseudo(args.decisions, args.ground_truth, args.iou)
        write_json(args.out, output)
    elif args.cmd == "make-student":
        output = assemble_student_dataset(args.labeled, args.pseudo,
                                          args.out, args.link_mode)
    else:
        raise ValueError("Unknown subcommand")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
