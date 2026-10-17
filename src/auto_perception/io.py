"""Data I/O helpers: JSONL, manifest generation, and safe dataset linking."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import yaml

from auto_perception.schema import FrameRecord


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open(encoding="utf-8") as file:
        for i, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f"Malformed JSON in {path}:{i}") from ex
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path}:{i}")
            out.append(payload)
    return out


def read_frames(path: str | Path) -> list[FrameRecord]:
    frames = [FrameRecord.from_dict(x) for x in read_jsonl(path)]
    seen: set[tuple[str, int]] = set()
    for record in frames:
        record.verify()
        if record.key in seen:
            raise ValueError(f"Duplicate frame key: {record.key}")
        seen.add(record.key)
    return sorted(frames, key=lambda x: x.key)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def link_or_copy(src: str | Path, dst: str | Path, mode: str = "symlink") -> None:
    src = Path(src).resolve(strict=True)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f"Would overwrite existing data: {dst}")
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown materialization mode: {mode}")
