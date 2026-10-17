"""Training, evaluation & reproducibility adapter.

All GPU/model operations are intentionally lazy imports, to keep CI CPU-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from auto_perception.io import write_json


def _yolo(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install optional vision deps: pip install -e ".[vision]"') from exc
    return YOLO(weights)


def train_detector(data_yaml: str, weights: str, output_dir: str, run_name: str,
                   epochs: int = 30, imgsz: int = 640, batch: int = 8,
                   device: str = "0", seed: int = 42, workers: int = 4) -> dict[str, Any]:
    if epochs < 1 or batch < 1 or imgsz < 64:
        raise ValueError("Invalid training budget")
    if not Path(data_yaml).is_file():
        raise FileNotFoundError(data_yaml)
    run_path = Path(output_dir) / run_name
    if run_path.exists():
        raise FileExistsError(f"Refusing to append a run name and mislabel results: {run_path}")
    model = _yolo(weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                device=device, seed=seed, workers=workers,
                project=output_dir, name=run_name, exist_ok=False,
                deterministic=True, plots=True, save=True)
    result = {"run_name": run_name, "project": output_dir, "dataset": data_yaml,
              "model_initialization": weights, "epochs": epochs, "imgsz": imgsz,
              "batch": batch, "seed": seed, "device": device,
              "note": "Inspect run output for metrics and best.pt; no metrics are invented."}
    write_json(Path(output_dir) / run_name / "experiment_setup.json", result)
    return result


def evaluate_detector(data_yaml: str, weights: str, output_dir: str,
                      split: str = "val", imgsz: int = 640, device: str = "0") -> dict[str, Any]:
    if split != "val":
        raise ValueError("Only held-out validation is supported by this CLI")
    if not Path(data_yaml).is_file():
        raise FileNotFoundError(data_yaml)
    model = _yolo(weights)
    result = model.val(data=data_yaml, split=split, imgsz=imgsz,
                       device=device, project=output_dir, name="evaluation", exist_ok=True)
    report = {"data": data_yaml, "weights": weights, "split": split,
              "map_50_95": float(result.box.map), "map_50": float(result.box.map50),
              "map_per_class": [float(x) for x in result.box.maps],
              "speed_ms": {str(k): float(v) for k, v in result.speed.items()}}
    write_json(Path(output_dir) / "evaluation_metrics.json", report)
    return report
