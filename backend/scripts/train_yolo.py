from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
import yaml
from ultralytics import YOLO

load_dotenv()


def default_data_yaml() -> Path:
    project = os.getenv("ROBOFLOW_PROJECT")
    version = os.getenv("ROBOFLOW_VERSION")
    if project and version:
        candidate = Path("data") / f"{project}-{version}" / "data.yaml"
        if candidate.exists():
            return candidate

    matches = list(Path("data").glob("**/data.yaml"))
    if matches:
        return matches[0]

    raise SystemExit(
        "Could not find data.yaml. Run scripts/download_roboflow_dataset.py first "
        "or pass --data path/to/data.yaml."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model for number plate detection.")
    parser.add_argument("--data", type=str, default="", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--run-name", type=str, default="plate_detector", help="Run directory name")
    parser.add_argument("--device", type=str, default="", help="Device, for example '0' or 'cpu'")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use for quick training (0.0-1.0).",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing run directory name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data) if args.data else default_data_yaml()

    if not data_yaml.exists():
        raise SystemExit(f"data.yaml does not exist: {data_yaml}")

    prepared_data_yaml = prepare_data_yaml(data_yaml)

    model = YOLO(args.model)
    results = model.train(
        data=str(prepared_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="runs",
        name=args.run_name,
        device=args.device or None,
        fraction=max(0.01, min(1.0, args.fraction)),
        exist_ok=args.exist_ok,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise SystemExit(f"best.pt not found at {best_weights}")

    target = Path("models") / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, target)
    print(f"Copied trained model to {target.resolve()}")


def prepare_data_yaml(data_yaml: Path) -> Path:
    """
    Roboflow may emit ../train-style paths depending on download location.
    This normalizes paths relative to the YAML directory so Ultralytics can resolve them.
    """
    with data_yaml.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    if not isinstance(content, dict):
        return data_yaml

    yaml_dir = data_yaml.parent
    changed = False

    for key in ("train", "val", "test"):
        value = content.get(key)
        if not isinstance(value, str):
            continue

        candidate = (yaml_dir / value).resolve()
        if candidate.exists():
            continue

        simplified = value.replace("../", "").lstrip("./")
        fallback_candidate = (yaml_dir / simplified).resolve()
        if fallback_candidate.exists():
            content[key] = simplified
            changed = True

    # Force an absolute base path so Ultralytics does not redirect relative paths
    # to its global datasets directory.
    absolute_base = str(yaml_dir.resolve())
    if content.get("path") != absolute_base:
        content["path"] = absolute_base
        changed = True

    if not changed:
        return data_yaml

    prepared = yaml_dir / "data.prepared.yaml"
    with prepared.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=False)

    print(f"Prepared data yaml at {prepared}")
    return prepared


if __name__ == "__main__":
    main()
