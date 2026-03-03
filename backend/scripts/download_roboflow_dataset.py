from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    api_key = required_env("ROBOFLOW_API_KEY")
    workspace = required_env("ROBOFLOW_WORKSPACE")
    project_name = required_env("ROBOFLOW_PROJECT")
    version = int(required_env("ROBOFLOW_VERSION"))

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8", location=str(output_dir))

    print("Downloaded dataset:")
    print(f"location: {dataset.location}")
    print(f"data yaml: {Path(dataset.location) / 'data.yaml'}")


if __name__ == "__main__":
    main()
