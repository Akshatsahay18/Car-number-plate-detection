import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _to_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_list(value: str, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    model_path: str
    target_class: str
    confidence_threshold: float
    max_detections: int
    fallback_confidence_threshold: float
    fallback_max_detections: int
    enable_fallback_pass: bool
    use_gpu: bool
    cors_origins: list[str]


settings = Settings(
    model_path=os.getenv("MODEL_PATH", "models/best.pt"),
    target_class=os.getenv("TARGET_CLASS", "license_plate"),
    confidence_threshold=_to_float(os.getenv("CONFIDENCE_THRESHOLD"), 0.25),
    max_detections=_to_int(os.getenv("MAX_DETECTIONS"), 10),
    fallback_confidence_threshold=_to_float(os.getenv("FALLBACK_CONFIDENCE_THRESHOLD"), 0.02),
    fallback_max_detections=_to_int(os.getenv("FALLBACK_MAX_DETECTIONS"), 50),
    enable_fallback_pass=_to_bool(os.getenv("ENABLE_FALLBACK_PASS"), True),
    use_gpu=_to_bool(os.getenv("USE_GPU"), False),
    cors_origins=_to_list(
        os.getenv("CORS_ORIGINS"),
        ["http://localhost:5173"],
    ),
)
