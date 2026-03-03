from __future__ import annotations

from pathlib import Path

import easyocr
import numpy as np
from ultralytics import YOLO

from .config import settings


class PlateDetector:
    def __init__(self) -> None:
        model_path = Path(settings.model_path)
        resolved_model = str(model_path) if model_path.exists() else "yolov8n.pt"
        self.model = YOLO(resolved_model)
        self.reader = easyocr.Reader(["en"], gpu=settings.use_gpu)
        self.loaded_model_path = resolved_model
        self.is_fallback_model = not model_path.exists()
        self.available_classes = self._extract_available_classes()
        self.target_class_present = self._target_class_present()

    def predict(self, image_bgr: np.ndarray) -> list[dict]:
        strict = self._run_pass(
            image_bgr=image_bgr,
            conf=settings.confidence_threshold,
            max_det=settings.max_detections,
            shape_filter=False,
            iou=0.7,
        )
        if strict:
            return strict[: settings.max_detections]

        if not settings.enable_fallback_pass:
            return []

        fallback = self._run_pass(
            image_bgr=image_bgr,
            conf=settings.fallback_confidence_threshold,
            max_det=settings.fallback_max_detections,
            shape_filter=True,
            iou=0.35,
        )
        return fallback[: settings.max_detections]

    def warning_message(self) -> str:
        if not settings.target_class:
            return ""

        if self.target_class_present:
            return ""

        if self.is_fallback_model:
            return (
                "Configured target class was not found because backend is using fallback "
                "model yolov8n.pt. Train your Roboflow model and place weights at models/best.pt."
            )

        return (
            f"Configured target class '{settings.target_class}' not found in model classes: "
            f"{', '.join(self.available_classes[:20])}"
        )

    def _run_pass(
        self,
        image_bgr: np.ndarray,
        conf: float,
        max_det: int,
        shape_filter: bool,
        iou: float,
    ) -> list[dict]:
        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            max_det=max_det,
            iou=iou,
            imgsz=1280,
            verbose=False,
        )
        if not results:
            return []

        names = results[0].names or {}
        detections: list[dict] = []
        image_h, image_w = image_bgr.shape[:2]
        image_area = float(max(1, image_h * image_w))

        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

            class_id = int(box.cls[0]) if box.cls is not None else -1
            label = str(names.get(class_id, f"class_{class_id}"))

            if settings.target_class and self._normalize(label) != self._normalize(settings.target_class):
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            ratio = box_w / max(1, box_h)
            area_ratio = (box_w * box_h) / image_area

            # Fallback pass keeps only plate-like geometry to reduce noisy boxes.
            if shape_filter and not (0.8 <= ratio <= 12.0 and 0.0003 <= area_ratio <= 0.35):
                continue

            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            crop = image_bgr[y1:y2, x1:x2]
            text = self._extract_text(crop)

            detections.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                }
            )

        if shape_filter:
            text_backed = []
            for item in detections:
                alnum = "".join(ch for ch in item["text"] if ch.isalnum())
                if len(alnum) >= 4:
                    text_backed.append(item)
            if text_backed:
                detections = text_backed

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return self._dedupe_overlaps(detections, iou_threshold=0.3)

    def _dedupe_overlaps(self, detections: list[dict], iou_threshold: float) -> list[dict]:
        kept: list[dict] = []
        for candidate in detections:
            if all(self._bbox_iou(candidate["bbox"], existing["bbox"]) < iou_threshold for existing in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _bbox_iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0

        a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        b_area = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(a_area + b_area - inter)

    def _extract_text(self, crop_bgr: np.ndarray) -> str:
        if crop_bgr.size == 0:
            return ""

        raw = self.reader.readtext(crop_bgr, detail=0, paragraph=False)
        cleaned = []

        for item in raw:
            filtered = "".join(ch for ch in item.upper() if ch.isalnum() or ch in {" ", "-"})
            filtered = " ".join(filtered.split())
            if filtered:
                cleaned.append(filtered)

        if not cleaned:
            return ""

        # Keep only a couple of candidates to avoid noisy OCR responses.
        return " | ".join(cleaned[:2])

    def _extract_available_classes(self) -> list[str]:
        names = self.model.names
        if isinstance(names, dict):
            return [str(v) for _, v in sorted(names.items(), key=lambda item: item[0])]
        if isinstance(names, list):
            return [str(v) for v in names]
        return []

    def _target_class_present(self) -> bool:
        if not settings.target_class:
            return True
        normalized_target = self._normalize(settings.target_class)
        return any(self._normalize(item) == normalized_target for item in self.available_classes)

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(ch for ch in value.lower().strip() if ch.isalnum())
