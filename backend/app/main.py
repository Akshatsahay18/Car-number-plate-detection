from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .detector import PlateDetector
from .schemas import PredictionResponse

app = FastAPI(title="YOLO + EasyOCR Number Plate Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = PlateDetector()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_path": detector.loaded_model_path,
        "target_class": settings.target_class,
        "is_fallback_model": detector.is_fallback_model,
        "available_classes": detector.available_classes,
        "warning": detector.warning_message(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    content = await file.read()
    np_buffer = np.frombuffer(content, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Image could not be decoded.")

    detections = detector.predict(image_bgr)
    return PredictionResponse(
        filename=file.filename or "uploaded_image",
        total_detections=len(detections),
        detections=detections,
        warning=detector.warning_message(),
    )
