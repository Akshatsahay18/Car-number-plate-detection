from pydantic import BaseModel, Field


class PlateDetection(BaseModel):
    label: str = Field(..., description="YOLO class label")
    confidence: float = Field(..., description="Detection confidence score")
    bbox: list[int] = Field(..., description="[x1, y1, x2, y2]")
    text: str = Field(default="", description="EasyOCR extracted text")


class PredictionResponse(BaseModel):
    filename: str
    total_detections: int
    detections: list[PlateDetection]
    warning: str = ""
