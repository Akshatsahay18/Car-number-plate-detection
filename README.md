# YOLO + EasyOCR Car Number Plate Detector

This project gives you:
- `backend/`: FastAPI API using YOLO for plate detection and EasyOCR for text extraction.
- `frontend/`: React (Vite) web UI for uploading car images and viewing detection/OCR output.
- Roboflow dataset flow for YOLO training.

## 1) Backend setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend/.env` with your Roboflow values:
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_PROJECT`
- `ROBOFLOW_VERSION`

Optional values:
- `TARGET_CLASS=license_plate` (change if your dataset class has another name)
- `USE_GPU=true` if you have CUDA and want OCR on GPU.

## 2) Download Roboflow dataset + train YOLO

From `backend/`:

```powershell
python scripts\download_roboflow_dataset.py
python scripts\train_yolo.py --epochs 50 --imgsz 640
```

After training, best weights are copied to:
- `backend/models/best.pt`

## 3) Start backend API

From `backend/` with venv active:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:
- `http://127.0.0.1:8000/health`

## 4) Frontend setup

Open another terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Frontend URL:
- `http://localhost:5173`

## API

### `POST /predict`
- Form field: `file` (image)
- Response:
  - detected boxes
  - confidence
  - OCR text per detection

## Notes

- If `models/best.pt` does not exist, backend falls back to `yolov8n.pt` (generic model, poor plate accuracy).
- Best accuracy comes from training on your plate dataset from Roboflow.

## Live hosting (Render)

This repo supports one-service deployment (FastAPI + built React frontend) using:
- `Dockerfile`
- `render.yaml`

### Deploy steps

1. Go to Render dashboard and create a **Web Service** from this GitHub repo.
2. Render will auto-detect `render.yaml`.
3. Deploy branch `main`.
4. Wait for build and open the generated URL.

Your app endpoints after deploy:
- `/` -> frontend UI
- `/health` -> backend health
- `/predict` -> prediction API
