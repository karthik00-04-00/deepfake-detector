from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
import tempfile
import os

from src.inference.image_inference import predict_image
from src.inference.video_inference import load_models as load_video_models
from src.inference.video_inference import predict_video


app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/inference/static"), name="static")


# -------------------------
# Load video models at startup (once)
# -------------------------
@app.on_event("startup")
def load_models_at_startup():
    load_video_models(
        backbone_ckpt_path="outputs/models/best_baseline.pth",
        lstm_ckpt_path="outputs/models/video_lstm_last.pth"
    )


# -------------------------
# Serve Web UI
# -------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_path = Path("src/inference/static/index.html")
    return html_path.read_text(encoding="utf-8")


# -------------------------
# Image Inference Endpoint
# -------------------------
@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    temp_path = temp_dir / f"{uuid.uuid4()}.jpg"

    try:
        # Save uploaded file temporarily
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run deterministic inference
        result = predict_image(str(temp_path))

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink()


# -------------------------
# Video Inference Endpoint
# -------------------------
@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid video."
        )

    # Use actual file extension
    suffix = Path(file.filename).suffix if file.filename else ".mp4"

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        result = predict_video(str(tmp_path))
        return result
    finally:
        if tmp_path.exists():
            tmp_path.unlink()