
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Vietnamese License Plate Recognition API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def home():
    return {"message": "License Plate Recognition API"}

@app.get("/api/health")
def health():
    return {"status": "ok"}

from fastapi import UploadFile, File, HTTPException
from backend.app.services.plate_service import recognize_plate


@app.post("/api/recognize")
def recognize(file: UploadFile = File(...)):

    allowed_types = {
        "image/jpeg",
        "image/png"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Chỉ hỗ trợ ảnh JPG và PNG"
        )

    # Giới hạn kích thước tải lên: 10 MB
    max_size = 10 * 1024 * 1024
    image_bytes = file.file.read(max_size + 1)

    if len(image_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail="Ảnh không được vượt quá 10 MB"
        )

    try:
        return recognize_plate(image_bytes)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
