
from pathlib import Path
from functools import lru_cache
import base64
import re

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "ai" / "models" / "license_plate.pt"


@lru_cache(maxsize=1)
def load_models():
    """Chỉ tải mô hình một lần."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy YOLO model: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        download_enabled=False
    )

    return model, reader


def recognize_plate(image_bytes):
    """Nhận diện biển số từ dữ liệu ảnh."""

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("File không phải ảnh hợp lệ")

    model, reader = load_models()

    results = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    height, width = image.shape[:2]
    plates = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]

        crop = cv2.resize(
            crop, None, fx=2, fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        ocr_results = reader.readtext(
            crop,
            detail=1,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
        )

        # Đọc các vùng chữ từ trên xuống dưới
        ocr_results.sort(
            key=lambda item: (
                item[0][0][1],
                item[0][0][0]
            )
        )

        raw_text = "".join(
            item[1] for item in ocr_results
        )

        plate_number = re.sub(
            r"[^A-Z0-9]",
            "",
            raw_text.upper()
        )

        confidence = float(box.conf[0])

        plates.append({
            "plate_number": plate_number,
            "confidence": round(confidence, 3),
            "bounding_box": [x1, y1, x2, y2]
        })

        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            image,
            plate_number or "Unknown",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Chuyển ảnh kết quả thành Base64
    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        raise ValueError("Không thể tạo ảnh kết quả")

    encoded_image = base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

    return {
        "success": True,
        "total_plates": len(plates),
        "plates": plates,
        "result_image": (
            "data:image/jpeg;base64," + encoded_image
        )
    }
