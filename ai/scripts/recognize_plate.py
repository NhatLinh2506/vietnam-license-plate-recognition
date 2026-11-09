
from pathlib import Path
import re
import cv2
import easyocr
from ultralytics import YOLO

# ==================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "ai" / "models" / "license_plate.pt"
IMAGE_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "results"

# Chỉ xử lý 5 ảnh đầu tiên để kiểm thử
TEST_LIMIT = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. KIỂM TRA FILE VÀ THƯ MỤC
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình YOLO: {MODEL_PATH}"
    )

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục ảnh: {IMAGE_DIR}"
    )

# Tìm tất cả ảnh trong data/raw
image_files = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

total_images = len(image_files)

print(f"Tổng số ảnh tìm thấy: {total_images}")

if not image_files:
    raise FileNotFoundError(
        "Không tìm thấy ảnh JPG, JPEG hoặc PNG trong data/raw"
    )

# Giới hạn số lượng ảnh thử nghiệm
image_files = image_files[:TEST_LIMIT]

print(f"Số ảnh sẽ xử lý: {len(image_files)}")

# ==================================================
# 3. KHỞI TẠO MÔ HÌNH YOLO
# ==================================================

print("\nĐang tải mô hình YOLO...")

model = YOLO(str(MODEL_PATH))

print("YOLO đã khởi tạo thành công!")

# ==================================================
# 4. KHỞI TẠO EASYOCR
# ==================================================

print("\nĐang khởi tạo EasyOCR...")

try:
    # Dùng model đã tải tại thư mục mặc định
    # Không tự động tải lại từ internet
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        download_enabled=False
    )

    print("EasyOCR đã khởi tạo thành công!")

except Exception as exc:
    raise RuntimeError(
        "Không thể khởi tạo EasyOCR. "
        "Hãy kiểm tra các model đã tải trong "
        "thư mục mặc định của EasyOCR. "
        f"Chi tiết: {exc}"
    ) from exc

# ==================================================
# 5. HÀM CHUẨN HÓA BIỂN SỐ
# ==================================================

def normalize_plate(text):
    """
    Chuyển kết quả OCR thành chữ in hoa,
    loại bỏ khoảng trắng và ký tự đặc biệt.
    """

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ==================================================
# 6. XỬ LÝ TỪNG ẢNH
# ==================================================

success_count = 0
detected_count = 0

for index, image_path in enumerate(image_files, start=1):

    print("\n" + "=" * 50)
    print(f"Đang xử lý ảnh {index}/{len(image_files)}")
    print(f"Tên ảnh: {image_path.name}")

    # Đọc ảnh (hỗ trợ đường dẫn Unicode trên Windows)
    try:
        import numpy as np

        image_data = np.fromfile(
            str(image_path),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_data,
            cv2.IMREAD_COLOR
        )

    except (OSError, ValueError, cv2.error):
        image = None

    if image is None:
        print("Không thể đọc ảnh. Bỏ qua.")
        continue

    # ==================================================
    # 7. PHÁT HIỆN BIỂN SỐ BẰNG YOLO
    # ==================================================

    results = model.predict(
        source=image,
        conf=0.25,
        imgsz=640,
        device="cpu",
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) == 0:
        print("YOLO không phát hiện được biển số.")

        # Vẫn lưu ảnh để kiểm tra
        result_path = OUTPUT_DIR / (
            f"{index:04d}_result.jpg"
        )

        cv2.imwrite(str(result_path), image)
        continue

    detected_count += 1

    print(f"Số biển số phát hiện: {len(boxes)}")

    height, width = image.shape[:2]

    # ==================================================
    # 8. XỬ LÝ TỪNG BIỂN SỐ
    # ==================================================

    for plate_index, box in enumerate(boxes, start=1):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Giới hạn tọa độ trong ảnh
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        if x2 <= x1 or y2 <= y1:
            print("Vùng biển số không hợp lệ.")
            continue

        # ==================================================
        # 9. CẮT VÙNG BIỂN SỐ
        # ==================================================

        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        # Phóng to vùng biển số 2 lần
        plate_crop = cv2.resize(
            plate_crop,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

        # ==================================================
        # 10. NHẬN DIỆN KÝ TỰ BẰNG EASYOCR
        # ==================================================

        ocr_results = reader.readtext(
            plate_crop,
            detail=1,
            paragraph=False,
            allowlist=(
                "0123456789"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "-."
            )
        )

        # Sắp xếp vùng chữ theo chiều từ trên xuống
        # rồi từ trái sang phải (bản thử nghiệm)
        ocr_results = sorted(
            ocr_results,
            key=lambda item: (
                round(
                    (
                        item[0][0][1]
                        + item[0][2][1]
                    ) / 2 / 20
                ),
                item[0][0][0]
            )
        )

        raw_text = "".join(
            item[1] for item in ocr_results
        )

        plate_number = normalize_plate(raw_text)

        yolo_confidence = float(box.conf[0])

        print(f"\nBiển số thứ {plate_index}")
        print(f"OCR gốc: {raw_text}")
        print(
            f"Kết quả: "
            f"{plate_number or 'Chưa đọc được'}"
        )
        print(
            f"Độ tin cậy YOLO: "
            f"{yolo_confidence:.2f}"
        )

        if plate_number:
            success_count += 1

        # ==================================================
        # 11. LƯU ẢNH BIỂN SỐ ĐÃ CẮT
        # ==================================================

        crop_path = OUTPUT_DIR / (
            f"{index:04d}_plate_{plate_index}.jpg"
        )

        cv2.imwrite(
            str(crop_path),
            plate_crop
        )

        # ==================================================
        # 12. VẼ KHUNG BIỂN SỐ
        # ==================================================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = plate_number or "Unknown"

        cv2.putText(
            image,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ==================================================
    # 13. LƯU ẢNH KẾT QUẢ
    # ==================================================

    result_path = OUTPUT_DIR / (
        f"{index:04d}_result.jpg"
    )

    cv2.imwrite(
        str(result_path),
        image
    )

    print(f"\nĐã lưu: {result_path.name}")

# ==================================================
# 14. THỐNG KÊ KẾT QUẢ
# ==================================================

print("\n" + "=" * 50)
print("HOÀN THÀNH NHẬN DIỆN!")
print("=" * 50)

print(f"Tổng số ảnh trong dataset: {total_images}")
print(f"Số ảnh đã chọn xử lý: {len(image_files)}")
print(f"Số ảnh phát hiện biển số: {detected_count}")
print(f"Số biển số OCR có văn bản: {success_count}")

print(f"\nThư mục kết quả: {OUTPUT_DIR}")
print("=" * 50)
