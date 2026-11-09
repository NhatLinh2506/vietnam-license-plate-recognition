import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/recognize";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleFileChange(event) {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    if (!["image/jpeg", "image/png"].includes(selectedFile.type)) {
      setError("Chỉ hỗ trợ ảnh JPG hoặc PNG");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Ảnh không được vượt quá 10 MB");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  }

  async function handleRecognize() {
    if (!file) {
      setError("Vui lòng chọn ảnh trước");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Nhận diện thất bại");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Không thể kết nối Backend");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Hệ thống nhận diện biển số xe AI</h1>
        <p>YOLO + EasyOCR + FastAPI + React</p>
      </header>

      <main className="container">
        <section className="card">
          <h2>Nhận diện biển số xe</h2>
          <p>Tải hình ảnh phương tiện lên để AI phân tích.</p>

          <label className="upload-box">
            <span>📷 Chọn hình ảnh phương tiện</span>
            <input
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleFileChange}
            />
          </label>

          {preview && (
            <div className="image-section">
              <h3>Ảnh đầu vào</h3>
              <img src={preview} alt="Ảnh phương tiện" />
            </div>
          )}

          <button
            className="recognize-btn"
            onClick={handleRecognize}
            disabled={loading || !file}
          >
            {loading ? "Đang nhận diện..." : "Nhận diện biển số"}
          </button>

          {error && <p className="error">{error}</p>}
        </section>

        {result && (
          <section className="card">
            <h2>Kết quả nhận diện</h2>

            <p>
              Số biển số phát hiện: <strong>{result.total_plates}</strong>
            </p>

            {result.plates.map((plate, index) => (
              <div className="plate-card" key={index}>
                <h3>Biển số {index + 1}</h3>
                <div className="plate-number">
                  {plate.plate_number || "Chưa đọc được"}
                </div>
                <p>Độ tin cậy YOLO: {(plate.confidence * 100).toFixed(1)}%</p>
              </div>
            ))}

            {result.result_image && (
              <div className="image-section">
                <h3>Ảnh đã xử lý</h3>
                <img src={result.result_image} alt="Ảnh nhận diện biển số" />
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
