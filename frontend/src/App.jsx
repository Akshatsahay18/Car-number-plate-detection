import { useEffect, useMemo, useState } from "react";
import { detectPlate, getHealth } from "./api";

function App() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendWarning, setBackendWarning] = useState("");
  const [result, setResult] = useState(null);

  const previewUrl = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onFileChange = (incoming) => {
    setError("");
    setResult(null);
    setFile(incoming || null);
  };

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const health = await getHealth();
        setBackendWarning(health.warning || "");
      } catch {
        setBackendWarning("Backend is not reachable. Ensure API is running on port 8000.");
      }
    };
    fetchHealth();
  }, []);

  const handleDrop = (event) => {
    event.preventDefault();
    setDragActive(false);
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFileChange(dropped);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Select an image first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await detectPlate(file);
      setResult(data);
      setBackendWarning(data.warning || "");
    } catch (err) {
      setError(err?.response?.data?.detail || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="page-shell">
      <section className="hero-card">
        <p className="eyebrow">YOLO + EasyOCR</p>
        <h1>Number Plate Detector</h1>
        <p className="subtitle">
          Upload a car image, detect plate bounding boxes, and read plate text instantly.
        </p>

        <div
          className={`drop-zone ${dragActive ? "active" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={(event) => onFileChange(event.target.files?.[0])}
          />
          <label htmlFor="fileInput">
            {file ? `Selected: ${file.name}` : "Drag image here or click to browse"}
          </label>
        </div>

        <button type="button" disabled={loading} onClick={handleSubmit}>
          {loading ? "Detecting..." : "Run Detection"}
        </button>

        {error && <p className="error-text">{error}</p>}
        {backendWarning && <p className="warn-text">{backendWarning}</p>}
      </section>

      <section className="result-card">
        {!previewUrl && !result && <p className="hint">Prediction output will appear here.</p>}

        {previewUrl && (
          <div className="preview-wrap">
            <img src={previewUrl} alt="Upload preview" />
          </div>
        )}

        {result && (
          <div className="prediction-wrap">
            <h2>
              Detections: <span>{result.total_detections}</span>
            </h2>
            {result.detections.length === 0 && (
              <p className="hint">
                {result.warning
                  ? result.warning
                  : "No number plate detected in this image. Try a clearer, closer plate image or lower the confidence threshold."}
              </p>
            )}

            {result.detections.map((item, index) => (
              <article className="detect-card" key={`${item.label}-${index}`}>
                <p>
                  <strong>Label:</strong> {item.label}
                </p>
                <p>
                  <strong>Confidence:</strong> {(item.confidence * 100).toFixed(2)}%
                </p>
                <p>
                  <strong>Plate Text:</strong> {item.text || "N/A"}
                </p>
                <p>
                  <strong>BBox:</strong> [{item.bbox.join(", ")}]
                </p>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

export default App;
