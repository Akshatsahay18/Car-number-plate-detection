import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "",
  timeout: 120000
});

export async function detectPlate(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post("/predict", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });

  return response.data;
}

export async function getHealth() {
  const response = await api.get("/health");
  return response.data;
}
