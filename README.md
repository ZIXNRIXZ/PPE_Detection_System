# 🛡️ PPE Detection System – Real-Time Fire & Safety Monitoring on Jetson TX2

A production-grade, real-time industrial safety monitoring system designed to detect PPE violations (helmets, vests) and environmental hazards (fire, smoke) using YOLO-NAS accelerated with TensorRT on NVIDIA Jetson TX2. Featuring a FastAPI backend and a responsive dashboard UI for making the system user friendly.

---

## ✨ Key Features

* Real-time detection of PPE compliance and fire/smoke hazards
* Low-latency inference using TensorRT on Jetson TX2 (<150ms per frame)
* CSV-based logging and analytics for incident tracking
* Dynamic dashboard for real-time alerts, statistics, and video feed
* One-click startup with `start_system.bat` or Linux bash script

---

## 🧠 Datasets Used

* **PPE Detection Dataset**: Custom-labeled dataset including helmets, vests in industrial environments
* **Fire & Smoke Detection Dataset**: Curated public datasets and industrial footage for training YOLO models
* **Data Volume**: 10,000+ images used across both detection models

---

## 🛠️ Technology Stack

* **AI Inference**: YOLO-Models → ONNX → TensorRT (FP16 acceleration on Jetson TX2)
* **Backend**: FastAPI, Python 3.10, OpenCV, Pygame (audio alerts), CSV logging
* **Frontend**: React, TypeScript, Vite, Tailwind CSS, Browser-based audio alerts
* **Automation**: Batch script (`start_system.bat`), Linux shell script, systemd service (Jetson)
* **Deployment**: Jetson TX2 (edge inference), Vercel (frontend only), Local/Remote API support

---

## 🚀 Local Deployment (Windows/Linux)

### 1. Clone and Set Up Environment

```bash
git clone <your-repo-url>
cd ppe-detection
python3 -m venv ppe-env
source ppe-env/bin/activate
pip install -r requirements.txt
```

### 2. Launch Backend

```bash
cd backend
uvicorn main.py --host 0.0.0.0 --port 8000
```

### 3. Launch Frontend

```bash
cd frontend/safety-eye-vision-guard
npm install
npm run dev
```

---

## 🚧 Jetson TX2 Configuration

1. Export YOLO-NAS model to ONNX:

```bash
yolo export model=yolonas.pt format=onnx
```

2. Convert ONNX to TensorRT engine:

```bash
trtexec --onnx=yolonas.onnx --saveEngine=yolonas.engine --fp16
```

3. Load TensorRT engine in `main.py` for high-speed inference

---


---

## 📊 Performance Benchmarks

* Detection Accuracy: 91% (tested on real-world data)
* Inference Latency: <150ms per frame
* Frame Rate: \~15 FPS on Jetson TX2

---
# What it looks like
<img width="1916" height="1079" alt="Screenshot 2025-07-13 183905" src="https://github.com/user-attachments/assets/bc6cfb1b-987f-42f3-95e0-b4af3c50615e" />
<img width="1919" height="1079" alt="Screenshot 2025-07-13 183954" src="https://github.com/user-attachments/assets/d8de3be4-36f9-4d03-815e-42462df7c617" />


---

## 🔐 License

MIT License. For production deployments in safety-critical environments, comprehensive validation and compliance checks are recommended.

---

**Deployment Note**: Vercel hosts the frontend UI only. For AI inference, the backend must be deployed on Jetson TX2 or a suitable server (Railway, Render, etc.).
