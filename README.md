# 🛡️ PPE Detection System – Real-Time Fire & Safety Monitoring on Jetson TX2

A production-grade, real-time industrial safety monitoring system designed to detect PPE violations (helmets, vests) and environmental hazards (fire, smoke) using YOLO-Models accelerated with TensorRT on NVIDIA Jetson TX2. Featuring a FastAPI backend and a responsive dashboard UI, the system is built to enhance safety compliance in industrial environments.

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

## 🌐 Frontend Deployment to Vercel

> **Note:** The system is designed for full-stack deployment but not yet hosted live. The frontend can be deployed to Vercel for demonstration.

1. Push the `frontend` directory to GitHub
2. Deploy via Vercel dashboard:

   * Build Command: `npm install && npm run build`
   * Output Directory: `dist`
3. Set `VITE_API_URL` to your backend endpoint

---

## 📊 Performance Benchmarks

* Detection Accuracy: 91% (tested on real-world data)
* Inference Latency: <150ms per frame
* Frame Rate: \~15 FPS on Jetson TX2

---

## 📥 Download Full Project ZIP

You can download the entire project (backend, frontend, scripts, and setup guide) here:
👉 [Download fire\_ppe.zip](https://github.com/ZIXNRIXZ/PPE_Detection_System/releases/download/v1.0/fire_ppe.zip)

> This ZIP is provided for code review, testing, and deployment by collaborators or recruiters. Please refer to the README inside for full setup instructions.

---

## 🎯 Project Overview for Recruiters

![Dashboard Screenshot 1](https://github.com/user-attachments/assets/f28beeb0-7591-41ff-a4d5-0cf0e04a1032)
![Dashboard Screenshot 2](https://github.com/user-attachments/assets/85c5267f-1b82-4062-be47-52a00609d4ce)

* Demonstrates ability to build full-stack AI systems with real-time video processing
* Optimized AI model deployment on edge hardware (Jetson TX2)
* Designed complete automation pipeline (scripts + UI + backend)
* Smart dashboard UI with real-time logging and alert capabilities
* Practical industry application with measurable performance metrics

---

## 🔐 License

MIT License. For production deployments in safety-critical environments, comprehensive validation and compliance checks are recommended.

---

**Deployment Note**: Vercel hosts the frontend UI only. For AI inference, the backend must be deployed on Jetson TX2 or a suitable server (Railway, Render, etc.).
