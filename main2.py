from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import threading
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import io
from PIL import Image
import os
import pygame
import time
import sqlite3
from contextlib import contextmanager

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import model loader
from model_loader import load_yolo_model

# Global variables
fire_model = None
ppe_model = None
camera_active = False
alert_enabled = True

# Initialize database
def init_database():
    """Initialize SQLite database for better data management"""
    try:
        with sqlite3.connect('detections.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model TEXT NOT NULL,
                    class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot TEXT,
                    bbox TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_detections INTEGER DEFAULT 0,
                    fire_detections INTEGER DEFAULT 0,
                    smoke_detections INTEGER DEFAULT 0,
                    ppe_violations INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert initial stats if empty
            cursor.execute('SELECT COUNT(*) FROM system_stats')
            if cursor.fetchone()[0] == 0:
                cursor.execute('INSERT INTO system_stats DEFAULT VALUES')
            
            conn.commit()
            print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect('detections.db')
    try:
        yield conn
    finally:
        conn.close()

def log_detection_to_db(detection_data):
    """Log detection to SQLite database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (timestamp, model, class, confidence, snapshot, bbox)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                detection_data['timestamp'],
                detection_data['model'],
                detection_data['class'],
                detection_data['confidence'],
                detection_data['snapshot'],
                str(detection_data['bbox'])
            ))
            
            # Update stats
            cursor.execute('''
                UPDATE system_stats 
                SET total_detections = total_detections + 1,
                    fire_detections = fire_detections + CASE WHEN ? = 'fire' THEN 1 ELSE 0 END,
                    smoke_detections = smoke_detections + CASE WHEN ? = 'smoke' THEN 1 ELSE 0 END,
                    ppe_violations = ppe_violations + CASE WHEN ? LIKE '%No%' THEN 1 ELSE 0 END,
                    last_updated = CURRENT_TIMESTAMP
            ''', (detection_data['class'], detection_data['class'], detection_data['class']))
            
            conn.commit()
    except Exception as e:
        print(f"❌ Database logging error: {e}")

# Load both models once at startup
def load_models():
    """Load YOLO models with error handling"""
    global fire_model, ppe_model
    
    try:
        print("🔄 Loading Fire/Smoke detection model...")
        fire_model = load_yolo_model("firensmoke.pt")
        print("✅ Fire/Smoke model loaded successfully!")

        print("🔄 Loading PPE detection model...")
        ppe_model = load_yolo_model("PPEdetect.pt")
        print("✅ PPE model loaded successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

# Initialize system
print("🚀 Initializing Fire & PPE Detection System...")
init_database()

if not load_models():
    print("❌ Failed to load models. Please check model files.")
    exit(1)

# Create folders/files
os.makedirs("snapshots", exist_ok=True)

# Initialize pygame mixer for audio alerts
try:
    pygame.mixer.init()
    print("✅ Audio system initialized")
except Exception as e:
    print(f"⚠️ Audio system initialization failed: {e}")

# Alert sound - using pygame for better compatibility
def play_alert():
    """Play alert sound using pygame - only if camera is active and alerts enabled"""
    if not camera_active or not alert_enabled:
        return
        
    try:
        if os.path.exists("alert.mp3"):
            pygame.mixer.music.load("alert.mp3")
            pygame.mixer.music.play()
            print("🔊 Alert sound played!")
        else:
            print("⚠️ Alert: alert.mp3 not found - audio alerts disabled")
    except pygame.error as e:
        print(f"🔊 Pygame audio error: {e}")
    except Exception as e:
        print(f"🔊 Sound error: {e}")

def play_alert_with_duration():
    """Play alert sound and wait for it to finish"""
    if not alert_enabled:
        return
        
    try:
        if os.path.exists("alert.mp3"):
            pygame.mixer.music.load("alert.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("🔊 Alert sound completed!")
        else:
            print("⚠️ Alert: alert.mp3 not found")
    except Exception as e:
        print(f"🔊 Sound error: {e}")

def process_image(image_array: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Process image with both YOLO models and return annotated frame with detection results"""
    if fire_model is None or ppe_model is None:
        print("❌ Models not loaded")
        return image_array, []
        
    frame = image_array.copy()
    detection_results = []
    
    try:
        # Run both model detections
        fire_results = fire_model.predict(frame, verbose=False)
        ppe_results = ppe_model.predict(frame, verbose=False)
        
        for model_name, results in zip(["FireSmoke", "PPE"], [fire_results, ppe_results]):
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls_id]
                
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw bounding box and label on frame
                    color = (0, 0, 255) if model_name == "FireSmoke" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Save snapshot
                    timestamp = datetime.now().isoformat()
                    filename = f"snapshots/{model_name}_{label}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    # Create detection data
                    detection_data = {
                        "timestamp": timestamp,
                        "model": model_name,
                        "class": label,
                        "confidence": round(conf, 2),
                        "snapshot": filename,
                        "bbox": [x1, y1, x2, y2]
                    }
                    
                    # Log to database
                    log_detection_to_db(detection_data)
                    
                    # Alert - play in background thread only if camera is active
                    if camera_active:
                        alert_thread = threading.Thread(target=play_alert)
                        alert_thread.daemon = True
                        alert_thread.start()
                    
                    # Store detection result
                    detection_results.append({
                        "model": model_name,
                        "class": label,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    print(f"🚨 Detection: {model_name} - {label} ({conf:.2f})")
    except Exception as e:
        print(f"❌ Detection processing error: {e}")
    
    return frame, detection_results

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Accept image file and return processed image with detection results"""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to numpy array - handle multiple formats
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        except Exception:
            nparr = np.frombuffer(contents, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image_array is None:
                raise ValueError("Could not decode image")
        
        # Process image
        processed_frame, detections = process_image(image_array)
        
        # Encode processed frame back to JPEG
        success, buffer = cv2.imencode('.jpg', processed_frame)
        if not success:
            raise ValueError("Could not encode processed image")
        
        # Return processed image as response
        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(len(detections)),
                "X-Detections": str(detections) if detections else "[]"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "🔥 Fire & PPE Detection API", 
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "camera_active": camera_active,
        "alert_enabled": alert_enabled,
        "endpoints": {
            "detect": "POST image to /detect endpoint",
            "live": "GET /live for webcam stream",
            "cctv": "GET /cctv for RTSP stream",
            "test_audio": "GET /test_audio to test sound",
            "stats": "GET /stats for detection statistics",
            "detections": "GET /detections for all detection records",
            "toggle_camera": "POST /toggle_camera to enable/disable camera",
            "toggle_alerts": "POST /toggle_alerts to enable/disable alerts"
        }
    }

@app.post("/toggle_camera")
def toggle_camera():
    """Toggle camera active state"""
    global camera_active
    camera_active = not camera_active
    return {
        "message": f"Camera {'activated' if camera_active else 'deactivated'}",
        "camera_active": camera_active
    }

@app.post("/toggle_alerts")
def toggle_alerts():
    """Toggle alert system"""
    global alert_enabled
    alert_enabled = not alert_enabled
    return {
        "message": f"Alerts {'enabled' if alert_enabled else 'disabled'}",
        "alert_enabled": alert_enabled
    }

@app.get("/test_audio")
def test_audio():
    """Test audio functionality"""
    try:
        if not alert_enabled:
            return {"message": "🔇 Alerts are currently disabled"}
            
        alert_thread = threading.Thread(target=play_alert_with_duration)
        alert_thread.daemon = True
        alert_thread.start()
        return {"message": "🔊 Audio test initiated - check console for results"}
    except Exception as e:
        return {"error": f"Audio test failed: {str(e)}"}

# Live webcam detection endpoint
@app.get("/live")
def live_feed():
    def generate_frames(source=0):
        global camera_active
        camera_active = True
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            camera_active = False
            raise RuntimeError("❌ Cannot access video source.")
        
        try:
            while camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = process_image(frame)
                
                # Encode frame
                success, buffer = cv2.imencode('.jpg', processed_frame)
                if not success:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        finally:
            cap.release()
            camera_active = False
    
    return StreamingResponse(generate_frames(0), media_type="multipart/x-mixed-replace; boundary=frame")

# Optional: CCTV/RTSP stream endpoint
@app.get("/cctv")
def cctv_feed(rtsp_url: str = "rtsp://username:password@ip_address:port/"):
    """Stream from CCTV/RTSP source"""
    def generate_frames(source=rtsp_url):
        global camera_active
        camera_active = True
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            camera_active = False
            raise RuntimeError(f"❌ Cannot access CCTV source: {source}")
        
        try:
            while camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, detections = process_image(frame)
                
                # Encode frame
                success, buffer = cv2.imencode('.jpg', processed_frame)
                if not success:
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        finally:
            cap.release()
            camera_active = False
    
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detections")
def get_detections():
    """Get all detection records from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY created_at DESC 
                LIMIT 100
            ''')
            
            columns = [description[0] for description in cursor.description]
            detections = [dict(zip(columns, row)) for row in cursor.fetchall()]

            # Patch each detection to match frontend expectations
            patched = []
            for d in detections:
                # Derive type
                if d["model"] == "FireSmoke":
                    if d["class"].lower() == "fire":
                        dtype = "fire"
                    elif d["class"].lower() == "smoke":
                        dtype = "smoke"
                    else:
                        dtype = "fire"
                elif d["model"] == "PPE":
                    if "no" in d["class"].lower():
                        dtype = "ppe_violation"
                    else:
                        dtype = "ppe_compliance"
                else:
                    dtype = "unknown"

                # Parse bbox safely
                bbox = d["bbox"]
                if isinstance(bbox, str):
                    try:
                        bbox = eval(bbox)
                    except Exception:
                        bbox = []

                patched.append({
                    "id": d["id"],
                    "timestamp": d["timestamp"],
                    "type": dtype,
                    "confidence": d["confidence"],
                    "location": "",  # Or use a real value if available
                    "snapshot": d["snapshot"],
                    "details": {
                        "model": d["model"],
                        "class": d["class"],
                        "bbox": bbox,
                    },
                    "created_at": d.get("created_at"),
                })

            return {
                "total_detections": len(patched),
                "detections": patched
            }
    except Exception as e:
        return {"error": f"Could not read detections: {str(e)}"}

@app.get("/stats")
def get_stats():
    """Get detection statistics from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get system stats
            cursor.execute('SELECT * FROM system_stats ORDER BY id DESC LIMIT 1')
            stats_row = cursor.fetchone()
            
            if stats_row:
                columns = [description[0] for description in cursor.description]
                stats = dict(zip(columns, stats_row))
            else:
                stats = {
                    "total_detections": 0,
                    "fire_detections": 0,
                    "smoke_detections": 0,
                    "ppe_violations": 0
                }
            
            # Get recent detections
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY created_at DESC 
                LIMIT 10
            ''')
            
            columns = [description[0] for description in cursor.description]
            recent_detections = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # Get detection counts by model and class
            cursor.execute('''
                SELECT model, class, COUNT(*) as count 
                FROM detections 
                GROUP BY model, class
            ''')
            
            by_model_class = {}
            for row in cursor.fetchall():
                model, class_name, count = row
                if model not in by_model_class:
                    by_model_class[model] = {}
                by_model_class[model][class_name] = count
            
            return {
                **stats,
                "by_model_class": by_model_class,
                "recent_detections": recent_detections,
                "camera_active": camera_active,
                "alert_enabled": alert_enabled
            }
    except Exception as e:
        return {"error": f"Could not generate stats: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Fire & PPE Detection API...")
    print("📊 Make sure you have:")
    print("   - firensmoke.pt model file")
    print("   - PPEdetect.pt model file") 
    print("   - alert.mp3 audio file (optional)")
    print("🔗 API will be available at: http://localhost:8000")
    uvicorn.run("main2:app", host="0.0.0.0", port=8000, reload=False)