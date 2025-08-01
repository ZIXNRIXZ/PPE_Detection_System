import cv2
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from playsound import playsound
import threading


fire_model = YOLO("firensmoke.pt")
ppe_model = YOLO("PPEdetect.pt")

os.makedirs("snapshots", exist_ok=True)
csv_file = "detections.csv"

if not os.path.exists(csv_file):
    pd.DataFrame(columns=["Time", "Model", "Class", "Confidence", "Snapshot"]).to_csv(csv_file, index=False)

def play_alert():
    playsound("alert.mp3") 

def detect_from_source(source=0):  
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ Cannot access source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        fire_results = fire_model.predict(frame, verbose=False)
        ppe_results = ppe_model.predict(frame, verbose=False)

        detections = []

        for model_name, results in zip(["FireSmoke", "PPE"], [fire_results, ppe_results]):
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names[cls_id]

                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"snapshots/{model_name}_{label}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)

                
                    new_row = {
                        "Time": timestamp,
                        "Model": model_name,
                        "Class": label,
                        "Confidence": round(conf, 2),
                        "Snapshot": filename
                    }
                    pd.concat([pd.read_csv(csv_file), pd.DataFrame([new_row])], ignore_index=True).to_csv(csv_file, index=False)

                 
                    threading.Thread(target=play_alert).start()

        cv2.imshow("Real-time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run on webcam
detect_from_source(0)


