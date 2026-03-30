import cv2
import json
import requests
from datetime import datetime
from ultralytics import YOLO

# ===================== CONFIG =====================
API_ENDPOINT = "http://localhost:5000/api/logs"  # Change to your backend
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_FRAMES = 5

RISK_LEVELS = {
    "PHONE": 60,
    "MULTIPLE_PERSONS": 100,
    "LOOKING_AWAY": 15,
    "BAD_POSTURE": 10,
    "NO_FACE": 25
}

# ===================== RISK ENGINE =====================
class RiskScoringEngine:
    def __init__(self):
        self.current_score = 0

    def update(self, violation):
        self.current_score = min(100, self.current_score + RISK_LEVELS.get(violation, 0))
        return self.current_score

# ===================== LOGGER =====================
class MonitoringLogger:
    def __init__(self, risk_engine):
        self.risk_engine = risk_engine

    def log_event(self, event_type, confidence, metadata=None):
        score = self.risk_engine.update(event_type)

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "confidence": round(confidence, 2),
            "risk_score": score,
            "metadata": metadata or {}
        }

        print("📡 EVENT:", payload)

        # Send to backend
        try:
            requests.post(API_ENDPOINT, json=payload, timeout=1)
        except:
            print("⚠️ Backend not reachable")

        return payload

# ===================== INIT =====================
obj_model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

risk_engine = RiskScoringEngine()
logger = MonitoringLogger(risk_engine)

cap = cv2.VideoCapture(0)

look_away_buffer = 0
posture_buffer = 0

# ===================== MAIN LOOP =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -------- OBJECT DETECTION --------
    results = obj_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    persons = []
    phones = []

    for box in results.boxes:
        label = obj_model.names[int(box.cls)]
        conf = float(box.conf)

        if label == "person":
            persons.append(box)
        elif label == "cell phone":
            phones.append(box)

    # MULTIPLE PERSONS
    if len(persons) > 1:
        logger.log_event("MULTIPLE_PERSONS", 1.0, {"count": len(persons)})

    # PHONE DETECTED
    if len(phones) > 0:
        logger.log_event("PHONE", float(phones[0].conf))

    # -------- POSE DETECTION --------
    pose = pose_model(frame, verbose=False)[0]

    if pose.keypoints and len(pose.keypoints.data) > 0:
        kpts = pose.keypoints.data[0].cpu().numpy()

        nose = kpts[0]
        l_eye, r_eye = kpts[1], kpts[2]
        l_sh, r_sh = kpts[5], kpts[6]

        # LOOKING AWAY
        eye_center = (l_eye[0] + r_eye[0]) / 2
        if abs(nose[0] - eye_center) > 20:
            look_away_buffer += 1
            if look_away_buffer > SMOOTHING_FRAMES:
                logger.log_event("LOOKING_AWAY", 0.8)
        else:
            look_away_buffer = 0

        # BAD POSTURE
        shoulder_y = (l_sh[1] + r_sh[1]) / 2
        if nose[1] > shoulder_y - 30:
            posture_buffer += 1
            if posture_buffer > SMOOTHING_FRAMES * 2:
                logger.log_event("BAD_POSTURE", 0.7)
        else:
            posture_buffer = 0

    else:
        logger.log_event("NO_FACE", 1.0)

    # -------- UI --------
    cv2.putText(frame, f"RISK: {risk_engine.current_score}%",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()