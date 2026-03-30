🎯 AI Proctoring Analytics System
Smart Exam Monitoring using YOLOv8 & Risk Scoring

🚀 Overview

A real-time AI-powered proctoring system that monitors candidates during online exams using YOLOv8 (object + pose detection).
It detects suspicious behavior and assigns a dynamic risk score, while logging events to a backend for analytics.

✨ Key Highlights

✔ Real-time webcam monitoring
✔ AI-based behavior analysis
✔ Automatic violation detection
✔ Dynamic risk scoring (0–100%)
✔ Backend logging for analytics

🔍 What It Detects

📱 Phone usage
👥 Multiple persons in frame
👀 Looking away from screen
🪑 Poor posture
🚫 No face detected

📊 Risk Scoring

PHONE → +60
MULTIPLE PERSONS → +100
LOOKING AWAY → +15
BAD POSTURE → +10
NO FACE → +25

➡ Score updates in real time and is capped at 100%

🧠 How It Works

Camera → YOLO Detection → Behavior Analysis → Risk Engine → API Logging

🛠️ Tech Stack

Python • OpenCV • YOLOv8 • NumPy • Flask • Requests

⚙️ Setup

Install dependencies
pip install -r requirements.txt

Run backend
python backend.py

Run system
python main.py

(Optional) Dashboard
streamlit run dashboard.py

📡 API Output Example

event_type: PHONE
confidence: 0.91
risk_score: 75

⚠️ Limitations

• Works best with good lighting
• Single camera setup
• No identity verification

🔮 Future Scope

• Face recognition
• Eye tracking
• Audio monitoring
• Cloud dashboard

📜 License

MIT License
