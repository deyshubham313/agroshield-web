# 🌿 AgroShield — Smart Crop Protection System
### Ideathon 2026 | Team Projexa 26E3134 | K.R. Mangalam University

A full-stack Python web application for real-time intelligent crop protection using
Computer Vision (OpenCV/YOLOv8), IoT sensor simulation, automated alerting,
and a live analytics dashboard.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install flask numpy opencv-python-headless Pillow
```

### 2. Run the server
```bash
cd agroshield
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## 📁 Project Structure

```
agroshield/
├── app.py                  # Flask backend — APIs, sensors, detection, DB
├── templates/
│   └── index.html          # Full frontend — all 5 pages
├── db/
│   └── agroshield.db       # SQLite database (auto-created on first run)
├── requirements.txt
└── README.md
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Live MJPEG Camera Feeds** | 6 OpenCV-rendered synthetic field cameras with real-time threat overlays |
| **IoT Sensor Simulator** | Background thread simulating Raspberry Pi / ESP32 sensor data every 2s |
| **AI Detection API** | POST `/api/detect/<cam_id>` — simulates YOLOv8 object detection |
| **Auto-Alert Engine** | Background thread randomly triggers detections & fires deterrents |
| **SQLite Database** | Persists all sensor readings, threat events, and alerts |
| **30-day Historical Data** | Seeded on first run for analytics charts |
| **REST API** | 10 clean JSON endpoints for all data |
| **5-page Dashboard** | Home, Dashboard, AI Detection, Analytics, About |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Main dashboard UI |
| GET | `/api/sensors` | Live sensor readings (all zones) |
| GET | `/api/sensors/history?zone=Zone-A&hours=24` | Historical sensor data |
| GET | `/video/<1-6>` | MJPEG camera stream |
| POST | `/api/detect/<cam_id>` | Trigger AI detection on camera |
| GET | `/api/alerts` | Live alert queue |
| GET | `/api/threats/active` | Currently active threat detections |
| GET | `/api/analytics` | Full analytics data from DB |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12 · Flask · SQLite · threading
- **Computer Vision**: OpenCV (synthetic field rendering + MJPEG streaming)
- **Frontend**: Vanilla JS · Chart.js · CSS Variables · Google Fonts
- **IoT Simulation**: Background threads mimicking MQTT/Raspberry Pi
- **Database**: SQLite with 30-day seeded historical data

---

## 👥 Team Projexa

| Name | Roll No |
|---|---|
| Bhuveeta Sarohi | 2301420011 |
| Shubham Dey | 2301420012 |
| Anjali | 2301420021 |
| Arshiya Tahim | 2301420031 |
| Jigyasa Singh | 2301420032 |

**Supervisor**: Dr. Shahid Ahmad Wani, K.R. Mangalam University
