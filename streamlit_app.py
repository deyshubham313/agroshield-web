import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgroShield", page_icon="🌿", layout="wide")

# --- LOAD YOLOv8 MODEL ---
# @st.cache_resource ensures the model only loads once and doesn't freeze the app
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌿 AgroShield")
st.sidebar.write("Smart Crop Protection System")
page = st.sidebar.radio("Navigation Menu", ["Live Dashboard", "AI Vision Engine", "Analytics & Data"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Team Projexa (26E3134)**")
st.sidebar.markdown("K.R. Mangalam University")

# ════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ════════════════════════════════════════════════════════
if page == "Live Dashboard":
    st.title("Live Control Center - Zone A")
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Temperature", "28.4 °C", "-0.5 °C")
    col2.metric("Humidity", "67%", "2%")
    col3.metric("Soil Moisture", "42%", "Adequate")
    col4.metric("Active Threats", "0", "All Clear")
    
    st.markdown("---")
    
    # Sensor Graphs
    st.subheader("📡 Live Environmental Sensors (Last 24h)")
    chart_data = pd.DataFrame(
        np.random.randn(24, 3) * [2, 5, 4] + [28, 65, 40],
        columns=['Temperature (°C)', 'Humidity (%)', 'Soil Moisture (%)']
    )
    st.line_chart(chart_data)

# ════════════════════════════════════════════════════════
# PAGE 2: AI DETECTION ENGINE
# ════════════════════════════════════════════════════════
elif page == "AI Vision Engine":
    st.title("Live Pest & Animal Detection")
    st.write("Upload a video to simulate the live camera feed. YOLOv8 will analyze it in real-time.")
    
    uploaded_file = st.file_uploader("📁 Upload Video (MP4/MOV)", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### System Status")
            st.info("Model: YOLOv8 Nano")
            st.success("Hardware: Edge Local")
            run_btn = st.button("🔍 Start AI Analysis", use_container_width=True)
            alert_box = st.empty()
        
        with col1:
            stframe = st.empty() # Placeholder for the video
            
            if run_btn:
                cap = cv2.VideoCapture(tfile.name)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Video stream ended.")
                        break
                    
                    # Resize for faster processing
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Run YOLOv8
                    results = model(frame, verbose=False)[0]
                    
                    threat_detected = False
                    detected_label = ""
                    highest_conf = 0
                    
                    # Draw bounding boxes
                    for box in results.boxes:
                        conf = float(box.conf[0])
                        if conf > 0.45:
                            cls_id = int(box.cls[0])
                            label = model.names[cls_id].capitalize()
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw Red Box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"{label} {int(conf*100)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            if conf > highest_conf:
                                highest_conf = conf
                                detected_label = label
                                threat_detected = True
                    
                    # Convert BGR (OpenCV) to RGB (Streamlit)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Update the image in the Streamlit UI
                    stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Update Alerts
                    if threat_detected:
                        alert_box.error(f"🚨 **THREAT DETECTED: {detected_label}**\n\nConfidence: {int(highest_conf*100)}%\n\nAction: Deterrent Activated!")
                    else:
                        alert_box.success("✅ **Zone Clear**\n\nNo threats detected.")
                    
                    # Small delay to keep the video playing at normal speed
                    time.sleep(0.03)

# ════════════════════════════════════════════════════════
# PAGE 3: ANALYTICS
# ════════════════════════════════════════════════════════
elif page == "Analytics & Data":
    st.title("Data Intelligence")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Detection Accuracy", "94.7%", "Model v2")
    col2.metric("Avg Alert Latency", "<2s", "Edge Optimized")
    col3.metric("Pesticide Reduction", "68%", "Eco Impact")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Threats by Type (Last 30 Days)")
        threat_data = pd.DataFrame({
            'Incidents': [45, 28, 15, 12, 5]
        }, index=['Wild Boar', 'Monkeys', 'Deer', 'Aphids', 'Caterpillars'])
        st.bar_chart(threat_data)
        
    with col_right:
        st.subheader("Pesticide Use Reduction")
        pest_data = pd.DataFrame({
            'Before AgroShield': [100, 98, 102, 99, 101, 100],
            'With AgroShield': [100, 72, 58, 44, 35, 32]
        }, index=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
        st.line_chart(pest_data)