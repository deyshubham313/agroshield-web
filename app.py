"""
AgroShield — Smart Crop Protection System  v3.0
Image Analysis Edition — Single Image + Dataset Mode
Team Projexa 26E3134 | K.R. Mangalam University
Team Members:
  Shubham Dey       — 2301420010
  Bhuveeta          — 2301420012
  Anjali            — 2301420021
  Jigyasa           — 2301420032
  Arshiya           — 2301420031
"""
import json, math, random, sqlite3, threading, time, datetime, os, csv, io, base64
import numpy as np
import cv2
from flask import Flask, jsonify, render_template, Response, request, send_from_directory, send_file
from werkzeug.utils import secure_filename

# ── PHONE ALERT CONFIG ────────────────────────────────────────────────────────
# Set your phone number here (with country code e.g. +919876543210)
ALERT_PHONE_NUMBER = "+919999999999"   # <-- change this to real number

# ── GLOBAL AI STATE ──────────────────────────────────────────────────────────
yolo_model       = None
yolo_loading     = False
yolo_error       = None

app = Flask(__name__)

UPLOAD_FOLDER  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SIM_FOLDER     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulated_dataset")
os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SIM_FOLDER,     exist_ok=True)

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["DATASET_FOLDER"]     = DATASET_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB cap

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "agroshield.db")

ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# ── CAMERA IMAGE ASSIGNMENT STATE ────────────────────────────────────────────
camera_images      = {}
camera_assign_lock = threading.Lock()
CAMERA_COUNT       = 6

# ── DATASET PROCESSING STATE ─────────────────────────────────────────────────
dataset_state = {
    "running": False, "total": 0, "processed": 0,
    "current_file": "", "results": [], "error": None,
    "done": False, "start_time": None,
}
dataset_lock = threading.Lock()

# ── PHONE ALERT STATE ─────────────────────────────────────────────────────────
phone_alert_log = []   # recent phone alerts shown in UI
phone_lock      = threading.Lock()

# ── DATABASE ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, zone TEXT NOT NULL,
            temperature REAL, humidity REAL,
            soil_moisture REAL, motion_level REAL);
        CREATE TABLE IF NOT EXISTS threat_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, zone TEXT NOT NULL,
            camera_id INTEGER, threat_type TEXT NOT NULL,
            confidence REAL, severity TEXT,
            action_taken TEXT, resolved INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, level TEXT NOT NULL,
            title TEXT NOT NULL, message TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS image_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            mode TEXT NOT NULL,
            detections_json TEXT,
            detection_count INTEGER,
            top_threat TEXT,
            top_confidence REAL,
            inference_ms REAL,
            camera_id INTEGER DEFAULT NULL);
        CREATE TABLE IF NOT EXISTS phone_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            phone TEXT NOT NULL,
            threat TEXT NOT NULL,
            zone TEXT,
            confidence REAL,
            status TEXT);
    """)
    conn.commit(); conn.close()
    _seed_data()

def _seed_data():
    conn = get_db()
    if conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0] > 0:
        conn.close(); return
    zones   = ["Zone-A", "Zone-B", "Zone-C", "Zone-D"]
    threats = [("Wild Boar","CRITICAL"),("Deer","CRITICAL"),("Monkeys","CRITICAL"),
               ("Aphid Cluster","WARNING"),("Caterpillar","WARNING"),
               ("Leaf Blight","WARNING"),("Stem Borer","WARNING"),("Rabbits","WARNING")]
    now = datetime.datetime.now(); sr, te = [], []
    for day in range(30):
        for hour in range(0, 24, 2):
            ts = (now - datetime.timedelta(days=29-day, hours=hour)).isoformat()
            for z in zones:
                sr.append((ts,z,round(24+random.gauss(4,2),1),round(60+random.gauss(8,5),1),
                           round(40+random.gauss(10,8),1),round(random.uniform(10,90),1)))
            if random.random() < 0.15:
                th,sv = random.choice(threats)
                te.append((ts,random.choice(zones),random.randint(1,6),th,
                           round(random.uniform(0.72,0.97),2),sv,"Deterrent+SMS",1))
    conn.executemany("INSERT INTO sensor_readings VALUES(NULL,?,?,?,?,?,?)", sr)
    conn.executemany("INSERT INTO threat_events VALUES(NULL,?,?,?,?,?,?,?,?)", te)
    conn.commit(); conn.close()

# ── SENSOR SIMULATION ────────────────────────────────────────────────────────
class SensorSim:
    def __init__(self):
        self.data = {
            "Zone-A": {"temp":28.4,"hum":67.2,"soil":42.1,"motion":78},
            "Zone-B": {"temp":27.1,"hum":71.5,"soil":55.3,"motion":12},
            "Zone-C": {"temp":29.8,"hum":64.0,"soil":38.6,"motion":45},
            "Zone-D": {"temp":26.5,"hum":73.2,"soil":61.4,"motion": 8},
        }
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            time.sleep(2)
            with self._lock:
                for d in self.data.values():
                    d["temp"]   = round(d["temp"]  + random.gauss(0,.15), 1)
                    d["hum"]    = round(max(30,min(95,d["hum"]  +random.gauss(0,.4))),1)
                    d["soil"]   = round(max(10,min(90,d["soil"] +random.gauss(0,.3))),1)
                    d["motion"] = round(max(0, min(100,d["motion"]+random.gauss(0,3))),1)
            try:
                conn = get_db(); ts = datetime.datetime.now().isoformat()
                with self._lock:
                    rows = [(ts,z,d["temp"],d["hum"],d["soil"],d["motion"]) for z,d in self.data.items()]
                conn.executemany("INSERT INTO sensor_readings VALUES(NULL,?,?,?,?,?,?)", rows)
                conn.commit(); conn.close()
            except: pass

    def get_all(self):
        with self._lock: return {k:dict(v) for k,v in self.data.items()}

sensors = SensorSim()

# ── THREAT DEFS & CAMERAS ────────────────────────────────────────────────────
THREAT_DEFS = [
    {"label":"Wild Boar",    "severity":"CRITICAL","color":(60,60,240)},
    {"label":"Deer",         "severity":"CRITICAL","color":(60,60,240)},
    {"label":"Monkeys",      "severity":"CRITICAL","color":(60,60,240)},
    {"label":"Aphid Cluster","severity":"WARNING", "color":(36,180,251)},
    {"label":"Caterpillar",  "severity":"WARNING", "color":(36,180,251)},
    {"label":"Leaf Blight",  "severity":"WARNING", "color":(60,130,251)},
    {"label":"Stem Borer",   "severity":"WARNING", "color":(60,130,251)},
    {"label":"Rabbits",      "severity":"WARNING", "color":(36,180,251)},
]
CAMERAS = {1:"Zone-A North",2:"Zone-A South",3:"Zone-B East",
           4:"Zone-B West", 5:"Zone-C Main", 6:"Zone-D Gate"}

_active = {}
_tlock  = threading.Lock()
_alerts = []
_alock  = threading.Lock()

# ── PHONE ALERT HELPER ────────────────────────────────────────────────────────
def send_phone_alert(threat_label, zone, confidence, camera_id=None):
    """
    Simulate / dispatch a phone call alert.
    Replace the body of this function with a real Twilio / Exotel / MSG91
    call trigger when you have credentials.
    """
    ts  = datetime.datetime.now().isoformat()
    msg = (f"AGROSHIELD ALERT: {threat_label} detected at {zone} "
           f"with {int(confidence*100)}% confidence. "
           f"Camera {camera_id or 'N/A'}. "
           f"Please take immediate action.")

    log_entry = {
        "id":        int(time.time()*1000 + random.randint(0,999)),
        "timestamp": ts,
        "phone":     ALERT_PHONE_NUMBER,
        "threat":    threat_label,
        "zone":      zone,
        "confidence": round(confidence,3),
        "message":   msg,
        "status":    "SIMULATED_CALL",   # change to "SENT" after real integration
    }

    # ── Twilio example (uncomment & fill credentials to activate) ─────────────
    # from twilio.rest import Client
    # client = Client("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")
    # call = client.calls.create(
    #     twiml=f'<Response><Say voice="alice">{msg}</Say></Response>',
    #     to=ALERT_PHONE_NUMBER,
    #     from_="+1XXXXXXXXXX"   # your Twilio number
    # )
    # log_entry["status"] = f"CALL_SID:{call.sid}"
    # ─────────────────────────────────────────────────────────────────────────

    with phone_lock:
        phone_alert_log.insert(0, log_entry)
        del phone_alert_log[30:]

    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO phone_alerts(timestamp,phone,threat,zone,confidence,status) VALUES(?,?,?,?,?,?)",
            (ts, ALERT_PHONE_NUMBER, threat_label, zone, round(confidence,3), log_entry["status"])
        )
        conn.commit(); conn.close()
    except: pass

    print(f"[PHONE ALERT] {log_entry['status']} → {ALERT_PHONE_NUMBER} | {threat_label} @ {zone}")
    return log_entry

# ── ALERT HELPER ─────────────────────────────────────────────────────────────
def push_alert(level, title, message, send_phone=False, threat_label=None, zone=None, confidence=0, camera_id=None):
    ts  = datetime.datetime.now().isoformat()
    obj = {"id":int(time.time()*1000+random.randint(0,999)),
           "timestamp":ts,"level":level,"title":title,"message":message}
    with _alock:
        _alerts.insert(0, obj); del _alerts[50:]
    try:
        conn = get_db()
        conn.execute("INSERT INTO alerts(timestamp,level,title,message) VALUES(?,?,?,?)",
                     (ts,level,title,message))
        conn.commit(); conn.close()
    except: pass

    if send_phone and threat_label and level in ("CRITICAL","WARNING"):
        send_phone_alert(threat_label, zone or "Unknown Zone", confidence, camera_id)

# ── YOLO LOADER ──────────────────────────────────────────────────────────────
def load_yolo():
    global yolo_model, yolo_loading, yolo_error
    try:
        from ultralytics import YOLO
        print("\n[AI ENGINE] Loading YOLOv8 model …")
        model = YOLO("yolov8n.pt")
        model(np.zeros((640,480,3), np.uint8), verbose=False)
        yolo_model = model
        print("[AI ENGINE] YOLOv8 ready!\n")
    except Exception as e:
        yolo_error = str(e)
        print(f"[AI ENGINE] Error: {e}\n")

def ensure_yolo():
    global yolo_loading
    if yolo_model is None and not yolo_loading and yolo_error is None:
        yolo_loading = True
        threading.Thread(target=load_yolo, daemon=True).start()

# ── DRAWING HELPERS ───────────────────────────────────────────────────────────
def draw_fancy_box(frame, x1, y1, x2, y2, label, conf, color):
    overlay = frame.copy()
    cv2.rectangle(overlay,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(overlay,.08,frame,1-.08,0,frame)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    L = 14
    for (cx,cy,sx,sy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(cx,cy),(cx+sx*L,cy),color,3)
        cv2.line(frame,(cx,cy),(cx,cy+sy*L),color,3)
    text  = f"{label}  {int(conf*100)}%"
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, .45, 1)
    px,py = x1, y1-6
    cv2.rectangle(frame,(px-2,py-th-6),(px+tw+8,py+2),color,-1)
    cv2.putText(frame,text,(px+3,py-1),cv2.FONT_HERSHEY_DUPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    bar_y = y2+6
    if bar_y+6 < frame.shape[0]:
        cv2.rectangle(frame,(x1,bar_y),(x2,bar_y+4),(40,40,40),-1)
        cv2.rectangle(frame,(x1,bar_y),(x1+int((x2-x1)*conf),bar_y+4),color,-1)

def draw_hud(frame, n_det, fps, ms, filename="", model_name="YOLOv8n", camera_id=None):
    h,w = frame.shape[:2]
    lines = [f"MODEL  {model_name}", f"FPS    {fps:.0f}", f"INF    {ms:.0f}ms", f"OBJ    {n_det}"]
    if camera_id:
        lines.append(f"CAM    {camera_id}")
    px,py = w-168, 12
    panel = frame[max(0,py-12):py+len(lines)*16+4, max(0,px-6):w-6].copy()
    black = np.zeros_like(panel); cv2.addWeighted(black,.5,panel,1,0,panel)
    frame[max(0,py-12):py+len(lines)*16+4, max(0,px-6):w-6] = panel
    for i,l in enumerate(lines):
        cv2.putText(frame,l,(px,py+i*16),cv2.FONT_HERSHEY_SIMPLEX,.38,(80,220,80),1,cv2.LINE_AA)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.rectangle(frame,(0,h-18),(w,h),(0,0,0),-1)
    label = filename[:40] if filename else "AgroShield"
    cv2.putText(frame,f"● ANALYSIS  {label}  {ts}",(6,h-5),cv2.FONT_HERSHEY_SIMPLEX,.36,(60,220,80),1,cv2.LINE_AA)

# ── SIMULATED CAMERA FRAME GENERATOR ─────────────────────────────────────────
def generate_simulated_camera_frame(cam_id, threat_info=None):
    """
    Generate a synthetic camera frame for simulation.
    If threat_info is provided, draws detection boxes on the frame.
    Returns base64-encoded JPEG string.
    """
    W, H = 640, 480
    # Background: dark green gradient to simulate field
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    # Sky gradient (top portion)
    for y in range(H // 3):
        shade = int(40 + (y / (H // 3)) * 30)
        frame[y, :] = [shade + 20, shade + 10, shade]

    # Field gradient (bottom portion)
    for y in range(H // 3, H):
        frac = (y - H // 3) / (H * 2 // 3)
        r = int(20 + frac * 10)
        g = int(60 + frac * 40)
        b = int(10 + frac * 5)
        frame[y, :] = [b, g, r]

    # Add some noise to simulate real camera texture
    noise = np.random.randint(0, 15, (H, W, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Draw some crop rows
    for x in range(0, W, 40):
        shade = random.randint(30, 70)
        cv2.line(frame, (x, H//3), (x + random.randint(-20, 20), H),
                 (10, shade, 10), random.randint(1, 3))

    # Draw simulated threat box if present
    if threat_info:
        label = threat_info["label"]
        conf  = threat_info["conf"]
        bx, by, bw, bh = threat_info.get("box", (200, 180, 140, 120))
        color = threat_info.get("color", (60, 60, 240))

        # Draw a blob to represent the animal/pest
        cx, cy = bx + bw // 2, by + bh // 2
        axes = (bw // 3, bh // 3)
        body_color = (random.randint(80,140), random.randint(60,100), random.randint(40,80))
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, body_color, -1)
        # Add some texture
        for _ in range(8):
            ox, oy = random.randint(-bw//4, bw//4), random.randint(-bh//4, bh//4)
            cv2.circle(frame, (cx+ox, cy+oy), random.randint(5,15),
                       tuple(max(0,c-20) for c in body_color), -1)

        draw_fancy_box(frame, bx, by, bx+bw, by+bh, label, conf, color)

    # HUD overlay
    zone_name = CAMERAS.get(cam_id, f"Camera {cam_id}")
    ts_str    = datetime.datetime.now().strftime("%H:%M:%S")
    n_det     = 1 if threat_info else 0

    # Status bar
    cv2.rectangle(frame, (0, H-18), (W, H), (0, 0, 0), -1)
    status = f"● LIVE  {zone_name}  {ts_str}"
    cv2.putText(frame, status, (6, H-5), cv2.FONT_HERSHEY_SIMPLEX, .36, (60, 220, 80), 1, cv2.LINE_AA)

    # Top-right mini HUD
    lines_hud = [f"CAM    {cam_id}", f"ZONE   {zone_name[:10]}", f"OBJ    {n_det}", f"TIME   {ts_str}"]
    px, py = W - 168, 12
    panel = frame[max(0,py-12):py+len(lines_hud)*16+4, max(0,px-6):W-6].copy()
    black = np.zeros_like(panel)
    cv2.addWeighted(black, .5, panel, 1, 0, panel)
    frame[max(0,py-12):py+len(lines_hud)*16+4, max(0,px-6):W-6] = panel
    for i, l in enumerate(lines_hud):
        cv2.putText(frame, l, (px, py+i*16), cv2.FONT_HERSHEY_SIMPLEX, .38, (80, 220, 80), 1, cv2.LINE_AA)

    # RED ALERT banner if threat
    if threat_info:
        cv2.rectangle(frame, (0, 0), (W, 22), (0, 0, 180), -1)
        cv2.putText(frame, f"⚠ THREAT DETECTED: {threat_info['label']}  {int(threat_info['conf']*100)}%",
                    (6, 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)

    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(enc.tobytes()).decode()

# ── SIMULATED DATASET GENERATOR ───────────────────────────────────────────────
def generate_simulated_dataset(n_images=20):
    """
    Generate n_images synthetic annotated images representing crop threats.
    Saves JPEGs to SIM_FOLDER and returns list of file paths.
    """
    paths = []
    threat_scenarios = [
        ("Wild_Boar",    (60, 60, 240),  "CRITICAL"),
        ("Deer",         (60, 60, 240),  "CRITICAL"),
        ("Monkeys",      (60, 60, 240),  "CRITICAL"),
        ("Aphid_Cluster",(36,180,251),   "WARNING"),
        ("Caterpillar",  (36,180,251),   "WARNING"),
        ("Leaf_Blight",  (60,130,251),   "WARNING"),
        ("Stem_Borer",   (60,130,251),   "WARNING"),
        ("Rabbits",      (36,180,251),   "WARNING"),
        ("Clear_Field",  (60,200,60),    "OK"),
    ]

    for i in range(n_images):
        W, H = 640, 480
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Sky
        for y in range(H // 3):
            shade = int(35 + (y/(H//3))*25)
            frame[y, :] = [shade+25, shade+12, shade]
        # Field
        for y in range(H//3, H):
            frac = (y - H//3) / (H*2//3)
            frame[y, :] = [int(12+frac*8), int(55+frac*45), int(12+frac*8)]

        noise = np.random.randint(0, 20, (H,W,3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        for x in range(0, W, 35):
            g = random.randint(35, 75)
            cv2.line(frame,(x,H//3),(x+random.randint(-15,15),H),(10,g,10),random.randint(1,2))

        # Pick a scenario
        threat_name, color, severity = random.choice(threat_scenarios)
        conf = round(random.uniform(0.65, 0.97), 2)
        filename = f"sim_{i+1:03d}_{threat_name}_{int(conf*100)}pct.jpg"
        fpath    = os.path.join(SIM_FOLDER, filename)

        if severity != "OK":
            n_boxes = random.randint(1, 3)
            for _ in range(n_boxes):
                bw  = random.randint(70,  160)
                bh  = random.randint(55,  130)
                bx  = random.randint(30,  W - bw - 30)
                by  = random.randint(H//3, H - bh - 30)
                cx, cy = bx+bw//2, by+bh//2
                body_c = (random.randint(70,130), random.randint(55,95), random.randint(35,75))
                cv2.ellipse(frame,(cx,cy),(bw//3,bh//3),0,0,360,body_c,-1)
                for _ in range(6):
                    ox,oy = random.randint(-bw//4,bw//4),random.randint(-bh//4,bh//4)
                    cv2.circle(frame,(cx+ox,cy+oy),random.randint(4,12),
                               tuple(max(0,c-20) for c in body_c),-1)
                draw_fancy_box(frame,bx,by,bx+bw,by+bh,
                               threat_name.replace("_"," "),conf,color)

        # Timestamp & label bar
        cv2.rectangle(frame,(0,H-18),(W,H),(0,0,0),-1)
        ts_str = (datetime.datetime.now()-datetime.timedelta(minutes=random.randint(0,1440))).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame,f"● SIM  {threat_name.replace('_',' ')}  {ts_str}",
                    (6,H-5),cv2.FONT_HERSHEY_SIMPLEX,.36,(60,220,80),1,cv2.LINE_AA)

        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(fpath)

    print(f"[SIM DATASET] Generated {len(paths)} images in {SIM_FOLDER}")
    return paths

# ── CORE IMAGE ANALYSIS FUNCTION ─────────────────────────────────────────────
def analyse_image_file(filepath, filename="", camera_id=None):
    global yolo_model
    ensure_yolo()

    waited = 0
    while yolo_model is None and yolo_error is None and waited < 60:
        time.sleep(0.5); waited += 0.5

    if yolo_error:
        return {"error": yolo_error}
    if yolo_model is None:
        return {"error": "Model not loaded yet. Please wait and retry."}

    raw = cv2.imread(filepath)
    if raw is None:
        return {"error": f"Cannot read image: {filename}"}

    h, w = raw.shape[:2]
    scale = min(640/w, 640/h, 1.0)
    frame = cv2.resize(raw, (int(w*scale), int(h*scale))) if scale < 1.0 else raw.copy()

    t0 = time.perf_counter()
    results = yolo_model(frame, verbose=False)[0]
    inf_ms  = (time.perf_counter()-t0)*1000
    fps_val = 1000/max(inf_ms, 1)

    detections = []
    best_label, best_conf = "", 0.0

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < 0.35: continue
        cls_id = int(box.cls[0])
        label  = yolo_model.names[cls_id].replace("-"," ").title()
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        col = (60,60,240) if conf >= 0.75 else ((36,180,251) if conf >= 0.55 else (60,200,60))
        draw_fancy_box(frame, x1,y1,x2,y2, label, conf, col)
        detections.append({"label":label,"conf":round(conf,3),"box":[x1,y1,x2-x1,y2-y1]})
        if conf > best_conf:
            best_conf, best_label = conf, label

    draw_hud(frame, len(detections), fps_val, inf_ms, filename, camera_id=camera_id)

    _, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(enc.tobytes()).decode()

    ts = datetime.datetime.now().isoformat()

    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO image_analysis(timestamp,filename,mode,detections_json,detection_count,top_threat,top_confidence,inference_ms,camera_id) VALUES(?,?,?,?,?,?,?,?,?)",
            (ts, filename, "single", json.dumps(detections), len(detections), best_label or "None", best_conf, round(inf_ms,1), camera_id)
        )
        if best_label:
            conn.execute("INSERT INTO threat_events VALUES(NULL,?,?,?,?,?,?,?,0)",
                (ts,f"Zone-{camera_id}" if camera_id else "Zone-A (Image)",camera_id or 1,best_label,best_conf,
                 "CRITICAL" if best_conf>=0.75 else "WARNING","Image Analysis"))
        conn.commit(); conn.close()
    except: pass

    if best_label:
        zone = CAMERAS.get(camera_id, "Zone-A")
        push_alert("CRITICAL" if best_conf>=0.75 else "WARNING",
                   f"🔍 {best_label} in Image",
                   f"{filename} · {int(best_conf*100)}% confidence · YOLOv8",
                   send_phone=True,
                   threat_label=best_label,
                   zone=zone,
                   confidence=best_conf,
                   camera_id=camera_id)

    return {
        "annotated_jpg_b64": img_b64,
        "detections": detections,
        "fps": round(fps_val,1),
        "inf_ms": round(inf_ms,1),
        "count": len(detections),
        "timestamp": ts,
        "filename": filename,
        "top_threat": best_label or "None",
        "top_confidence": round(best_conf,3),
        "camera_id": camera_id,
    }

# ── CAMERA ASSIGNMENT HELPERS ────────────────────────────────────────────────
def get_next_available_camera():
    with camera_assign_lock:
        for cam_id in range(1, CAMERA_COUNT + 1):
            if cam_id not in camera_images:
                return cam_id
    return None

def assign_image_to_camera(img_b64, filename, detections, count, top_threat, top_confidence):
    cam_id = get_next_available_camera()
    if cam_id is None:
        return None
    with camera_assign_lock:
        camera_images[cam_id] = {
            "image_b64": img_b64, "filename": filename,
            "timestamp": datetime.datetime.now().isoformat(),
            "detections": detections, "count": count,
            "top_threat": top_threat, "top_confidence": top_confidence,
        }
    return cam_id

def clear_camera_image(cam_id):
    with camera_assign_lock:
        camera_images.pop(cam_id, None)

def get_all_camera_states():
    with camera_assign_lock:
        return dict(camera_images)

# ── DATASET PROCESSING WORKER ─────────────────────────────────────────────────
def run_dataset_worker(image_paths):
    global dataset_state
    with dataset_lock:
        dataset_state.update({
            "running": True, "total": len(image_paths),
            "processed": 0, "current_file": "",
            "results": [], "error": None, "done": False,
            "start_time": datetime.datetime.now().isoformat()
        })

    for path in image_paths:
        fname = os.path.basename(path)
        with dataset_lock:
            dataset_state["current_file"] = fname

        cam_id = get_next_available_camera()
        result = analyse_image_file(path, fname, camera_id=cam_id)
        result["source_path"] = path

        if cam_id and "annotated_jpg_b64" in result:
            assign_image_to_camera(
                result["annotated_jpg_b64"], fname,
                result.get("detections", []), result.get("count", 0),
                result.get("top_threat", "None"), result.get("top_confidence", 0)
            )
            result["assigned_camera"] = cam_id

        with dataset_lock:
            dataset_state["results"].append(result)
            dataset_state["processed"] += 1

        time.sleep(0.05)

    with dataset_lock:
        dataset_state["running"] = False
        dataset_state["done"]    = True
        dataset_state["current_file"] = ""

    _generate_result_files(dataset_state["results"])
    push_alert("INFO","✅ Dataset Analysis Complete",
               f"{len(image_paths)} images processed · Results ready to download")

def _generate_result_files(results):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(RESULTS_FOLDER, f"agroshield_results_{ts}.json")
    clean = []
    for r in results:
        clean.append({
            "filename": r.get("filename",""), "timestamp": r.get("timestamp",""),
            "detection_count": r.get("count",0), "top_threat": r.get("top_threat","None"),
            "top_confidence": r.get("top_confidence",0), "inference_ms": r.get("inf_ms",0),
            "camera_id": r.get("assigned_camera"),
            "detections": r.get("detections",[]), "error": r.get("error",""),
        })
    with open(json_path,"w") as f:
        json.dump({"generated_at":datetime.datetime.now().isoformat(),
                   "total_images":len(results),"results":clean},f,indent=2)

    csv_path = os.path.join(RESULTS_FOLDER, f"agroshield_results_{ts}.csv")
    with open(csv_path,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","timestamp","detection_count","top_threat",
                    "top_confidence_pct","inference_ms","camera_id","all_detections","error"])
        for r in clean:
            det_str = "; ".join([f"{d['label']}({int(d['conf']*100)}%)" for d in r["detections"]])
            w.writerow([r["filename"],r["timestamp"],r["detection_count"],
                        r["top_threat"],round(r["top_confidence"]*100,1),
                        r["inference_ms"],r.get("camera_id",""),det_str,r["error"]])

    with dataset_lock:
        dataset_state["json_file"] = os.path.basename(json_path)
        dataset_state["csv_file"]  = os.path.basename(csv_path)

# ── SIMULATED LIVE CAMERA FRAMES ──────────────────────────────────────────────
_sim_frames = {}   # cam_id -> latest b64 frame
_sim_lock   = threading.Lock()

def sim_camera_loop():
    """Continuously regenerate simulated frames for all cameras."""
    while True:
        time.sleep(1.5)
        with _tlock:
            active_copy = dict(_active)
        with camera_assign_lock:
            uploaded_copy = set(camera_images.keys())

        for cam_id in range(1, CAMERA_COUNT + 1):
            if cam_id in uploaded_copy:
                continue   # uploaded image takes priority; don't overwrite
            threat_info = active_copy.get(cam_id)
            try:
                frame_b64 = generate_simulated_camera_frame(cam_id, threat_info)
                with _sim_lock:
                    _sim_frames[cam_id] = frame_b64
            except Exception as e:
                print(f"[SIM CAM {cam_id}] frame error: {e}")

# ── SYNTHETIC DETECTION LOOP ──────────────────────────────────────────────────
def detection_loop():
    while True:
        time.sleep(random.uniform(10,20))
        if random.random() < .45:
            cid = random.randint(1, 6)
            with camera_assign_lock:
                if cid in camera_images:
                    continue
            th  = random.choice(THREAT_DEFS)
            cf  = round(random.uniform(.74,.97),2)
            bw,bh = random.randint(80,180), random.randint(60,140)
            bx,by = random.randint(40,460-bw), random.randint(160,320-bh)
            with _tlock:
                _active[cid] = {"label":th["label"],"conf":cf,
                                 "box":(bx,by,bw,bh),"color":th["color"],"severity":th["severity"]}
            zone = CAMERAS.get(cid,"Unknown")
            push_alert(th["severity"],
                       f"⚠️ {th['label']} Detected!",
                       f"{zone} · Cam {cid} · {int(cf*100)}% · Deterrent activated",
                       send_phone=True,
                       threat_label=th["label"],
                       zone=zone,
                       confidence=cf,
                       camera_id=cid)
            try:
                conn=get_db()
                conn.execute("INSERT INTO threat_events VALUES(NULL,?,?,?,?,?,?,?,0)",
                    (datetime.datetime.now().isoformat(),zone,cid,th["label"],cf,
                     th["severity"],"Deterrent+SMS"))
                conn.commit(); conn.close()
            except: pass
            def _clear(c=cid):
                time.sleep(random.uniform(8,16))
                with _tlock: _active.pop(c,None)
                push_alert("INFO","Zone Cleared",f"Camera {c} — threat no longer detected")
            threading.Thread(target=_clear,daemon=True).start()

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

# ── SIMULATED CAMERA FRAMES API ───────────────────────────────────────────────
@app.route("/api/cameras/sim_frames")
def api_sim_frames():
    """Return current simulated frames for all cameras."""
    with _sim_lock:
        frames = dict(_sim_frames)
    # Overlay uploaded images
    with camera_assign_lock:
        for cid, data in camera_images.items():
            frames[cid] = data["image_b64"]
    return jsonify({str(k): v for k, v in frames.items()})

@app.route("/api/cameras/sim_frame/<int:cam_id>")
def api_sim_frame_single(cam_id):
    """Return current simulated frame for one camera."""
    with camera_assign_lock:
        if cam_id in camera_images:
            return jsonify({"frame": camera_images[cam_id]["image_b64"], "type": "uploaded"})
    with _sim_lock:
        frame = _sim_frames.get(cam_id, "")
    return jsonify({"frame": frame, "type": "simulated"})

# ── CAMERA STATE API ─────────────────────────────────────────────────────────
@app.route("/api/cameras/state")
def api_camera_state():
    cam_states = {}
    uploaded = get_all_camera_states()
    with _tlock:
        active_copy = dict(_active)
    with _sim_lock:
        sim_copy = dict(_sim_frames)

    for cam_id in range(1, CAMERA_COUNT + 1):
        if cam_id in uploaded:
            cam_states[cam_id] = {
                "type": "uploaded",
                "filename":       uploaded[cam_id]["filename"],
                "image_b64":      uploaded[cam_id]["image_b64"],
                "timestamp":      uploaded[cam_id]["timestamp"],
                "detections":     uploaded[cam_id]["detections"],
                "count":          uploaded[cam_id]["count"],
                "top_threat":     uploaded[cam_id]["top_threat"],
                "top_confidence": uploaded[cam_id]["top_confidence"],
            }
        elif cam_id in active_copy:
            cam_states[cam_id] = {
                "type":       "threat",
                "label":      active_copy[cam_id]["label"],
                "confidence": active_copy[cam_id]["conf"],
                "severity":   active_copy[cam_id]["severity"],
                "sim_frame":  sim_copy.get(cam_id, ""),
            }
        else:
            cam_states[cam_id] = {
                "type":      "live",
                "sim_frame": sim_copy.get(cam_id, ""),
            }

    return jsonify(cam_states)

@app.route("/api/cameras/clear/<int:cam_id>", methods=["POST"])
def api_clear_camera(cam_id):
    if cam_id < 1 or cam_id > CAMERA_COUNT:
        return jsonify({"success": False, "error": "Invalid camera ID"}), 400
    clear_camera_image(cam_id)
    return jsonify({"success": True, "message": f"Camera {cam_id} cleared"})

@app.route("/api/cameras/clear_all", methods=["POST"])
def api_clear_all_cameras():
    global camera_images
    with camera_assign_lock:
        camera_images = {}
    return jsonify({"success": True, "message": "All cameras cleared"})

# ── SINGLE IMAGE ANALYSIS ─────────────────────────────────────────────────────
@app.route("/api/analyse_image", methods=["POST"])
def api_analyse_image():
    if "image" not in request.files:
        return jsonify({"success":False,"error":"No image file provided"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"success":False,"error":"Empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"success":False,"error":f"Unsupported format: {ext}. Use JPG/PNG/BMP/TIFF/WEBP"}), 400

    fname    = f"{int(time.time())}_{secure_filename(f.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    f.save(filepath)

    ensure_yolo()

    if yolo_model is None and yolo_error is None:
        return jsonify({"success":False,"loading":True,"error":"Model loading… retry in a few seconds"}), 202

    cam_id = get_next_available_camera()
    result = analyse_image_file(filepath, f.filename, camera_id=cam_id)
    if "error" in result:
        return jsonify({"success":False,"error":result["error"]}), 500

    if cam_id:
        assign_image_to_camera(
            result["annotated_jpg_b64"], f.filename,
            result.get("detections",[]), result.get("count",0),
            result.get("top_threat","None"), result.get("top_confidence",0)
        )
        result["assigned_camera"] = cam_id
    else:
        result["assigned_camera"] = None
        result["camera_full"]     = True

    result["success"] = True
    return jsonify(result)

# ── SIMULATED DATASET ENDPOINT ─────────────────────────────────────────────────
@app.route("/api/dataset/generate_simulated", methods=["POST"])
def api_generate_simulated():
    """Generate and immediately queue a synthetic dataset for processing."""
    global dataset_state
    with dataset_lock:
        if dataset_state["running"]:
            return jsonify({"success":False,"error":"A dataset is already being processed"}), 409

    n = min(int(request.json.get("n", 20) if request.is_json else 20), 100)
    paths = generate_simulated_dataset(n)

    ensure_yolo()
    threading.Thread(target=run_dataset_worker, args=(paths,), daemon=True).start()

    return jsonify({
        "success": True,
        "message": f"Simulated dataset of {n} images queued for analysis",
        "total": n,
        "folder": SIM_FOLDER,
    })

# ── DATASET UPLOAD ────────────────────────────────────────────────────────────
@app.route("/api/upload_dataset", methods=["POST"])
def api_upload_dataset():
    global dataset_state
    with dataset_lock:
        if dataset_state["running"]:
            return jsonify({"success":False,"error":"A dataset is already being processed"}), 409

    files = request.files.getlist("images")
    if not files:
        return jsonify({"success":False,"error":"No images provided"}), 400

    saved_paths, skipped = [], []
    ds_folder = os.path.join(DATASET_FOLDER, f"ds_{int(time.time())}")
    os.makedirs(ds_folder, exist_ok=True)

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_IMAGE_EXT:
            skipped.append(f.filename); continue
        fname    = secure_filename(f.filename)
        filepath = os.path.join(ds_folder, fname)
        f.save(filepath); saved_paths.append(filepath)

    if not saved_paths:
        return jsonify({"success":False,"error":"No valid images found in upload"}), 400

    ensure_yolo()
    threading.Thread(target=run_dataset_worker, args=(saved_paths,), daemon=True).start()

    return jsonify({
        "success": True,
        "message": f"Dataset queued: {len(saved_paths)} images (skipped {len(skipped)})",
        "total": len(saved_paths), "skipped": skipped,
    })

# ── DATASET STATUS ────────────────────────────────────────────────────────────
@app.route("/api/dataset/status")
def api_dataset_status():
    with dataset_lock:
        s = dict(dataset_state)
    summary_results = []
    for r in s.get("results", []):
        summary_results.append({
            "filename": r.get("filename",""), "count": r.get("count",0),
            "top_threat": r.get("top_threat","None"),
            "top_confidence": r.get("top_confidence",0),
            "inf_ms": r.get("inf_ms",0), "error": r.get("error",""),
            "assigned_camera": r.get("assigned_camera"), "thumbnail": "",
        })
    s["results"] = summary_results
    return jsonify(s)

@app.route("/api/dataset/result/<int:idx>")
def api_dataset_result_image(idx):
    with dataset_lock:
        results = dataset_state.get("results",[])
    if idx >= len(results):
        return jsonify({"error":"Index out of range"}), 404
    r = results[idx]
    return jsonify({
        "filename": r.get("filename",""),
        "annotated_jpg_b64": r.get("annotated_jpg_b64",""),
        "detections": r.get("detections",[]),
        "count": r.get("count",0),
        "top_threat": r.get("top_threat","None"),
        "top_confidence": r.get("top_confidence",0),
        "inf_ms": r.get("inf_ms",0),
        "timestamp": r.get("timestamp",""),
        "assigned_camera": r.get("assigned_camera"),
    })

# ── DOWNLOAD RESULTS ──────────────────────────────────────────────────────────
@app.route("/api/dataset/download/<fmt>")
def api_dataset_download(fmt):
    with dataset_lock:
        json_file = dataset_state.get("json_file","")
        csv_file  = dataset_state.get("csv_file","")
    if fmt == "json":
        if not json_file or not os.path.exists(os.path.join(RESULTS_FOLDER,json_file)):
            return jsonify({"error":"No results file available yet"}), 404
        return send_from_directory(RESULTS_FOLDER, json_file, as_attachment=True, mimetype="application/json")
    elif fmt == "csv":
        if not csv_file or not os.path.exists(os.path.join(RESULTS_FOLDER,csv_file)):
            return jsonify({"error":"No results file available yet"}), 404
        return send_from_directory(RESULTS_FOLDER, csv_file, as_attachment=True, mimetype="text/csv")
    else:
        return jsonify({"error":"Invalid format. Use 'json' or 'csv'"}), 400

# ── SINGLE IMAGE HISTORY DOWNLOAD ─────────────────────────────────────────────
@app.route("/api/analysis/export")
def api_analysis_export():
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp,filename,mode,detection_count,top_threat,top_confidence,inference_ms,detections_json,camera_id "
        "FROM image_analysis ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["timestamp","filename","mode","detection_count","top_threat",
                "top_confidence_pct","inference_ms","camera_id","detections_json"])
    for r in rows:
        w.writerow([r["timestamp"],r["filename"],r["mode"],r["detection_count"],
                    r["top_threat"],round((r["top_confidence"] or 0)*100,1),
                    r["inference_ms"],r["camera_id"] or "",r["detections_json"]])
    output.seek(0)
    fname = f"agroshield_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})

# ── PHONE ALERT APIs ───────────────────────────────────────────────────────────
@app.route("/api/phone_alerts")
def api_phone_alerts():
    with phone_lock:
        return jsonify(phone_alert_log[:20])

@app.route("/api/phone_alerts/config", methods=["GET","POST"])
def api_phone_config():
    global ALERT_PHONE_NUMBER
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_num = data.get("phone","").strip()
        if new_num:
            ALERT_PHONE_NUMBER = new_num
            return jsonify({"success":True,"phone":ALERT_PHONE_NUMBER})
        return jsonify({"success":False,"error":"No phone number provided"}), 400
    return jsonify({"phone": ALERT_PHONE_NUMBER})

@app.route("/api/phone_alerts/test", methods=["POST"])
def api_phone_test():
    """Trigger a test phone alert."""
    entry = send_phone_alert("TEST ALERT", "Zone-A (Test)", 0.99, camera_id=1)
    push_alert("INFO","📞 Test Call Sent",f"Test alert dispatched to {ALERT_PHONE_NUMBER}")
    return jsonify({"success":True,"entry":entry})

# ── STANDARD APIS ──────────────────────────────────────────────────────────────
@app.route("/api/sensors")
def api_sensors(): return jsonify(sensors.get_all())

@app.route("/api/sensors/history")
def api_sensor_history():
    zone  = request.args.get("zone","Zone-A")
    hours = int(request.args.get("hours",24))
    since = (datetime.datetime.now()-datetime.timedelta(hours=hours)).isoformat()
    conn  = get_db()
    rows  = conn.execute(
        "SELECT timestamp,temperature,humidity,soil_moisture,motion_level "
        "FROM sensor_readings WHERE zone=? AND timestamp>? ORDER BY timestamp ASC LIMIT 200",
        (zone,since)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/alerts")
def api_alerts():
    with _alock: return jsonify(_alerts[:25])

@app.route("/api/threats/active")
def api_active():
    with _tlock:
        return jsonify({str(k):{"label":v["label"],"confidence":v["conf"],
                                 "severity":v["severity"],"zone":CAMERAS.get(k)}
                        for k,v in _active.items()})

@app.route("/api/analytics")
def api_analytics():
    conn = get_db()
    daily   = conn.execute("SELECT DATE(timestamp) day, COUNT(*) cnt FROM threat_events WHERE timestamp>=DATE('now','-30 days') GROUP BY day ORDER BY day").fetchall()
    by_type = conn.execute("SELECT threat_type, COUNT(*) cnt FROM threat_events GROUP BY threat_type ORDER BY cnt DESC LIMIT 8").fetchall()
    by_sev  = conn.execute("SELECT severity, COUNT(*) cnt FROM threat_events GROUP BY severity").fetchall()
    total   = conn.execute("SELECT COUNT(*) FROM threat_events").fetchone()[0]
    wk      = conn.execute("SELECT COUNT(*) FROM threat_events WHERE timestamp>=DATE('now','-7 days')").fetchone()[0]
    avg_cf  = conn.execute("SELECT AVG(confidence) FROM threat_events").fetchone()[0] or 0
    img_cnt = conn.execute("SELECT COUNT(*) FROM image_analysis").fetchone()[0]
    conn.close()
    return jsonify({"daily":[dict(r) for r in daily],"by_type":[dict(r) for r in by_type],
        "by_severity":[dict(r) for r in by_sev],
        "summary":{"total_threats":total,"this_week":wk,"accuracy":round(avg_cf*100,1),
                   "pesticide_reduction":68,"losses_prevented":"Rs.2.4L","images_analysed":img_cnt}})

@app.route("/api/yolo_status")
def api_yolo_status():
    return jsonify({"loaded": yolo_model is not None,
                    "loading": yolo_loading, "error": yolo_error})

# ── BOOT ──────────────────────────────────────────────────────────────────────
init_db()
ensure_yolo()
push_alert("INFO","AgroShield Online","Image Analysis + Simulation Mode — upload images or generate a simulated dataset")
threading.Thread(target=detection_loop,  daemon=True).start()
threading.Thread(target=sim_camera_loop, daemon=True).start()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AgroShield v3.0 — Image Analysis Edition")
    print("  Team Projexa 26E3134 | K.R. Mangalam University")
    print("  Team: Shubham Dey · Bhuveeta · Anjali · Jigyasa · Arshiya")
    print("="*60)
    print("  Open → http://localhost:5001")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=True)
