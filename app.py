"""
AgroShield — Smart Crop Protection System
Full-stack Flask + SQLite + OpenCV
Ideathon 2026 | Team Projexa 26E3134 | K.R. Mangalam University
"""
import json, math, random, sqlite3, threading, time, base64, datetime, os
import numpy as np
import cv2
from flask import Flask, jsonify, render_template, Response, request

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "agroshield.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
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
            soil_moisture REAL, motion_level REAL
        );
        CREATE TABLE IF NOT EXISTS threat_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, zone TEXT NOT NULL,
            camera_id INTEGER, threat_type TEXT NOT NULL,
            confidence REAL, severity TEXT,
            action_taken TEXT, resolved INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL, level TEXT NOT NULL,
            title TEXT NOT NULL, message TEXT NOT NULL,
            acknowledged INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()
    _seed_data()

def _seed_data():
    conn = get_db()
    if conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0] > 0:
        conn.close(); return
    zones  = ["Zone-A","Zone-B","Zone-C","Zone-D"]
    threats = [("Wild Boar","CRITICAL"),("Deer","CRITICAL"),("Monkeys","CRITICAL"),
               ("Aphid Cluster","WARNING"),("Caterpillar","WARNING"),
               ("Leaf Blight","WARNING"),("Stem Borer","WARNING"),("Rabbits","WARNING")]
    now = datetime.datetime.now()
    sr, te = [], []
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
    conn.executemany("INSERT INTO sensor_readings (timestamp,zone,temperature,humidity,soil_moisture,motion_level) VALUES(?,?,?,?,?,?)", sr)
    conn.executemany("INSERT INTO threat_events (timestamp,zone,camera_id,threat_type,confidence,severity,action_taken,resolved) VALUES(?,?,?,?,?,?,?,?)", te)
    conn.commit(); conn.close()

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
                    d["temp"]   = round(d["temp"]  + random.gauss(0,0.15), 1)
                    d["hum"]    = round(max(30,min(95, d["hum"]  +random.gauss(0,0.4))),1)
                    d["soil"]   = round(max(10,min(90, d["soil"] +random.gauss(0,0.3))),1)
                    d["motion"] = round(max(0, min(100,d["motion"]+random.gauss(0,3))),1)
            try:
                conn = get_db()
                ts = datetime.datetime.now().isoformat()
                with self._lock:
                    rows = [(ts,z,d["temp"],d["hum"],d["soil"],d["motion"]) for z,d in self.data.items()]
                conn.executemany("INSERT INTO sensor_readings (timestamp,zone,temperature,humidity,soil_moisture,motion_level) VALUES(?,?,?,?,?,?)", rows)
                conn.commit(); conn.close()
            except: pass

    def get_all(self):
        with self._lock: return {k:dict(v) for k,v in self.data.items()}

sensors = SensorSim()

THREAT_DEFS = [
    {"label":"Wild Boar",    "emoji":"Boar",  "severity":"CRITICAL","color":(60,60,240)},
    {"label":"Deer",         "emoji":"Deer",  "severity":"CRITICAL","color":(60,60,240)},
    {"label":"Monkeys",      "emoji":"Monkey","severity":"CRITICAL","color":(60,60,240)},
    {"label":"Aphid Cluster","emoji":"Aphid", "severity":"WARNING", "color":(36,180,251)},
    {"label":"Caterpillar",  "emoji":"Larva", "severity":"WARNING", "color":(36,180,251)},
    {"label":"Leaf Blight",  "emoji":"Blight","severity":"WARNING", "color":(60,130,251)},
]
CAMERAS = {1:"Zone-A North",2:"Zone-A South",3:"Zone-B East",
           4:"Zone-B West", 5:"Zone-C Main", 6:"Zone-D Gate"}

_ftick = 0
_active = {}
_tlock  = threading.Lock()

def render_frame(cam_id:int, w=640, h=480) -> np.ndarray:
    global _ftick; _ftick += 1
    t = _ftick * 0.04 + cam_id
    frame = np.zeros((h,w,3), np.uint8)
    for y in range(int(h*0.45)):
        v = int(8 + y*0.12); frame[y,:] = [v+2, v+10, v+2]
    for y in range(int(h*0.45), h):
        v = int(10+(y-h*0.45)*0.05); frame[y,:] = [v, v+12, v]
    rs = int(h*0.48)
    for row in range(9):
        yb = rs + row*20
        if yb >= h: break
        gi = int(80*max(0.1,1-row*0.08))
        for col in range(20):
            x = int(col*(w/19))
            sw = int(math.sin(t+col*0.5+row*0.3)*3)
            cv2.line(frame,(x+sw,yb+12),(x+sw*2,yb),(18,gi,10),1)
            cv2.ellipse(frame,(x+sw*2,yb-2),(6,3),20,0,180,(14,gi+20,10),-1)
    rng = np.random.default_rng(cam_id*100)
    for _ in range(20):
        sx,sy = int(rng.uniform(0,w)), int(rng.uniform(0,h*0.4))
        bri = int(100+80*math.sin(t*0.5+sx*0.1))
        cv2.circle(frame,(sx,sy),1,(bri,bri,bri),-1)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.rectangle(frame,(0,0),(230,18),(0,0,0),-1)
    cv2.putText(frame,f"REC  {CAMERAS.get(cam_id)}  {ts}",(6,13),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,(60,220,80),1)
    with _tlock: th = _active.get(cam_id)
    if th:
        if int(_ftick*0.5)%2==0:
            cv2.rectangle(frame,(3,3),(w-3,h-3),(60,60,240),2)
        bx,by,bw,bh = th["box"]
        col = th["color"]
        cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),col,2)
        lbl = f"{th['label']} {int(th['conf']*100)}%"
        (lw,lh),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        cv2.rectangle(frame,(bx,by-lh-8),(bx+lw+8,by),col,-1)
        cv2.putText(frame,lbl,(bx+4,by-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        bar_y = by+bh+6
        if bar_y+10<h:
            cv2.rectangle(frame,(bx,bar_y),(bx+bw,bar_y+6),(40,40,40),-1)
            cv2.rectangle(frame,(bx,bar_y),(bx+int(bw*th["conf"]),bar_y+6),col,-1)
    return frame

def mjpeg(cam_id:int):
    while True:
        frame = render_frame(cam_id)
        _,buf = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,78])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
        time.sleep(0.1)

_alerts = []
_alock  = threading.Lock()

def push_alert(level,title,message):
    ts = datetime.datetime.now().isoformat()
    obj = {"id":int(time.time()*1000+random.randint(0,999)),
           "timestamp":ts,"level":level,"title":title,"message":message}
    with _alock:
        _alerts.insert(0,obj)
        del _alerts[50:]
    try:
        conn=get_db()
        conn.execute("INSERT INTO alerts(timestamp,level,title,message) VALUES(?,?,?,?)",(ts,level,title,message))
        conn.commit(); conn.close()
    except: pass

def detection_loop():
    EMOJI_MAP={"Wild Boar":"[Boar]","Deer":"[Deer]","Monkeys":"[Monkey]",
               "Aphid Cluster":"[Pest]","Caterpillar":"[Larva]","Leaf Blight":"[Blight]"}
    while True:
        time.sleep(random.uniform(7,15))
        if random.random()<0.55:
            cid = random.randint(1,6)
            th  = random.choice(THREAT_DEFS)
            cf  = round(random.uniform(0.74,0.97),2)
            bw,bh = random.randint(80,180),random.randint(60,140)
            bx,by = random.randint(40,460-bw),random.randint(160,320-bh)
            with _tlock:
                _active[cid] = {"label":th["label"],"conf":cf,
                                 "box":(bx,by,bw,bh),"color":th["color"],"severity":th["severity"]}
            zone = CAMERAS.get(cid,"Unknown")
            em = EMOJI_MAP.get(th["label"],"[!]")
            push_alert(th["severity"],f"{em} {th['label']} Detected!",
                       f"{zone} - Cam {cid} - {int(cf*100)}% confidence - Deterrent activated")
            try:
                conn=get_db()
                conn.execute("INSERT INTO threat_events(timestamp,zone,camera_id,threat_type,confidence,severity,action_taken) VALUES(?,?,?,?,?,?,?)",
                    (datetime.datetime.now().isoformat(),zone,cid,th["label"],cf,th["severity"],"Deterrent+SMS"))
                conn.commit(); conn.close()
            except: pass
            def clear(c=cid):
                time.sleep(random.uniform(8,16))
                with _tlock: _active.pop(c,None)
                push_alert("INFO","Zone Cleared",f"Camera {c} - threat no longer detected")
            threading.Thread(target=clear,daemon=True).start()

@app.route("/")
def index(): return render_template("index.html")

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

@app.route("/video/<int:cid>")
def video(cid):
    if cid not in CAMERAS: return "Not found",404
    return Response(mjpeg(cid),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/detect/<int:cid>", methods=["POST"])
def detect(cid):
    th  = random.choice(THREAT_DEFS)
    cf  = round(random.uniform(0.78,0.97),2)
    bw,bh = random.randint(90,170),random.randint(70,130)
    bx,by = random.randint(40,450),random.randint(160,310)
    with _tlock:
        _active[cid] = {"label":th["label"],"conf":cf,"box":(bx,by,bw,bh),
                         "color":th["color"],"severity":th["severity"]}
    push_alert(th["severity"],f"{th['label']} Detected (Manual Scan)",
               f"Cam {cid} - {CAMERAS.get(cid)} - {int(cf*100)}% - Deterrent activated")
    return jsonify({"threat":th["label"],"confidence":cf,"severity":th["severity"]})

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
    daily = conn.execute(
        "SELECT DATE(timestamp) day, COUNT(*) cnt FROM threat_events "
        "WHERE timestamp>=DATE('now','-30 days') GROUP BY day ORDER BY day").fetchall()
    by_type = conn.execute(
        "SELECT threat_type, COUNT(*) cnt FROM threat_events GROUP BY threat_type ORDER BY cnt DESC LIMIT 8").fetchall()
    by_sev  = conn.execute(
        "SELECT severity, COUNT(*) cnt FROM threat_events GROUP BY severity").fetchall()
    total   = conn.execute("SELECT COUNT(*) FROM threat_events").fetchone()[0]
    wk      = conn.execute("SELECT COUNT(*) FROM threat_events WHERE timestamp>=DATE('now','-7 days')").fetchone()[0]
    avg_cf  = conn.execute("SELECT AVG(confidence) FROM threat_events").fetchone()[0] or 0
    conn.close()
    return jsonify({
        "daily":[dict(r) for r in daily],
        "by_type":[dict(r) for r in by_type],
        "by_severity":[dict(r) for r in by_sev],
        "summary":{"total_threats":total,"this_week":wk,
                   "accuracy":round(avg_cf*100,1),"pesticide_reduction":68,
                   "losses_prevented":"Rs.2.4L"}
    })

init_db()
push_alert("INFO","AgroShield Online","All sensors nominal - monitoring active across 4 zones")
threading.Thread(target=detection_loop,daemon=True).start()

if __name__=="__main__":
    print("\n" + "="*50)
    print("  AgroShield - Smart Crop Protection System")
    print("  Ideathon 2026 | Team Projexa 26E3134")
    print("="*50)
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=True)
