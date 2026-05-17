"""
LineCall Video Analyzer v2
===========================
Mejoras principales:
- Ball detection: frame differencing + BGS + color HSV combinados
- Court calibration: Hough lines (lineas blancas) en vez de color de cancha
- Bounce detection: analisis de trayectoria completa con suavizado
- Confidence scores para cada call
"""

import json, logging, os, uuid, traceback
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyzer")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict = {}
UPLOAD_DIR = Path("/tmp/linecall_videos")
UPLOAD_DIR.mkdir(exist_ok=True)


# ======================================================
# BALL DETECTOR — frame differencing + BGS + color
# ======================================================
class BallDetector:
    """
    3 metodos combinados:
    1. Background subtraction (MOG2) — detecta objetos en movimiento
    2. Frame differencing — detecta cambios entre frames consecutivos
    3. Color HSV — filtra por color de pelota de pickleball
    Candidatos cercanos se fusionan y se elige el de mayor confianza.
    """

    BALL_COLORS = [
        (np.array([18, 50, 100]), np.array([45, 255, 255])),   # amarillo-verde
        (np.array([20, 80, 150]), np.array([38, 255, 255])),   # amarillo brillante
        (np.array([38, 50, 100]), np.array([75, 255, 255])),   # verde-amarillo
        (np.array([5,  80, 120]), np.array([20, 255, 255])),   # naranja
        (np.array([0,   0, 180]), np.array([180, 45, 255])),   # blanco
    ]

    def __init__(self):
        self.prev_gray = None
        self.bgs = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=25, detectShadows=False
        )

    def reset(self):
        self.prev_gray = None
        self.bgs = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=25, detectShadows=False
        )

    def detect(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k3 = np.ones((3, 3), np.uint8)
        k5 = np.ones((5, 5), np.uint8)
        candidates = []

        # Method 1: Background subtraction
        fg = self.bgs.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k5)
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 15 or area > 4000:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4 * np.pi * area / (peri * peri)
            if circ < 0.35:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            candidates.append((float(cx), float(cy), float(r), circ * min(1.0, area/200)))

        # Method 2: Frame differencing
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, dt = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            dt = cv2.morphologyEx(dt, cv2.MORPH_CLOSE, k5)
            cnts2, _ = cv2.findContours(dt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts2:
                area = cv2.contourArea(cnt)
                if area < 10 or area > 3000:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circ = 4 * np.pi * area / (peri * peri)
                if circ < 0.30:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                candidates.append((float(cx), float(cy), float(r), circ * 0.7))

        # Method 3: Color HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for lo, hi in self.BALL_COLORS:
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
            cnts3, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts3:
                area = cv2.contourArea(cnt)
                if area < 12 or area > 5000:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circ = 4 * np.pi * area / (peri * peri)
                if circ < 0.40:
                    continue
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                candidates.append((float(cx), float(cy), float(r), circ * 0.8))

        self.prev_gray = gray

        if not candidates:
            return None

        # Merge nearby candidates
        merged = self._merge(candidates)
        if not merged:
            return None
        best = max(merged, key=lambda x: x[3])
        return best if best[3] > 0.15 else None

    def _merge(self, candidates):
        merged = []
        used   = [False] * len(candidates)
        for i, c in enumerate(candidates):
            if used[i]:
                continue
            group = [c]
            used[i] = True
            for j, c2 in enumerate(candidates):
                if used[j] or i == j:
                    continue
                if ((c[0]-c2[0])**2 + (c[1]-c2[1])**2)**0.5 < 30:
                    group.append(c2)
                    used[j] = True
            tc = sum(g[3] for g in group)
            if tc == 0:
                continue
            cx   = sum(g[0]*g[3] for g in group) / tc
            cy   = sum(g[1]*g[3] for g in group) / tc
            r    = sum(g[2]*g[3] for g in group) / tc
            conf = min(1.0, tc * (0.5 + 0.5 * len(group) / 3))
            merged.append((cx, cy, r, conf))
        return merged


# ======================================================
# TRAJECTORY ANALYZER
# ======================================================
class TrajectoryAnalyzer:
    def smooth(self, data, window=5):
        result = []
        for i in range(len(data)):
            s = max(0, i-window)
            e = min(len(data), i+window+1)
            result.append(sum(data[s:e]) / (e-s))
        return result

    def find_bounces(self, positions):
        if len(positions) < 8:
            return []
        frames    = [p[0] for p in positions]
        ys        = [p[2] for p in positions]
        confs     = [p[3] if len(p) > 3 else 1.0 for p in positions]
        ys_smooth = self.smooth(ys, window=4)
        bounces   = []
        last      = -999
        for i in range(3, len(ys_smooth) - 3):
            vb = ys_smooth[i]     - ys_smooth[max(0, i-4)]
            va = ys_smooth[min(len(ys_smooth)-1, i+4)] - ys_smooth[i]
            hit      = (vb > 2.5 and va < -2.0)
            hit_inv  = (vb < -2.5 and va > 2.0)
            gap_ok   = (frames[i] - last) > 10
            avg_conf = sum(confs[max(0,i-2):i+3]) / 5
            if (hit or hit_inv) and gap_ok and avg_conf > 0.15:
                bounces.append(frames[i])
                last = frames[i]
                log.info(f"  bounce frame={frames[i]} vb={vb:.1f} va={va:.1f} conf={avg_conf:.2f}")
        return bounces


# ======================================================
# COURT
# ======================================================
class Court:
    W   = 6.10
    H   = 13.41
    NVZ = 2.235

    def __init__(self):
        self.H_mat      = None
        self.court_poly = np.array([[0,0],[self.W,0],[self.W,self.H],[0,self.H]], dtype=np.float32)
        self.nvz_near   = np.array([[0,self.H-self.NVZ],[self.W,self.H-self.NVZ],[self.W,self.H],[0,self.H]], dtype=np.float32)
        self.nvz_far    = np.array([[0,0],[self.W,0],[self.W,self.NVZ],[0,self.NVZ]], dtype=np.float32)

    @property
    def ready(self):
        return self.H_mat is not None

    def calibrate(self, src_points):
        src = np.array(src_points, dtype=np.float32)
        dst = np.array([[0,0],[self.W,0],[self.W,self.H],[0,self.H]], dtype=np.float32)
        self.H_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return self.H_mat is not None

    def auto_calibrate(self, frame):
        # Primary: Hough line detection on white court lines
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        k = np.ones((3,3), np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)
        edges = cv2.Canny(white, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
            threshold=80, minLineLength=w*0.15, maxLineGap=30)

        if lines is not None and len(lines) >= 4:
            h_lines, v_lines = [], []
            for line in lines:
                x1,y1,x2,y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if angle < 20 or angle > 160:
                    h_lines.append((x1,y1,x2,y2))
                elif 70 < angle < 110:
                    v_lines.append((x1,y1,x2,y2))

            if len(h_lines) >= 2 and len(v_lines) >= 2:
                h_lines.sort(key=lambda l: min(l[1],l[3]))
                v_lines.sort(key=lambda l: min(l[0],l[2]))
                tl = self._intersect(h_lines[0],  v_lines[0])
                tr = self._intersect(h_lines[0],  v_lines[-1])
                br = self._intersect(h_lines[-1], v_lines[-1])
                bl = self._intersect(h_lines[-1], v_lines[0])
                if all([tl,tr,br,bl]):
                    corners = [list(tl),list(tr),list(br),list(bl)]
                    y_span  = abs(min(bl[1],br[1]) - min(tl[1],tr[1]))
                    x_span  = abs(max(tr[0],br[0]) - min(tl[0],bl[0]))
                    if y_span > h*0.20 and x_span > w*0.20:
                        log.info("Auto-calibrated via Hough lines")
                        return corners

        # Fallback: color-based
        return self._color_fallback(frame)

    def _intersect(self, l1, l2):
        x1,y1,x2,y2 = l1
        x3,y3,x4,y4 = l2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None
        t  = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        xi = x1 + t*(x2-x1)
        yi = y1 + t*(y2-y1)
        return (float(xi), float(yi))

    def _color_fallback(self, frame):
        h, w = frame.shape[:2]
        ranges = [
            (np.array([30,25,40]),  np.array([90,255,255])),
            (np.array([95,40,30]),  np.array([135,255,255])),
            (np.array([78,25,30]),  np.array([102,255,255])),
            (np.array([0, 0, 50]), np.array([180,60,200])),
        ]
        best_area = 0
        best_box  = None
        for lo, hi in ranges:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lo, hi)
            kk   = np.ones((11,11), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kk)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kk)
            cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            lg   = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(lg)
            if area < w*h*0.15 or area > w*h*0.93:
                continue
            hull = cv2.convexHull(lg)
            rect = cv2.minAreaRect(hull)
            box  = cv2.boxPoints(rect).astype(np.float32)
            s0   = np.linalg.norm(box[0]-box[1])
            s1   = np.linalg.norm(box[1]-box[2])
            if s0<1 or s1<1:
                continue
            ratio = max(s0,s1)/min(s0,s1)
            if ratio<1.1 or ratio>7.0:
                continue
            if area > best_area:
                best_area = area
                best_box  = self._order(box).tolist()
        return best_box

    def _order(self, pts):
        pts  = pts.reshape(4,2)
        rect = np.zeros((4,2), dtype=np.float32)
        s    = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d    = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def to_court(self, px, py):
        pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H_mat)
        return float(res[0][0][0]), float(res[0][0][1])

    def call(self, cx, cy):
        dist    = float(cv2.pointPolygonTest(self.court_poly, (cx,cy), True))
        result  = "IN" if dist >= -0.038 else "OUT"
        kitchen = None
        if cv2.pointPolygonTest(self.nvz_near,(cx,cy),False) >= 0:
            kitchen = "near"
        elif cv2.pointPolygonTest(self.nvz_far,(cx,cy),False) >= 0:
            kitchen = "far"
        return {"result":result, "dist_cm":round(abs(dist)*100,1),
                "court_x":round(cx,3), "court_y":round(cy,3), "kitchen":kitchen}


# ======================================================
# HELPERS
# ======================================================
def calc_speeds(calls, fps):
    speeds = []
    for i in range(1, len(calls)):
        a, b   = calls[i-1], calls[i]
        dx     = b["court_x"] - a["court_x"]
        dy     = b["court_y"] - a["court_y"]
        dist_m = (dx**2 + dy**2)**0.5
        dt     = b["timestamp_s"] - a["timestamp_s"]
        if 0.05 < dt < 4.0 and dist_m > 0.05:
            s = dist_m / dt * 3.6
            if 3 < s < 200:
                speeds.append(round(s,1))
    return speeds

def build_heatmap(calls):
    COLS, ROWS = 12, 22
    grid = [[0]*COLS for _ in range(ROWS)]
    for c in calls:
        if c.get("court_x") is not None:
            col = min(COLS-1, max(0, int(c["court_x"]/Court.W*COLS)))
            row = min(ROWS-1, max(0, int(c["court_y"]/Court.H*ROWS)))
            grid[row][col] += 1
    return grid

def fmt_time(s):
    m = int(s//60)
    return f"{m}:{s%60:05.2f}"


# ======================================================
# PROCESSOR
# ======================================================
def process_video(job_id, video_path, court_points=None):
    try:
        JOBS[job_id]["status"]   = "processing"
        JOBS[job_id]["progress"] = 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s   = total_frames / fps

        log.info(f"[{job_id}] {width}x{height} @{fps:.1f}fps {total_frames}fr {duration_s:.1f}s")

        detector = BallDetector()
        court    = Court()
        analyzer = TrajectoryAnalyzer()

        # Step 1: Court calibration
        JOBS[job_id]["progress"] = 5
        if court_points:
            ok = court.calibrate(court_points)
            log.info(f"[{job_id}] Manual cal: {'OK' if ok else 'FAIL'}")
        else:
            for fi in [0, 30, 60, 90, 120, 150]:
                if fi >= total_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret:
                    continue
                pts = court.auto_calibrate(frame)
                if pts:
                    court.calibrate(pts)
                    log.info(f"[{job_id}] Auto-cal OK at frame {fi}")
                    break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Step 2: Warmup background model
        warmup = min(60, total_frames // 4)
        for fi in range(warmup):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                detector.bgs.apply(frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        detector.prev_gray = None

        # Step 3: Ball detection
        log.info(f"[{job_id}] Detecting ball...")
        positions  = []
        frame_num  = 0
        detections = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            JOBS[job_id]["progress"] = 10 + int((frame_num/total_frames)*55)
            det = detector.detect(frame)
            if det:
                cx, cy, r, conf = det
                if conf > 0.2:
                    positions.append((frame_num, cx, cy, conf))
                    detections += 1
            frame_num += 1

        cap.release()
        log.info(f"[{job_id}] {detections}/{frame_num} frames ({detections/max(1,frame_num)*100:.1f}%)")
        JOBS[job_id]["progress"] = 70

        # Step 4: Bounces
        log.info(f"[{job_id}] Finding bounces...")
        bounce_frames = analyzer.find_bounces(positions)
        log.info(f"[{job_id}] {len(bounce_frames)} bounces")
        JOBS[job_id]["progress"] = 80

        # Step 5: IN/OUT calls
        calls   = []
        pos_map = {p[0]: p for p in positions}

        for bf in bounce_frames:
            closest, min_d = None, 999
            for fi in range(bf-3, bf+4):
                if fi in pos_map and abs(fi-bf) < min_d:
                    min_d   = abs(fi-bf)
                    closest = pos_map[fi]
            if not closest:
                continue
            fn, px, py, conf = closest
            ts = bf / fps
            if court.ready:
                try:
                    cx, cy  = court.to_court(px, py)
                    verdict = court.call(cx, cy)
                except Exception as e:
                    log.warning(f"Transform err: {e}")
                    verdict = {"result":"UNKNOWN","dist_cm":0,"court_x":0,"court_y":0,"kitchen":None}
            else:
                verdict = {"result":"UNKNOWN","dist_cm":0,
                           "court_x":round(px/width*Court.W,3),
                           "court_y":round(py/height*Court.H,3),"kitchen":None}
            calls.append({"frame":bf,"timestamp_s":round(ts,3),"timestamp":fmt_time(ts),
                          "px":round(px),"py":round(py),"confidence":round(conf,2), **verdict})

        JOBS[job_id]["progress"] = 90

        # Step 6: Stats
        in_c   = [c for c in calls if c["result"]=="IN"]
        out_c  = [c for c in calls if c["result"]=="OUT"]
        speeds = calc_speeds(calls, fps)
        stats  = {
            "total_bounces":   len(calls),
            "in_calls":        len(in_c),
            "out_calls":       len(out_c),
            "detection_rate":  round(detections/max(1,frame_num)*100,1),
            "avg_speed_kmh":   round(sum(speeds)/len(speeds),1) if speeds else 0,
            "max_speed_kmh":   round(max(speeds),1) if speeds else 0,
            "duration_s":      round(duration_s,1),
            "fps":             round(fps,1),
            "total_frames":    total_frames,
            "calibrated":      court.ready,
        }
        result = {
            "job_id":     job_id, "status":"done",
            "video_info": {"width":width,"height":height,"fps":round(fps,1),
                           "duration_s":round(duration_s,1),"total_frames":total_frames},
            "calibrated": court.ready,
            "calls":      calls, "stats":stats,
            "heatmap":    build_heatmap(calls),
            "speeds":     speeds[:50],
        }
        JOBS[job_id].update({"status":"done","progress":100,"result":result})
        log.info(f"[{job_id}] Done: {len(calls)} calls, {len(out_c)} OUT")
        try: os.remove(video_path)
        except: pass

    except Exception as e:
        log.error(f"[{job_id}] {e}\n{traceback.format_exc()}")
        JOBS[job_id].update({"status":"error","error":str(e)})


# ======================================================
# ENDPOINTS
# ======================================================
@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    court_points: Optional[str] = Form(None),
):
    job_id    = str(uuid.uuid4())[:8]
    ext       = Path(video.filename).suffix or ".mp4"
    save_path = str(UPLOAD_DIR / f"{job_id}{ext}")
    content   = await video.read()
    with open(save_path,"wb") as f:
        f.write(content)
    log.info(f"[{job_id}] {video.filename} ({len(content)//1024}KB)")
    pts = None
    if court_points:
        try: pts = json.loads(court_points)
        except: pass
    JOBS[job_id] = {"status":"queued","progress":0,"result":None}
    background_tasks.add_task(process_video, job_id, save_path, pts)
    return {"job_id":job_id,"status":"queued"}

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error":"Not found"},404)
    j = JOBS[job_id]
    return {"job_id":job_id,"status":j["status"],"progress":j.get("progress",0),"error":j.get("error")}

@app.get("/result/{job_id}")
def result(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error":"Not found"},404)
    j = JOBS[job_id]
    if j["status"] != "done":
        return JSONResponse({"error":"Not ready","status":j["status"]},425)
    return j["result"]

@app.get("/health")
def health():
    return {"status":"ok","jobs":len(JOBS),"version":"2.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT",8001))
    log.info(f"Analyzer v2 port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
