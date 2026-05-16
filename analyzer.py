"""
LineCall Video Analyzer
========================
Procesa un video completo y retorna todos los calls IN/OUT
con timestamps, posiciones y estadísticas.

Endpoints:
  POST /analyze        — sube video, retorna job_id
  GET  /status/{id}    — estado del procesamiento
  GET  /result/{id}    — resultado completo
  GET  /health         — health check

Instalar:
  pip install fastapi uvicorn[standard] opencv-python-headless numpy python-multipart
"""

import asyncio, json, logging, os, time, uuid, traceback
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacenamiento en memoria de jobs
JOBS: dict = {}
UPLOAD_DIR = Path("/tmp/linecall_videos")
UPLOAD_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════
# BALL DETECTOR — por color + forma
# ══════════════════════════════════════════════════════════
class BallDetector:
    """
    Detecta la pelota de pickleball por color HSV + circularidad.
    Mucho más confiable que YOLO para video de cancha.
    """
    COLORS = [
        # Amarillo-verde fluorescente (la más común)
        (np.array([22, 60, 120]),  np.array([45, 255, 255])),
        # Verde-amarillo
        (np.array([40, 60, 120]),  np.array([80, 255, 255])),
        # Naranja
        (np.array([5,  80, 120]),  np.array([22, 255, 255])),
        # Amarillo brillante (indoor)
        (np.array([20, 40, 180]),  np.array([35, 255, 255])),
        # Blanco (indoor)
        (np.array([0,   0, 200]),  np.array([180, 30, 255])),
    ]

    def detect(self, frame: np.ndarray) -> Optional[tuple]:
        """Retorna (cx, cy, radius) o None."""
        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # También buscar usando diferencia de frames si hay previo
        best       = None
        best_score = 0

        for lo, hi in self.COLORS:
            mask = cv2.inRange(hsv, lo, hi)
            k    = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            for cnt in cnts:
                area = cv2.contourArea(cnt)
                # Pelota de pickleball: 5-7cm diámetro
                # En video típico ocupa 10-200 px²
                if area < 8 or area > 6000:
                    continue

                peri  = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circ  = 4 * np.pi * area / (peri * peri)
                if circ < 0.40: continue  # debe ser circular

                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                score = area * circ
                if score > best_score:
                    best_score = score
                    best = (float(cx), float(cy), float(r))

        return best


# ══════════════════════════════════════════════════════════
# TRAJECTORY ANALYZER — analiza trayectoria completa
# ══════════════════════════════════════════════════════════
class TrajectoryAnalyzer:
    """
    Analiza la trayectoria completa del video para detectar
    bounces con alta precisión usando toda la información disponible.
    """

    def find_bounces(self, positions: list) -> list:
        """
        positions: [(frame, x, y), ...]
        Retorna lista de frames donde hay bounce.
        
        Algoritmo:
        1. Suavizar la trayectoria vertical (spline)
        2. Encontrar mínimos locales en y (punto más bajo = bounce en imagen)
        3. Verificar que la velocidad cambia de signo (bajando → subiendo)
        """
        if len(positions) < 6:
            return []

        frames = [p[0] for p in positions]
        ys     = [p[2] for p in positions]  # coordenada y (vertical en imagen)

        # Suavizar con media móvil
        def smooth(data, w=3):
            result = []
            for i in range(len(data)):
                start = max(0, i-w)
                end   = min(len(data), i+w+1)
                result.append(sum(data[start:end]) / (end-start))
            return result

        ys_smooth = smooth(ys, w=4)

        bounces = []
        min_gap = 8  # mínimo frames entre bounces

        for i in range(2, len(ys_smooth)-2):
            # Velocidad antes y después
            vy_before = ys_smooth[i] - ys_smooth[max(0,i-3)]
            vy_after  = ys_smooth[min(len(ys_smooth)-1,i+3)] - ys_smooth[i]

            # Bounce: bajando (vy_before > 0) → subiendo (vy_after < 0)
            # En coordenadas imagen: y aumenta hacia abajo
            if vy_before > 3.0 and vy_after < -2.0:
                # Verificar que es mínimo local
                if ys_smooth[i] >= max(ys_smooth[max(0,i-2):i]):
                    # Verificar gap con bounce anterior
                    if not bounces or (frames[i] - bounces[-1]) > min_gap:
                        bounces.append(frames[i])
                        log.info(f"  Bounce en frame {frames[i]}: vy_before={vy_before:.1f} vy_after={vy_after:.1f}")

        return bounces


# ══════════════════════════════════════════════════════════
# COURT — calibración y line calls
# ══════════════════════════════════════════════════════════
class Court:
    W   = 6.10   # metros
    H   = 13.41  # metros
    NVZ = 2.235  # metros desde la red

    def __init__(self):
        self.H_mat = None
        self.court_poly = np.array([
            [0,0],[self.W,0],[self.W,self.H],[0,self.H]
        ], dtype=np.float32)
        self.nvz_near = np.array([
            [0,self.H-self.NVZ],[self.W,self.H-self.NVZ],[self.W,self.H],[0,self.H]
        ], dtype=np.float32)
        self.nvz_far = np.array([
            [0,0],[self.W,0],[self.W,self.NVZ],[0,self.NVZ]
        ], dtype=np.float32)

    def calibrate(self, src_points: list) -> bool:
        src = np.array(src_points, dtype=np.float32)
        dst = np.array([
            [0,0],[self.W,0],[self.W,self.H],[0,self.H]
        ], dtype=np.float32)
        self.H_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return self.H_mat is not None

    def auto_calibrate(self, frame: np.ndarray) -> Optional[list]:
        """Detecta cancha automáticamente por color."""
        h, w = frame.shape[:2]
        ranges = [
            (np.array([30,25,40]),  np.array([90,255,255])),
            (np.array([95,40,30]),  np.array([135,255,255])),
            (np.array([78,25,30]),  np.array([102,255,255])),
        ]
        best = None
        best_area = 0
        for lo, hi in ranges:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lo, hi)
            k    = np.ones((11,11), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            largest = max(cnts, key=cv2.contourArea)
            area    = cv2.contourArea(largest)
            if area < w*h*0.15 or area > w*h*0.93: continue
            hull = cv2.convexHull(largest)
            rect = cv2.minAreaRect(hull)
            box  = cv2.boxPoints(rect).astype(np.float32)
            s0   = np.linalg.norm(box[0]-box[1])
            s1   = np.linalg.norm(box[1]-box[2])
            if s0<1 or s1<1: continue
            ratio = max(s0,s1)/min(s0,s1)
            if ratio < 1.1 or ratio > 7.0: continue
            if area > best_area:
                best_area = area
                best = self._order(box).tolist()
        return best

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
        pt  = np.array([[[float(px),float(py)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H_mat)
        return float(res[0][0][0]), float(res[0][0][1])

    def call(self, cx, cy) -> dict:
        dist = float(cv2.pointPolygonTest(self.court_poly,(cx,cy),True))
        TOLERANCE = 0.038  # radio pelota ~3.8cm
        result = "IN" if dist >= -TOLERANCE else "OUT"
        kitchen = None
        if cv2.pointPolygonTest(self.nvz_near,(cx,cy),False) >= 0: kitchen="near"
        elif cv2.pointPolygonTest(self.nvz_far,(cx,cy),False) >= 0: kitchen="far"
        return {
            "result":  result,
            "dist_cm": round(abs(dist)*100, 1),
            "court_x": round(cx, 3),
            "court_y": round(cy, 3),
            "kitchen": kitchen,
        }

    @property
    def ready(self):
        return self.H_mat is not None


# ══════════════════════════════════════════════════════════
# MAIN PROCESSOR — procesa el video completo
# ══════════════════════════════════════════════════════════
def process_video(job_id: str, video_path: str, court_points: Optional[list] = None):
    """
    Procesa el video completo:
    1. Detecta la pelota en cada frame
    2. Encuentra todos los bounces
    3. Hace call IN/OUT en cada bounce
    4. Retorna resultado completo
    """
    try:
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["progress"] = 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("No se pudo abrir el video")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s  = total_frames / fps

        log.info(f"[{job_id}] Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames ({duration_s:.1f}s)")

        detector   = BallDetector()
        court      = Court()
        analyzer   = TrajectoryAnalyzer()

        # ── Paso 1: Calibrar cancha ───────────────────────
        if court_points:
            ok = court.calibrate(court_points)
            log.info(f"[{job_id}] Calibración manual: {'OK' if ok else 'FAIL'}")
        else:
            # Auto-calibrar con los primeros 30 frames
            log.info(f"[{job_id}] Intentando auto-calibración...")
            for fi in range(min(30, total_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if not ret: continue
                pts = court.auto_calibrate(frame)
                if pts:
                    court.calibrate(pts)
                    log.info(f"[{job_id}] Auto-calibración OK en frame {fi}")
                    break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not court.ready:
            log.warning(f"[{job_id}] Sin calibración de cancha — calls serán en píxeles relativos")

        # ── Paso 2: Detectar pelota en cada frame ─────────
        log.info(f"[{job_id}] Detectando pelota...")
        positions = []  # [(frame_num, x_px, y_px, radius)]
        prev_gray = None

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Actualizar progreso
            JOBS[job_id]["progress"] = int((frame_num / total_frames) * 60)

            # Procesar 1 de cada 2 frames para velocidad
            # (interpolamos los que saltamos)
            if frame_num % 2 == 0:
                det = detector.detect(frame)
                if det:
                    cx, cy, r = det
                    positions.append((frame_num, cx, cy, r))

            frame_num += 1

        cap.release()
        log.info(f"[{job_id}] {len(positions)} detecciones en {frame_num} frames")
        JOBS[job_id]["progress"] = 65

        # ── Paso 3: Encontrar bounces ─────────────────────
        log.info(f"[{job_id}] Analizando trayectoria...")
        pos_for_analysis = [(p[0], p[1], p[2]) for p in positions]
        bounce_frames    = analyzer.find_bounces(pos_for_analysis)
        log.info(f"[{job_id}] {len(bounce_frames)} bounces detectados")
        JOBS[job_id]["progress"] = 80

        # ── Paso 4: Call IN/OUT en cada bounce ────────────
        log.info(f"[{job_id}] Calculando calls...")
        calls = []

        # Crear mapa frame → posición para búsqueda rápida
        pos_map = {p[0]: p for p in positions}

        for bf in bounce_frames:
            # Buscar la posición más cercana al frame del bounce
            closest = None
            min_dist = 999
            for fi in range(bf-2, bf+3):
                if fi in pos_map:
                    d = abs(fi - bf)
                    if d < min_dist:
                        min_dist = d
                        closest  = pos_map[fi]

            if not closest:
                continue

            _, px, py, r = closest
            timestamp_s  = bf / fps

            if court.ready:
                try:
                    cx, cy  = court.to_court(px, py)
                    verdict = court.call(cx, cy)
                except Exception as e:
                    log.warning(f"Court transform error: {e}")
                    verdict = {"result":"UNKNOWN","dist_cm":0,"court_x":0,"court_y":0,"kitchen":None}
            else:
                # Sin calibración: estimar por posición relativa en frame
                rel_x = px / width
                rel_y = py / height
                verdict = {
                    "result":  "UNKNOWN",
                    "dist_cm": 0,
                    "court_x": round(rel_x * 6.10, 3),
                    "court_y": round(rel_y * 13.41, 3),
                    "kitchen": None,
                }

            calls.append({
                "frame":       bf,
                "timestamp_s": round(timestamp_s, 3),
                "timestamp":   _fmt_time(timestamp_s),
                "px":          round(px),
                "py":          round(py),
                "result":      verdict["result"],
                "dist_cm":     verdict["dist_cm"],
                "court_x":     verdict["court_x"],
                "court_y":     verdict["court_y"],
                "kitchen":     verdict["kitchen"],
            })

        JOBS[job_id]["progress"] = 90

        # ── Paso 5: Estadísticas ──────────────────────────
        in_calls    = [c for c in calls if c["result"] == "IN"]
        out_calls   = [c for c in calls if c["result"] == "OUT"]
        kit_faults  = [c for c in calls if c["kitchen"] and c["result"] == "IN"]

        # Heatmap de bounces en cancha (grid 10x20)
        heatmap = _build_heatmap(calls)

        # Velocidades (distancia entre bounces consecutivos)
        speeds = _calc_speeds(calls, fps)

        stats = {
            "total_bounces":  len(calls),
            "in_calls":       len(in_calls),
            "out_calls":      len(out_calls),
            "kitchen_faults": len(kit_faults),
            "detection_rate": round(len(positions)/max(1,frame_num//2)*100, 1),
            "avg_speed_kmh":  round(sum(speeds)/len(speeds), 1) if speeds else 0,
            "max_speed_kmh":  round(max(speeds), 1) if speeds else 0,
            "duration_s":     round(duration_s, 1),
            "fps":            round(fps, 1),
            "total_frames":   total_frames,
            "calibrated":     court.ready,
        }

        result = {
            "job_id":      job_id,
            "status":      "done",
            "video_info":  {"width":width,"height":height,"fps":round(fps,1),"duration_s":round(duration_s,1),"total_frames":total_frames},
            "calibrated":  court.ready,
            "calls":       calls,
            "stats":       stats,
            "heatmap":     heatmap,
            "speeds":      speeds[:50],  # máximo 50 para no sobrecargar
        }

        JOBS[job_id].update({"status":"done","progress":100,"result":result})
        log.info(f"[{job_id}] Análisis completo: {len(calls)} calls ({len(out_calls)} OUT)")

        # Limpiar video
        try: os.remove(video_path)
        except: pass

    except Exception as e:
        log.error(f"[{job_id}] Error: {e}\n{traceback.format_exc()}")
        JOBS[job_id].update({"status":"error","error":str(e)})


def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:05.2f}"


def _build_heatmap(calls: list) -> list:
    """Grid 12x22 con conteo de bounces por celda."""
    COLS, ROWS = 12, 22
    W, H = 6.10, 13.41
    grid = [[0]*COLS for _ in range(ROWS)]
    for c in calls:
        if c["court_x"] and c["court_y"]:
            col = min(COLS-1, max(0, int(c["court_x"] / W * COLS)))
            row = min(ROWS-1, max(0, int(c["court_y"] / H * ROWS)))
            grid[row][col] += 1
    return grid


def _calc_speeds(calls: list, fps: float) -> list:
    """Velocidad entre bounces consecutivos en km/h."""
    speeds = []
    for i in range(1, len(calls)):
        a, b = calls[i-1], calls[i]
        dx = (b["court_x"] - a["court_x"])
        dy = (b["court_y"] - a["court_y"])
        dist_m = (dx**2 + dy**2)**0.5
        dt_s   = b["timestamp_s"] - a["timestamp_s"]
        if 0.1 < dt_s < 3.0 and dist_m > 0.1:
            speed = dist_m / dt_s * 3.6
            if 5 < speed < 180:
                speeds.append(round(speed, 1))
    return speeds


# ══════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    court_points: Optional[str] = Form(None),  # JSON string
):
    """Sube un video y comienza el análisis en background."""
    job_id    = str(uuid.uuid4())[:8]
    ext       = Path(video.filename).suffix or ".mp4"
    save_path = str(UPLOAD_DIR / f"{job_id}{ext}")

    # Guardar video
    content = await video.read()
    with open(save_path, "wb") as f:
        f.write(content)

    log.info(f"[{job_id}] Video recibido: {video.filename} ({len(content)//1024}KB)")

    # Parsear puntos de calibración si vienen
    pts = None
    if court_points:
        try: pts = json.loads(court_points)
        except: pass

    # Inicializar job
    JOBS[job_id] = {"status":"queued","progress":0,"result":None}

    # Procesar en background
    background_tasks.add_task(process_video, job_id, save_path, pts)

    return {"job_id": job_id, "status": "queued"}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error":"Job not found"}, 404)
    job = JOBS[job_id]
    return {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job.get("progress", 0),
        "error":    job.get("error"),
    }


@app.get("/result/{job_id}")
def result(job_id: str):
    if job_id not in JOBS:
        return JSONResponse({"error":"Job not found"}, 404)
    job = JOBS[job_id]
    if job["status"] != "done":
        return JSONResponse({"error":"Not ready yet","status":job["status"]}, 425)
    return job["result"]


@app.get("/health")
def health():
    return {"status":"ok","jobs":len(JOBS)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    log.info(f"Analyzer arrancando en puerto {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
