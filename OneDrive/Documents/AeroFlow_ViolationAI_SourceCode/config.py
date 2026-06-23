"""
config.py  —  AeroFlow Violation AI
Central configuration for the entire system.
Edit this file to adapt to any intersection or camera setup.
"""
import os

# ── Intersection identity ─────────────────────────────────────────────────────
INTERSECTION_NAME = "ITO Crossing"
CITY              = "Delhi"
STATE             = "Delhi"

# ── Lane → nearest CPCB monitoring station ────────────────────────────────────
LANES: dict[str, tuple[str, str]] = {
    "Lane 1": ("ITO",           "Delhi"),
    "Lane 2": ("Anand Vihar",   "Delhi"),
    "Lane 3": ("RK Puram",      "Delhi"),
    "Lane 4": ("Punjabi Bagh",  "Delhi"),
}

# ── YOLO Models ───────────────────────────────────────────────────────────────
YOLO_MODEL_PATH   = "yolov8n.pt"         # base vehicle detection
HELMET_MODEL_PATH = "helmet_model.pt"    # fine-tuned helmet detector
                                          # Download: https://huggingface.co/keremberke/yolov8n-helmet-detection
VIDEO_SOURCE      = 0                    # 0 = webcam | or a video file path

# ── Signal guardrails ─────────────────────────────────────────────────────────
MIN_GREEN_TIME   = 15
MAX_GREEN_TIME   = 60
YELLOW_INTERVAL  = 3
TOTAL_CYCLE_TIME = 120

# ── CPCB AQI API ──────────────────────────────────────────────────────────────
CPCB_API_KEY   = os.getenv("CPCB_API_KEY", "")
AQI_CACHE_TTL  = 3600
AQI_POLLUTANT  = "PM2.5"
CSV_FALLBACK   = "aqi_dataset.csv"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = "lane_logs.csv"
LOG_COLS = [
    "Timestamp", "Lane", "VehicleCount",
    "High", "Medium", "BS-VI", "Clean",
    "Score", "AQI", "AQICategory", "Priority", "GreenTime",
]

# ── Evidence output ───────────────────────────────────────────────────────────
EVIDENCE_DIR         = "evidence"
EVIDENCE_FRAMES_DIR  = os.path.join(EVIDENCE_DIR, "frames")
EVIDENCE_REPORTS_DIR = os.path.join(EVIDENCE_DIR, "reports")
VIOLATIONS_LOG       = os.path.join(EVIDENCE_DIR, "violations.csv")
VIOLATIONS_LOG_COLS  = [
    "ViolationID", "Timestamp", "FrameID", "TrackID",
    "VehicleClass", "ViolationType", "Confidence",
    "BBoxX1", "BBoxY1", "BBoxX2", "BBoxY2",
    "PlateText", "PlateConfidence", "EvidenceFramePath",
]

# ── Stop-line configuration ───────────────────────────────────────────────────
# Y pixel coordinate of the stop line in the camera frame.
# Set to None to auto-detect at 65% of frame height.
STOP_LINE_Y: int | None = None
STOP_LINE_COLOR         = (0, 0, 255)   # Red line drawn on frame

# ── No-parking zones (list of [x1, y1, x2, y2] pixel rectangles) ─────────────
# Set to [] to disable; define zones after calibrating your camera.
NO_PARKING_ZONES: list[list[int]] = []

# ── Traffic flow direction ────────────────────────────────────────────────────
# "left_to_right" : vehicles naturally move LEFT → RIGHT in the frame
# "right_to_left" : vehicles naturally move RIGHT → LEFT
TRAFFIC_DIRECTION = "left_to_right"

# ── Violation detection thresholds ────────────────────────────────────────────
TRIPLE_RIDING_THRESHOLD     = 2      # persons on one 2-wheeler = violation
STATIONARY_FRAMES_THRESHOLD = 90     # frames without movement = illegal park
STATIONARY_PIXEL_TOLERANCE  = 15     # pixels of movement to consider "moved"
MIN_DETECTION_CONFIDENCE    = 0.40   # ignore boxes below this confidence
WRONG_SIDE_FRAMES_THRESHOLD = 10     # frames moving wrong way before flagging

# ── PM2.5 savings estimation ──────────────────────────────────────────────────
PM25_SAVED_PER_HIGH_VEHICLE = 0.15   # grams per vehicle per cycle

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SECS = 5

# ── Emission weights ──────────────────────────────────────────────────────────
EMISSION_WEIGHTS = {"High": 5, "Medium": 3, "BS-VI": 1, "Clean": 0}
