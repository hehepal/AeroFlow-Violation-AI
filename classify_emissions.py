"""
classify_emissions.py  —  AeroFlow AI
Single source of truth for:
  - EMISSION_PROFILE  : maps YOLOv8 class names → emission categories
  - EMISSION_WEIGHTS  : research-backed pollution score weights per category
  - classify_and_count(): processes one YOLO result → counts + annotated boxes
  - compute_pollution_score(): weighted score with CPCB AQI multiplier

Emission weight rationale (BS standard research):
  ┌──────────┬──────────────────────────────┬──────────┬────────┐
  │ Category │ Vehicle type                 │ PM mg/km │ Weight │
  ├──────────┼──────────────────────────────┼──────────┼────────┤
  │ High     │ Pre-BS-IV diesel trucks/buses│ 100-200  │   5    │
  │ Medium   │ BS-III/IV two-wheelers       │ ~25      │   3    │
  │ BS-VI    │ Post-2020 cars/vans          │ ~4.5     │   1    │
  │ Clean    │ EV / Bicycle                 │ 0        │   0    │
  └──────────┴──────────────────────────────┴──────────┴────────┘
  BS-VI diesel emits ≥70% less NOx and ~80% less PM than older norms.
  Source: MoRTH / ICCT BS-VI policy update, Business Standard 2025.
"""
from ultralytics import YOLO
import cv2

# ── Emission category mapping (YOLOv8 COCO class names) ──────────────────────
EMISSION_PROFILE: dict[str, str] = {
    # High emitters — pre-BS-IV heavy diesel vehicles
    "bus":        "High",
    "truck":      "High",
    "train":      "High",

    # Medium emitters — BS-III/IV two-wheelers (dominant on Indian roads)
    "motorbike":  "Medium",
    "motorcycle": "Medium",
    "scooter":    "Medium",

    # BS-VI compliant (mandatory in India since April 2020)
    "car":        "BS-VI",
    "van":        "BS-VI",

    # Zero tailpipe emission
    "bicycle":    "Clean",

    # Non-vehicle — ignored in scoring
    "person":     "Ignore",
}

# ── Pollution score weights (proportional to real PM2.5 emission factors) ─────
EMISSION_WEIGHTS: dict[str, int] = {
    "High":   5,
    "Medium": 3,
    "BS-VI":  1,
    "Clean":  0,
}

# ── Bounding-box draw colours per category ────────────────────────────────────
BOX_COLORS: dict[str, tuple] = {
    "High":   (0,   0,   255),   # Red
    "Medium": (0,   165, 255),   # Orange
    "BS-VI":  (0,   255, 0),     # Green
    "Clean":  (255, 255, 0),     # Yellow
}


def classify_and_count(
    result, model
) -> tuple[dict[str, int], list[tuple]]:
    """
    Classify all detected vehicles in one YOLO result frame.

    Args:
        result : ultralytics Results object  (i.e. results[0])
        model  : loaded YOLO model (for .names class-id lookup)

    Returns:
        counts (dict)         : {'High': int, 'Medium': int, 'BS-VI': int, 'Clean': int}
        annotated_boxes (list): [(xyxy_array, label_str, bgr_color), ...]
    """
    counts: dict[str, int] = {"High": 0, "Medium": 0, "BS-VI": 0, "Clean": 0}
    annotated_boxes: list[tuple] = []

    if result.boxes is None or len(result.boxes) == 0:
        return counts, annotated_boxes

    for box in result.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id]
        category = EMISSION_PROFILE.get(cls_name, "Unknown")

        if category in ("Ignore", "Unknown"):
            continue

        counts[category] += 1

        xyxy  = box.xyxy[0].cpu().numpy().astype(int)
        label = f"{cls_name} [{category}]"
        color = BOX_COLORS.get(category, (200, 200, 200))
        annotated_boxes.append((xyxy, label, color))

    return counts, annotated_boxes


def compute_pollution_score(
    counts: dict[str, int],
    aqi: float | None = None
) -> float:
    """
    Compute a Pollution Urgency Score for one lane.

    Formula:
        base = Σ (count[cat] × weight[cat])
        Then apply CPCB AQI category multiplier if aqi is provided.

    AQI multipliers (CPCB breakpoints):
        > 300  Severe/Very Poor → ×1.50
        > 200  Poor             → ×1.30
        > 100  Moderate         → ×1.15
        < 50   Good             → ×0.90  (light winds disperse emissions faster)

    Args:
        counts : vehicle emission counts dict from classify_and_count()
        aqi    : ambient PM2.5 AQI value at the lane location (or None)

    Returns:
        float : rounded pollution urgency score
    """
    base = sum(counts.get(cat, 0) * w for cat, w in EMISSION_WEIGHTS.items())

    if aqi is not None:
        if aqi > 300:
            base *= 1.50
        elif aqi > 200:
            base *= 1.30
        elif aqi > 100:
            base *= 1.15
        elif aqi < 50:
            base *= 0.90

    return round(base, 2)


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(0)
    # Use a video file instead:
    # cap = cv2.VideoCapture(r"path\to\traffic_test.mp4")

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        exit(1)

    print("AeroFlow — Emission Classifier  |  Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results           = model(frame)
        annotated         = frame.copy()
        counts, boxes     = classify_and_count(results[0], model)

        for xyxy, label, color in boxes:
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(annotated, label, (xyxy[0], xyxy[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        score = compute_pollution_score(counts)
        print(f"🔴 High:{counts['High']}  🟠 Medium:{counts['Medium']}  "
              f"🟢 BS-VI:{counts['BS-VI']}  🚴 Clean:{counts['Clean']}  "
              f"→ Score: {score}")

        y = 35
        for text, col in [
            (f"Score: {score}",                (0, 255, 255)),
            (f"High (diesel bus/truck): {counts['High']}",   (0,   0, 255)),
            (f"Medium (2-wheeler BS-III): {counts['Medium']}",(0, 165, 255)),
            (f"BS-VI (car/van): {counts['BS-VI']}",          (0, 255,   0)),
            (f"Clean (EV/cycle): {counts['Clean']}",         (255,255,  0)),
        ]:
            size = 1.0 if "Score" in text else 0.65
            cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, size, col, 2)
            y += 38 if "Score" in text else 28

        cv2.imshow("AeroFlow: Emission Classifier", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
