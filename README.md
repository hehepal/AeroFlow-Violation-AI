# 🚦 AeroFlow Violation AI
### Automated Photo Identification and Classification for Traffic Violations Using Computer Vision
> **Theme 3 Submission** — Scalable AI-based traffic image analysis system

---

## Overview

AeroFlow Violation AI is a real-time computer vision system that automatically detects, classifies, and documents traffic violations from camera feeds. It extends an emission-aware traffic signal control system (AeroFlow AI) with a full violation detection, evidence generation, and analytics layer.

---

## System Architecture — Sense → Think → Act

```
┌──────────────────────────────────────────────────────────────────┐
│  SENSE  —  Perception Layer                                      │
│  Camera / Video → preprocess.py (CLAHE, denoise, sharpen)        │
│                 → YOLOv8 + ByteTrack (vehicle + person tracking) │
├──────────────────────────────────────────────────────────────────┤
│  THINK  —  Violation Analysis                                    │
│  violation_detector.py  →  7 violation types detected            │
│  license_plate_reader.py → EasyOCR for Indian number plates      │
│  classify_emissions.py   → Emission tier per vehicle             │
│  aqi_reader.py           → Live PM2.5 from CPCB API              │
├──────────────────────────────────────────────────────────────────┤
│  ACT  —  Evidence + Control + Monitoring                         │
│  evidence_generator.py → annotated frames + violation CSV        │
│  signal_controller.py  → GREEN/YELLOW/RED state machine          │
│  analytics.py          → stats, trends, searchable records       │
│  evaluate.py           → FPS, Precision, Recall, F1, mAP         │
│  dashboard.py          → 5-tab Streamlit real-time dashboard     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Violations Detected (All 7 from Theme 3)

| # | Violation | Detection Method |
|---|-----------|-----------------|
| 1 | **Helmet Non-Compliance** | Head ROI crop → helmet model / colour heuristic |
| 2 | **Seatbelt Non-Compliance** | Fine-tuned seatbelt model (plug-in ready) |
| 3 | **Triple Riding** | Person count overlapping 2-wheeler bbox |
| 4 | **Wrong-Side Driving** | Centroid trajectory direction vs traffic flow |
| 5 | **Stop-Line Violation** | Vehicle bbox crosses configured stop-line Y |
| 6 | **Red-Light Violation** | Stop-line crossing during RED signal phase |
| 7 | **Illegal Parking** | Vehicle stationary >90 frames in no-parking zone |

---

## Complete File Map

| File | Purpose | Theme 3 Task |
|------|---------|-------------|
| `prototype.py` | **Main entry point** — runs full pipeline | All |
| `preprocess.py` | CLAHE, denoising, sharpening, condition detection | Task 1 |
| `classify_emissions.py` | Vehicle + emission category detection | Task 2 |
| `violation_detector.py` | All 7 violation detectors with ByteTrack | Task 3, 4 |
| `license_plate_reader.py` | EasyOCR Indian plate recognition | Task 5 |
| `evidence_generator.py` | Annotated frames, metadata, CSV log | Task 6 |
| `analytics.py` | Stats, trends, searchable records, reports | Task 7 |
| `evaluate.py` | FPS, Precision, Recall, F1, mAP | Task 8 |
| `dashboard.py` | 5-tab Streamlit dashboard | Visualisation |
| `signal_controller.py` | GREEN→YELLOW→RED state machine | Bonus |
| `aqi_reader.py` | CPCB live API + CSV fallback + cache | Bonus |
| `config.py` | All configuration in one place | — |
| `lane_decision_engine.py` | Simulation mode test harness | — |
| `pm25_estimator.py` | PM2.5 savings estimation | Bonus |

---

## Datasets Used

| Dataset | Size | Used For |
|---------|------|---------|
| **COCO** (via YOLOv8 pretrained weights) | 25 GB / 1.5M images | Vehicle + person detection |
| **CPCB AQI** (data.gov.in) | ~50K rows, live | Ambient PM2.5 per location |
| **Helmet Detection Dataset** (Roboflow) | ~5K images | Helmet compliance model |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Live AQI
```bash
export CPCB_API_KEY="your_key_from_data.gov.in"
```

### 3. (Optional) Helmet model
Download `yolov8n-helmet-detection` from:
https://huggingface.co/keremberke/yolov8n-helmet-detection

Place as `helmet_model.pt` in the project folder.
Without it, a colour-heuristic fallback is used automatically.

### 4. Run prototype
```bash
# Webcam
python prototype.py

# Video file
python prototype.py --source traffic_test.mp4

# Headless (no display window)
python prototype.py --source traffic.mp4 --no-show
```

### 5. Dashboard (separate terminal)
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`

---

## Output Files

| File | Contents |
|------|----------|
| `evidence/violations.csv` | Full violation log: type, plate, timestamp, bbox, frame path |
| `evidence/frames/` | Annotated JPEG evidence frames per violation |
| `evidence/reports/` | Auto-generated text summary reports |
| `lane_logs.csv` | Per-frame lane emission scores, AQI, signal decisions |

---

## Image Preprocessing Pipeline (Task 1)

```
Raw Frame → Condition Detection → CLAHE (contrast)
         → Gamma Correction (low-light) → Denoising (noise)
         → Unsharp Masking (blur) → Enhanced Frame → YOLO
```

Conditions handled: low light, overexposure, motion blur, sensor noise, rain.

---

## License Plate Recognition (Task 5)

- **OCR Engine**: EasyOCR (English, GPU optional)
- **Plate Region**: Contour + aspect-ratio filtering (Indian plates: ~4.5:1)
- **Preprocessing**: Adaptive threshold + 2× upscale for small plates
- **Format**: Regex match for Indian format `DL 3C AB 1234`

---

## Signal Guardrails (Safety)

| Parameter | Value |
|-----------|-------|
| Min green time | 15 s |
| Max green time | 60 s |
| Yellow interval | 3 s (fixed — never adjusted) |
| Full cycle budget | 120 s |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| FPS (YOLOv8n, CPU) | ≥ 15 |
| Precision | ≥ 0.80 |
| Recall | ≥ 0.75 |
| F1-Score | ≥ 0.77 |
| mAP@0.45 | ≥ 0.70 |

---

## Roadmap

- [ ] Seatbelt detection model integration
- [ ] Per-vehicle PDF challan (fine notice) generation
- [ ] ANPR database lookup for repeat offenders
- [ ] Multi-camera support (one ViolationDetector per lane)
- [ ] RL agent replacing heuristic signal controller
- [ ] SUMO digital twin for simulation testing

---

*AeroFlow Violation AI — Delhi-NCR Traffic Enforcement*
*YOLOv8 · ByteTrack · EasyOCR · CPCB AQI · Streamlit*
