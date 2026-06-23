"""
prototype.py  —  AeroFlow Violation AI
Main entry point — complete Camera-to-Challan pipeline.

SENSE → THINK → ACT:
  Sense  : Camera/video → preprocess.py → YOLOv8 ByteTrack
  Think  : ViolationDetector → EvidenceScorer → LicensePlateReader
  Act    : ChallanGenerator + EvidenceGenerator + HotspotEngine + Dashboard

Usage:
    python prototype.py                         # webcam
    python prototype.py --source traffic.mp4    # video file
    python prototype.py --no-show               # headless mode
"""
import argparse
import os
import sys
import time

import cv2
from ultralytics import YOLO

import config
from preprocess           import enhance_frame, draw_condition_overlay, detect_conditions
from classify_emissions   import classify_and_count, compute_pollution_score
from violation_detector   import ViolationDetector
from license_plate_reader import LicensePlateReader
from evidence_scorer      import EvidenceScorer
from evidence_generator   import EvidenceGenerator
from challan_generator    import ChallanGenerator
from hotspot_engine       import HotspotEngine
from analytics            import ViolationAnalytics
from evaluate             import PerformanceEvaluator
from aqi_reader           import get_aqi
from signal_controller    import SignalController


# ── Boot helpers ──────────────────────────────────────────────────────────────

def load_helmet_model():
    path = config.HELMET_MODEL_PATH
    if os.path.exists(path):
        try:
            m = YOLO(path)
            print(f"[BOOT] Helmet model loaded: {path}")
            return m
        except Exception as e:
            print(f"[BOOT] Helmet model failed ({e}) — heuristic fallback")
    else:
        print(f"[BOOT] helmet_model.pt not found — heuristic fallback active")
    return None


def prefetch_aqi() -> dict:
    print("[BOOT] Fetching AQI...")
    cache = {lane: get_aqi(loc, city) for lane, (loc, city) in config.LANES.items()}
    print(f"[BOOT] AQI: {cache}")
    return cache


# ── HUD drawing ───────────────────────────────────────────────────────────────

def draw_vehicle_boxes(frame, results, model):
    from classify_emissions import BOX_COLORS, EMISSION_PROFILE
    if results.boxes is None:
        return frame
    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id]
        category = EMISSION_PROFILE.get(cls_name, "Unknown")
        if category in ("Ignore", "Unknown"):
            continue
        xyxy  = box.xyxy[0].cpu().numpy().astype(int)
        color = BOX_COLORS.get(category, (200, 200, 200))
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 1)
        cv2.putText(frame, f"{cls_name}[{category}]",
                    (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    return frame


def draw_hud(frame, frame_id, fps, lane_scores, priority_lane,
             total_violations, total_challans, signal_state, stop_line_y):
    h, w = frame.shape[:2]
    # Stop line
    cv2.line(frame, (0, stop_line_y), (w, stop_line_y), (0, 0, 255), 2)
    cv2.putText(frame, "STOP", (4, stop_line_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # Signal dot
    sig_col = {"GREEN": (0,220,0), "RED": (0,0,220), "YELLOW": (0,200,200)}
    cv2.circle(frame, (w - 28, 28), 16, sig_col.get(signal_state,(150,150,150)), -1)
    cv2.putText(frame, signal_state, (w - 75, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                sig_col.get(signal_state,(150,150,150)), 1)
    # FPS
    cv2.putText(frame, f"FPS:{fps}  Frame:{frame_id}",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)
    # Lane scores
    y = 16
    for lane, score in lane_scores.items():
        marker = "★" if lane == priority_lane else " "
        col    = (0,255,0) if lane == priority_lane else (0,220,220)
        cv2.putText(frame, f"{marker}{lane}:{score}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        y += 15
    # Violation + challan counter
    if total_violations:
        cv2.rectangle(frame, (0, h-34), (240, h), (0,0,140), -1)
        cv2.putText(frame,
                    f"VIOLATIONS:{total_violations}  CHALLANS:{total_challans}",
                    (4, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
    return frame


# ── Main run loop ─────────────────────────────────────────────────────────────

def run(source=0, show=True):
    print("\n" + "="*60)
    print("  AeroFlow Violation AI  — Camera to Challan")
    print("="*60)

    os.makedirs(config.EVIDENCE_FRAMES_DIR,  exist_ok=True)
    os.makedirs(config.EVIDENCE_REPORTS_DIR, exist_ok=True)

    # Boot all modules
    model          = YOLO(config.YOLO_MODEL_PATH)
    helmet_model   = load_helmet_model()
    aqi_cache      = prefetch_aqi()

    plate_reader   = LicensePlateReader(gpu=False)
    scorer         = EvidenceScorer()
    evidence_gen   = EvidenceGenerator()
    challan_gen    = ChallanGenerator()
    hotspot        = HotspotEngine()
    violation_det  = ViolationDetector(helmet_model=helmet_model)
    evaluator      = PerformanceEvaluator()
    sig_ctrl       = SignalController(verbose=False)
    sig_ctrl.start()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Determine stop line
    ret0, frame0 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    h0           = frame0.shape[0] if ret0 else 480
    stop_line_y  = config.STOP_LINE_Y or int(h0 * 0.65)

    print(f"[BOOT] Source       : {source}")
    print(f"[BOOT] Stop line Y  : {stop_line_y}px")
    print(f"[BOOT] Press Q=quit  H=hotspot report  R=text report\n")

    frame_id       = 0
    total_vio      = 0
    total_challans = 0
    last_report_t  = time.time()
    HOTSPOT_EVERY  = 60   # regenerate hotspot map every 60 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── SENSE ─────────────────────────────────────────────────────────────
        cond     = detect_conditions(frame)
        enhanced = enhance_frame(frame)

        results  = model.track(
            enhanced, persist=True,
            tracker="bytetrack.yaml", verbose=False
        )
        result   = results[0]

        # ── THINK — Emission scoring ───────────────────────────────────────────
        q           = w // 4
        lane_scores = {}
        lane_counts = {}
        for i, (lane_name, (loc, city)) in enumerate(config.LANES.items()):
            lane_frame  = enhanced[:, i*q:(i+1)*q]
            lane_res    = model(lane_frame, verbose=False)
            counts, _   = classify_and_count(lane_res[0], model)
            aqi         = aqi_cache.get(lane_name)
            score       = compute_pollution_score(counts, aqi)
            lane_scores[lane_name] = score
            lane_counts[lane_name] = counts

        sig_ctrl.update_scores(lane_scores)
        priority_lane = min(lane_scores, key=lane_scores.get)

        sig_status    = sig_ctrl.get_status()
        # Default to GREEN so vehicles near stop line aren't falsely
        # flagged as red-light violations before the signal controller
        # has completed its first cycle.
        active_phase  = next(
            (v["phase"] for v in sig_status.values() if v["phase"] == "RED"),
            "GREEN"
        )

        # ── THINK — Violation detection ────────────────────────────────────────
        violations = violation_det.detect_all(
            enhanced, result, frame_id,
            signal_state=active_phase,
            stop_line_y=stop_line_y,
        )

        # ── THINK — Plate reading (only for violations) ────────────────────────
        plate_results = []
        for v in violations:
            if v.vehicle_class != "person":
                plate = plate_reader.read_plate(enhanced, v.bbox)
                v.plate_text       = plate.plate_text
                v.plate_confidence = plate.confidence
                plate_results.append(plate)
            else:
                plate_results.append(None)

        # ── ACT — Evidence scoring ─────────────────────────────────────────────
        if violations:
            ev_score = scorer.score(enhanced, violations, 
                                    [p for p in plate_results if p])
            enhanced = scorer.annotate_score(enhanced.copy(), ev_score)

            if ev_score.court_ready:
                # Save evidence frame
                saved_paths = evidence_gen.save(
                    enhanced, violations, frame_id, stop_line_y
                )
                total_vio += len(violations)

                # Generate challan for highest-priority violation
                if saved_paths:
                    primary_v = max(violations,
                                    key=lambda v: v.confidence)
                    challan = challan_gen.generate(
                        violation      = primary_v,
                        score          = ev_score,
                        evidence_frame = saved_paths[0],
                        intersection   = "Bengaluru Traffic Intersection",
                    )
                    if challan:
                        total_challans += 1
                        print(f"  📄 Challan {challan.challan_id} | "
                              f"₹{challan.fine_amount:,} | Grade {ev_score.grade}")
            else:
                print(f"  ⚠️  Frame {frame_id} discarded: {ev_score.discard_reason}")

        # ── ACT — Annotate + display ───────────────────────────────────────────
        annotated = draw_vehicle_boxes(enhanced.copy(), result, model)
        annotated = evidence_gen.annotate_live(
            annotated, violations, lane_scores, stop_line_y
        )
        annotated = draw_hud(
            annotated, frame_id,
            evaluator.fps_tracker.fps,
            lane_scores, priority_lane,
            total_vio, total_challans,
            active_phase, stop_line_y,
        )
        annotated = draw_condition_overlay(annotated, cond)

        evaluator.record_frame(violations)

        # ── Periodic hotspot refresh ───────────────────────────────────────────
        if time.time() - last_report_t >= HOTSPOT_EVERY:
            hotspot.generate_all()
            last_report_t = time.time()

        if show:
            cv2.imshow("AeroFlow Violation AI  |  Q=quit", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                hotspot.generate_all()
            elif key == ord("r"):
                from analytics import ViolationAnalytics
                print(ViolationAnalytics().generate_text_report(save=True))

        frame_id += 1

    # ── Shutdown ───────────────────────────────────────────────────────────────
    sig_ctrl.stop()
    cap.release()
    if show:
        cv2.destroyAllWindows()

    print("\n[DONE] Generating final reports...")
    hotspot.generate_all()
    evaluator.print_report()
    from analytics import ViolationAnalytics
    print(ViolationAnalytics().generate_text_report(save=True))
    print(f"\n  Challans generated : {total_challans}")
    print(f"  Violations logged  : {total_vio}")
    print(f"  Evidence folder    : {config.EVIDENCE_FRAMES_DIR}")
    print(f"  Challans folder    : evidence/challans/")
    print(f"  Hotspot map        : {config.EVIDENCE_REPORTS_DIR}/hotspot_map.html")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroFlow Violation AI")
    parser.add_argument("--source", default=0,
                        help="Webcam index or video file path. Default: 0")
    parser.add_argument("--show",    dest="show", action="store_true",  default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    args   = parser.parse_args()
    src    = args.source
    try:
        src = int(src)
    except (ValueError, TypeError):
        pass
    run(source=src, show=args.show)