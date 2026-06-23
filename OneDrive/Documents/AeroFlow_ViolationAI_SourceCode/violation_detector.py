"""
violation_detector.py  —  AeroFlow Violation AI
Detects all Theme 3 traffic violations from YOLOv8 tracked detections.

Violations detected:
  1. Helmet non-compliance     (person on 2-wheeler without helmet)
  2. Seatbelt non-compliance   (car driver without seatbelt — needs model)
  3. Triple riding             (>2 persons on single 2-wheeler)
  4. Wrong-side driving        (vehicle moving against traffic flow)
  5. Stop-line violation       (crossing stop line during RED phase)
  6. Red-light violation       (entering intersection on RED)
  7. Illegal parking           (stationary vehicle in no-parking zone)

Uses YOLOv8's built-in ByteTrack (model.track()) for cross-frame identity.
"""
from __future__ import annotations

import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from config import (
    TRAFFIC_DIRECTION,
    STOP_LINE_Y,
    NO_PARKING_ZONES,
    TRIPLE_RIDING_THRESHOLD,
    STATIONARY_FRAMES_THRESHOLD,
    STATIONARY_PIXEL_TOLERANCE,
    WRONG_SIDE_FRAMES_THRESHOLD,
    MIN_DETECTION_CONFIDENCE,
)

# ── Two-wheeler class names (from COCO) ───────────────────────────────────────
TWO_WHEELER_CLASSES  = {"motorcycle", "motorbike", "scooter", "bicycle"}
FOUR_WHEELER_CLASSES = {"car", "truck", "bus", "van"}
ALL_VEHICLE_CLASSES  = TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES


# ── Violation result dataclass ────────────────────────────────────────────────

@dataclass
class ViolationResult:
    violation_id    : str
    timestamp       : str
    frame_id        : int
    track_id        : int
    vehicle_class   : str
    violation_type  : str
    confidence      : float
    bbox            : tuple[int, int, int, int]   # x1, y1, x2, y2
    plate_text      : str  = ""
    plate_confidence: float = 0.0
    evidence_path   : str  = ""

    def to_row(self) -> list:
        x1, y1, x2, y2 = self.bbox
        return [
            self.violation_id, self.timestamp, self.frame_id, self.track_id,
            self.vehicle_class, self.violation_type, round(self.confidence, 3),
            x1, y1, x2, y2,
            self.plate_text, round(self.plate_confidence, 3),
            self.evidence_path,
        ]


# ── Violation detector ────────────────────────────────────────────────────────

class ViolationDetector:
    """
    Stateful detector — call detect_all() once per frame.
    Maintains vehicle history across frames for temporal violations.
    """

    def __init__(self, helmet_model=None):
        """
        Args:
            helmet_model : optional YOLO model fine-tuned for helmet detection.
                           If None, rule-based heuristic is used as fallback.
        """
        self.helmet_model = helmet_model

        # Track: {track_id → deque of (cx, cy) centroid positions}
        self.position_history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=120)
        )
        # Track: {track_id → frames spent stationary}
        self.stationary_counts: dict[int, int] = defaultdict(int)
        # Track: {track_id → consecutive wrong-direction frames}
        self.wrong_side_counts: dict[int, int] = defaultdict(int)
        # Track: which track IDs already had stop-line violation flagged
        self.stop_line_flagged: set[int] = set()
        # Violation counter for unique IDs
        self._violation_counter = 0

    def _new_vid(self) -> str:
        self._violation_counter += 1
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return f"VIO-{ts}-{self._violation_counter:04d}"

    # ── Public entry point ────────────────────────────────────────────────────

    def detect_all(
        self,
        frame       : np.ndarray,
        results,                   # ultralytics Results object with .boxes
        frame_id    : int,
        signal_state: str = "RED", # "RED" | "GREEN" | "YELLOW"
        stop_line_y : int | None = None,
    ) -> list[ViolationResult]:
        """
        Run all violation detectors on one frame.

        Args:
            frame        : preprocessed BGR frame
            results      : ultralytics Results from model.track()
            frame_id     : monotonically increasing frame counter
            signal_state : current signal phase for stop-line/red-light checks
            stop_line_y  : Y pixel of stop line (auto-computed if None)

        Returns:
            List of ViolationResult objects found in this frame.
        """
        h, w = frame.shape[:2]
        if stop_line_y is None:
            stop_line_y = STOP_LINE_Y if STOP_LINE_Y else int(h * 0.65)

        violations: list[ViolationResult] = []

        if results.boxes is None or len(results.boxes) == 0:
            return violations

        boxes      = results.boxes
        model_names = results.names  # id → class name

        # ── Parse all detections ──────────────────────────────────────────────
        detections = []
        for box in boxes:
            conf = float(box.conf[0])
            if conf < MIN_DETECTION_CONFIDENCE:
                continue
            cls_id     = int(box.cls[0])
            cls_name   = model_names[cls_id]
            xyxy       = box.xyxy[0].cpu().numpy().astype(int)
            track_id   = int(box.id[0]) if box.id is not None else -1
            cx         = int((xyxy[0] + xyxy[2]) / 2)
            cy         = int((xyxy[1] + xyxy[3]) / 2)

            detections.append({
                "cls": cls_name, "conf": conf,
                "bbox": tuple(xyxy),
                "track_id": track_id,
                "cx": cx, "cy": cy,
            })

            if track_id >= 0:
                self.position_history[track_id].append((cx, cy))

        # ── Update stationary counters ────────────────────────────────────────
        self._update_stationary(detections)

        # ── Run each detector ─────────────────────────────────────────────────
        violations += self._detect_triple_riding(detections, frame_id)
        violations += self._detect_helmet(frame, detections, frame_id)
        violations += self._detect_wrong_side(detections, frame_id)
        violations += self._detect_stop_line(
            detections, frame_id, signal_state, stop_line_y
        )
        violations += self._detect_illegal_parking(detections, frame_id, frame)

        return violations

    # ── Detector 1: Triple riding ─────────────────────────────────────────────

    def _detect_triple_riding(
        self, detections: list, frame_id: int
    ) -> list[ViolationResult]:
        results = []
        two_wheelers = [d for d in detections if d["cls"] in TWO_WHEELER_CLASSES]
        persons      = [d for d in detections if d["cls"] == "person"]

        for bike in two_wheelers:
            bx1, by1, bx2, by2 = bike["bbox"]
            # Expand bike bbox slightly to capture riders
            bx1e = max(0, bx1 - 20)
            by1e = max(0, by1 - 60)   # extend upward for heads
            bx2e = bx2 + 20
            by2e = by2 + 10

            riders = [
                p for p in persons
                if self._bbox_overlap_ratio(
                    p["bbox"], (bx1e, by1e, bx2e, by2e)
                ) > 0.25
            ]

            if len(riders) > TRIPLE_RIDING_THRESHOLD:
                results.append(ViolationResult(
                    violation_id   = self._new_vid(),
                    timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_id       = frame_id,
                    track_id       = bike["track_id"],
                    vehicle_class  = bike["cls"],
                    violation_type = f"Triple Riding ({len(riders)} persons)",
                    confidence     = round(bike["conf"], 3),
                    bbox           = bike["bbox"],
                ))
        return results

    # ── Detector 2: Helmet non-compliance ─────────────────────────────────────

    def _detect_helmet(
        self, frame: np.ndarray, detections: list, frame_id: int
    ) -> list[ViolationResult]:
        results      = []
        two_wheelers = [d for d in detections if d["cls"] in TWO_WHEELER_CLASSES]
        persons      = [d for d in detections if d["cls"] == "person"]

        for bike in two_wheelers:
            bx1, by1, bx2, by2 = bike["bbox"]
            bx1e = max(0, bx1 - 15)
            by1e = max(0, by1 - 80)
            bx2e = bx2 + 15
            by2e = by2 + 5

            riders = [
                p for p in persons
                if self._bbox_overlap_ratio(
                    p["bbox"], (bx1e, by1e, bx2e, by2e)
                ) > 0.2
            ]

            for rider in riders:
                rx1, ry1, rx2, ry2 = rider["bbox"]
                rider_h            = ry2 - ry1
                rider_w            = rx2 - rx1
                # Head region = top 35% of person bbox (increased from 28%)
                # Also expand horizontally by 10% each side for partial detections
                head_y2  = ry1 + int(rider_h * 0.35)
                hx1      = max(0, rx1 - int(rider_w * 0.1))
                hx2      = min(frame.shape[1], rx2 + int(rider_w * 0.1))
                head_roi = frame[ry1:head_y2, hx1:hx2]

                # Upscale very small head ROIs so helmet model can analyse them
                if head_roi.size > 0:
                    h_roi_h, h_roi_w = head_roi.shape[:2]
                    if h_roi_h < 40 or h_roi_w < 40:
                        scale    = max(40 / max(h_roi_h, 1), 40 / max(h_roi_w, 1), 2.0)
                        head_roi = cv2.resize(head_roi,
                                              (int(h_roi_w * scale), int(h_roi_h * scale)),
                                              interpolation=cv2.INTER_CUBIC)

                has_helmet = self._classify_helmet(head_roi)

                if not has_helmet:
                    # Use the BIKE bbox (not rider bbox) so the evidence
                    # scorer and plate reader can find the number plate.
                    # The bike bbox contains both the vehicle and the rider.
                    bx1_full = max(0, bx1 - 10)
                    by1_full = max(0, by1 - 40)   # include head area
                    results.append(ViolationResult(
                        violation_id   = self._new_vid(),
                        timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        frame_id       = frame_id,
                        track_id       = bike["track_id"],
                        vehicle_class  = bike["cls"],
                        violation_type = "Helmet Non-Compliance",
                        confidence     = round(rider["conf"] * 0.85, 3),
                        bbox           = (bx1_full, by1_full, bx2, by2),
                    ))
        return results

    def _classify_helmet(self, head_roi: np.ndarray) -> bool:
        """
        Returns True if helmet detected, False if no helmet (violation).

        Priority:
          1. Fine-tuned helmet YOLO model (if loaded)
          2. Colour/shape heuristic fallback
        """
        if head_roi is None or head_roi.size == 0:
            return True  # can't determine → assume compliant (safe default)

        if self.helmet_model is not None:
            res = self.helmet_model(head_roi, verbose=False)
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                # Pick the highest-confidence detection
                best_conf = 0.0
                best_cls  = ""
                for box in res[0].boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        best_cls  = res[0].names[int(box.cls[0])].lower()

                # Lower threshold to 0.30 so marginal detections aren't missed
                if best_conf >= 0.30:
                    if "without" in best_cls or "no_helmet" in best_cls:
                        return False   # violation — no helmet
                    if "with" in best_cls or ("helmet" in best_cls and "without" not in best_cls):
                        return True    # compliant — helmet present

            # Model detected nothing → fall through to heuristic below
            # (do NOT default to True/compliant here — let heuristic decide)

        # ── Colour heuristic fallback ─────────────────────────────────────────
        # Helmets tend to have a large single-colour region in the head ROI.
        # A bare head shows skin tones (HSV hue 0-25, sat 40-170).
        # A helmet shows high-saturation non-skin colours OR very dark regions.
        if head_roi.shape[0] < 8 or head_roi.shape[1] < 8:
            return True

        hsv       = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        # Skin colour mask (HSV)
        lower_skin = np.array([0,  40,  60], dtype=np.uint8)
        upper_skin = np.array([25, 170, 255], dtype=np.uint8)
        skin_mask  = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = np.count_nonzero(skin_mask) / max(skin_mask.size, 1)

        # Large skin fraction in head region = likely no helmet
        return skin_ratio < 0.45

    # ── Detector 3: Wrong-side driving ────────────────────────────────────────

    def _detect_wrong_side(
        self, detections: list, frame_id: int
    ) -> list[ViolationResult]:
        results  = []
        vehicles = [d for d in detections
                    if d["cls"] in ALL_VEHICLE_CLASSES and d["track_id"] >= 0]

        for v in vehicles:
            tid   = v["track_id"]
            hist  = self.position_history[tid]
            if len(hist) < 6:
                continue

            # Average dx over last 6 positions
            positions = list(hist)[-6:]
            dxs       = [positions[i+1][0] - positions[i][0]
                          for i in range(len(positions) - 1)]
            avg_dx    = sum(dxs) / len(dxs)

            wrong = False
            if TRAFFIC_DIRECTION == "left_to_right" and avg_dx < -4:
                wrong = True
            elif TRAFFIC_DIRECTION == "right_to_left" and avg_dx > 4:
                wrong = True

            if wrong:
                self.wrong_side_counts[tid] += 1
            else:
                self.wrong_side_counts[tid] = max(
                    0, self.wrong_side_counts[tid] - 1
                )

            if self.wrong_side_counts[tid] >= WRONG_SIDE_FRAMES_THRESHOLD:
                results.append(ViolationResult(
                    violation_id   = self._new_vid(),
                    timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_id       = frame_id,
                    track_id       = tid,
                    vehicle_class  = v["cls"],
                    violation_type = "Wrong-Side Driving",
                    confidence     = round(v["conf"], 3),
                    bbox           = v["bbox"],
                ))
                self.wrong_side_counts[tid] = 0   # reset after flagging

        return results

    # ── Detector 4 & 5: Stop-line / Red-light violation ───────────────────────

    def _detect_stop_line(
        self,
        detections  : list,
        frame_id    : int,
        signal_state: str,
        stop_line_y : int,
    ) -> list[ViolationResult]:
        if signal_state not in ("RED", "YELLOW"):
            return []

        results  = []
        vehicles = [d for d in detections if d["cls"] in ALL_VEHICLE_CLASSES]

        for v in vehicles:
            tid          = v["track_id"]
            _, _, _, vy2 = v["bbox"]   # bottom of vehicle box
            # Vehicle has crossed the stop line if its bottom exceeds stop_line_y
            if vy2 > stop_line_y and tid not in self.stop_line_flagged:
                vtype = ("Red-Light Violation"
                         if signal_state == "RED"
                         else "Stop-Line Violation")
                results.append(ViolationResult(
                    violation_id   = self._new_vid(),
                    timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    frame_id       = frame_id,
                    track_id       = tid,
                    vehicle_class  = v["cls"],
                    violation_type = vtype,
                    confidence     = round(v["conf"], 3),
                    bbox           = v["bbox"],
                ))
                if tid >= 0:
                    self.stop_line_flagged.add(tid)

        # Clear flagged IDs when signal goes GREEN (new cycle)
        if signal_state == "GREEN":
            self.stop_line_flagged.clear()

        return results

    # ── Detector 6: Illegal parking ───────────────────────────────────────────

    def _detect_illegal_parking(
        self, detections: list, frame_id: int, frame: np.ndarray
    ) -> list[ViolationResult]:
        results  = []
        vehicles = [d for d in detections
                    if d["cls"] in ALL_VEHICLE_CLASSES and d["track_id"] >= 0]

        for v in vehicles:
            tid = v["track_id"]
            if self.stationary_counts[tid] >= STATIONARY_FRAMES_THRESHOLD:
                # Check if vehicle is in a no-parking zone
                in_zone = self._in_no_parking_zone(v["bbox"])
                if in_zone or len(NO_PARKING_ZONES) == 0:
                    results.append(ViolationResult(
                        violation_id   = self._new_vid(),
                        timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        frame_id       = frame_id,
                        track_id       = tid,
                        vehicle_class  = v["cls"],
                        violation_type = "Illegal Parking",
                        confidence     = round(v["conf"] * 0.9, 3),
                        bbox           = v["bbox"],
                    ))
                    self.stationary_counts[tid] = 0   # reset after flagging

        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_stationary(self, detections: list) -> None:
        for d in detections:
            tid = d["track_id"]
            if tid < 0:
                continue
            hist = self.position_history[tid]
            if len(hist) < 2:
                continue
            prev_cx, prev_cy = hist[-2]
            curr_cx, curr_cy = hist[-1]
            moved = abs(curr_cx - prev_cx) + abs(curr_cy - prev_cy)
            if moved < STATIONARY_PIXEL_TOLERANCE:
                self.stationary_counts[tid] += 1
            else:
                self.stationary_counts[tid] = 0

    def _bbox_overlap_ratio(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Return intersection area / area of box_a (rider containment check)."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter  = (ix2 - ix1) * (iy2 - iy1)
        area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
        return inter / area_a

    def _in_no_parking_zone(self, bbox: tuple) -> bool:
        if not NO_PARKING_ZONES:
            return True   # treat entire frame as no-parking if zones not defined
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        for zx1, zy1, zx2, zy2 in NO_PARKING_ZONES:
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return True
        return False