"""
evidence_generator.py  —  AeroFlow Violation AI
Produces annotated evidence frames and structured violation logs.

Theme 3 Tasks covered:
  - Evidence Generation: annotated images highlighting violations
  - Store violation metadata and timestamps
  - Searchable violation records (CSV log)
"""
from __future__ import annotations

import csv
import datetime
import os

import cv2
import numpy as np

from config import (
    EVIDENCE_FRAMES_DIR,
    EVIDENCE_REPORTS_DIR,
    VIOLATIONS_LOG,
    VIOLATIONS_LOG_COLS,
    STOP_LINE_Y,
)
from violation_detector import ViolationResult

# ── Violation type → overlay colour ──────────────────────────────────────────
VIOLATION_COLORS: dict[str, tuple] = {
    "Helmet Non-Compliance"  : (0,   0,   255),   # Red
    "Triple Riding"          : (0,   140, 255),   # Orange
    "Wrong-Side Driving"     : (255, 0,   255),   # Magenta
    "Stop-Line Violation"    : (255, 255, 0  ),   # Cyan
    "Red-Light Violation"    : (0,   0,   200),   # Dark red
    "Illegal Parking"        : (0,   165, 255),   # Orange
    "Triple Riding (3 persons)": (0,  140, 255),
}
DEFAULT_COLOR = (0, 0, 255)


class EvidenceGenerator:
    """
    Saves annotated evidence frames and appends to violation CSV log.
    Thread-safe for single-threaded use (one writer per process).
    """

    def __init__(self):
        os.makedirs(EVIDENCE_FRAMES_DIR,  exist_ok=True)
        os.makedirs(EVIDENCE_REPORTS_DIR, exist_ok=True)
        self._init_csv()

    # ── Public API ─────────────────────────────────────────────────────────────

    def save(
        self,
        frame      : np.ndarray,
        violations : list[ViolationResult],
        frame_id   : int,
        stop_line_y: int | None = None,
    ) -> list[str]:
        """
        Annotate frame with violation overlays and save to disk.

        Args:
            frame       : BGR frame (preprocessed)
            violations  : list of ViolationResult for this frame
            frame_id    : frame counter
            stop_line_y : Y pixel of stop line to draw

        Returns:
            List of saved evidence file paths.
        """
        if not violations:
            return []

        h, w = frame.shape[:2]
        sl_y = stop_line_y or STOP_LINE_Y or int(h * 0.65)

        annotated = frame.copy()

        # Draw stop line
        cv2.line(annotated, (0, sl_y), (w, sl_y), (0, 0, 255), 2)
        cv2.putText(annotated, "STOP LINE", (8, sl_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw violation boxes and labels
        for v in violations:
            color = self._color_for(v.violation_type)
            x1, y1, x2, y2 = v.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label1 = f"{v.violation_type}"
            label2 = f"{v.vehicle_class} | Conf:{v.confidence:.2f}"
            label3 = f"Plate: {v.plate_text}" if v.plate_text else ""

            # Black background for readability
            for i, lbl in enumerate([l for l in [label1, label2, label3] if l]):
                lx = x1
                ly = max(y1 - 12 - (len([label1, label2, label3]) - 1 - i) * 18, 12)
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(annotated, (lx - 1, ly - th - 2),
                              (lx + tw + 1, ly + 2), (0, 0, 0), -1)
                cv2.putText(annotated, lbl, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

        # Timestamp watermark
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, f"AeroFlow AI  |  {ts_str}  |  Frame {frame_id}",
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Violation summary banner (top)
        banner = f"VIOLATIONS DETECTED: {len(violations)}"
        cv2.rectangle(annotated, (0, 0), (w, 32), (0, 0, 180), -1)
        cv2.putText(annotated, banner, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save frame
        fname = (f"VIO_frame{frame_id:06d}_"
                 f"{datetime.datetime.now().strftime('%H%M%S')}.jpg")
        fpath = os.path.join(EVIDENCE_FRAMES_DIR, fname)
        cv2.imwrite(fpath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Update violation objects with path and log
        saved_paths = []
        for v in violations:
            v.evidence_path = fpath
            self._log_violation(v)
            saved_paths.append(fpath)

        return saved_paths

    def annotate_live(
        self,
        frame      : np.ndarray,
        violations : list[ViolationResult],
        lane_scores: dict[str, float] | None = None,
        stop_line_y: int | None              = None,
    ) -> np.ndarray:
        """
        Lightweight real-time annotation for the live display window.
        Does NOT save to disk — call save() separately for evidence.
        """
        h, w     = frame.shape[:2]
        sl_y     = stop_line_y or STOP_LINE_Y or int(h * 0.65)
        annotated = frame.copy()

        # Stop line
        cv2.line(annotated, (0, sl_y), (w, sl_y), (0, 0, 255), 2)

        for v in violations:
            color = self._color_for(v.violation_type)
            x1, y1, x2, y2 = v.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{v.violation_type}"
            cv2.putText(annotated, lbl, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Lane scores strip (top-left)
        if lane_scores:
            y = 45
            for lane, score in lane_scores.items():
                cv2.putText(annotated, f"{lane}: {score}",
                            (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1)
                y += 18

        # Violation count badge
        vc = len(violations)
        if vc:
            badge_color = (0, 0, 200)
            cv2.rectangle(annotated, (w - 210, 0), (w, 30), badge_color, -1)
            cv2.putText(annotated, f"VIOLATIONS: {vc}",
                        (w - 200, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        return annotated

    # ── CSV logging ───────────────────────────────────────────────────────────

    def _init_csv(self) -> None:
        if not os.path.exists(VIOLATIONS_LOG):
            os.makedirs(os.path.dirname(VIOLATIONS_LOG), exist_ok=True)
            with open(VIOLATIONS_LOG, "w", newline="") as f:
                csv.writer(f).writerow(VIOLATIONS_LOG_COLS)

    def _log_violation(self, v: ViolationResult) -> None:
        with open(VIOLATIONS_LOG, "a", newline="") as f:
            csv.writer(f).writerow(v.to_row())

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _color_for(vtype: str) -> tuple:
        for key, col in VIOLATION_COLORS.items():
            if key.lower() in vtype.lower():
                return col
        return DEFAULT_COLOR
