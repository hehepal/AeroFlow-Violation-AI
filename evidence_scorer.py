"""
evidence_scorer.py  —  AeroFlow Violation AI
Grades each violation frame before it is saved as official evidence.

The core insight: not every detected violation produces usable evidence.
A frame where the plate is unreadable, the vehicle is a distant blur,
or multiple overlapping boxes make the violation ambiguous will NOT
hold up in court — and filing it wastes enforcement resources.

This module scores every candidate frame on three axes before saving:
  1. Plate Readability  (0–40 pts) — OCR confidence + format match
  2. Vehicle Visibility (0–35 pts) — bbox size relative to frame
  3. Detection Clarity  (0–25 pts) — model confidence + box cleanliness

Grades:
  A  (≥ 75)  — Court-ready. Save as primary evidence.
  B  (50–74) — Acceptable. Save as supporting evidence.
  C  (< 50)  — Poor quality. Discard. Flag for manual review.

Only A and B frames are saved to disk. C frames are counted but discarded.
This directly addresses "reducing manual effort" from the theme — officers
no longer wade through hundreds of blurry, unusable images.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2

from violation_detector import ViolationResult
from license_plate_reader import PlateResult


# ── Fine amounts per violation (Karnataka MVA) ────────────────────────────────
FINE_AMOUNTS: dict[str, int] = {
    "Helmet Non-Compliance"  : 1000,
    "Triple Riding"          : 2000,
    "Red-Light Violation"    : 5000,
    "Stop-Line Violation"    : 500,
    "Wrong-Side Driving"     : 5000,
    "Illegal Parking"        : 500,
    "Seatbelt Non-Compliance": 1000,
}


@dataclass
class EvidenceScore:
    total           : int       # 0–100
    grade           : str       # A / B / C
    court_ready     : bool      # True if grade A or B
    plate_score     : int       # 0–40
    visibility_score: int       # 0–35
    clarity_score   : int       # 0–25
    fine_amount     : int       # ₹ applicable fine
    discard_reason  : str = ""  # reason if grade C

    def label(self) -> str:
        return (f"Grade {self.grade}  [{self.total}/100]  "
                f"Fine: ₹{self.fine_amount:,}")


class EvidenceScorer:
    """
    Scores a (frame, violations, plate_results) triple and decides
    whether it is worth saving as evidence.
    """

    # Minimum bbox area ratio to be considered "visible"
    MIN_BBOX_RATIO = 0.005   # bbox must be at least 0.5% of frame area

    def score(
        self,
        frame           : np.ndarray,
        violations      : list[ViolationResult],
        plate_results   : list[PlateResult],
    ) -> EvidenceScore:
        """
        Score the evidence quality of a violation frame.

        Args:
            frame         : BGR frame (enhanced, not raw)
            violations    : list of ViolationResult detected in this frame
            plate_results : list of PlateResult corresponding to violations

        Returns:
            EvidenceScore with grade and save recommendation.
        """
        if not violations:
            return EvidenceScore(0, "C", False, 0, 0, 0, 0,
                                 discard_reason="No violations in frame")

        h, w        = frame.shape[:2]
        frame_area  = max(h * w, 1)

        # ── 1. Plate readability (0–40) ───────────────────────────────────────
        plate_score = 0
        for plate in plate_results:
            if plate and plate.plate_text:
                text = plate.plate_text.replace(" ", "")
                if len(text) >= 8:                     # full Indian plate
                    pts = int(plate.confidence * 40)
                elif len(text) >= 5:                   # partial plate
                    pts = int(plate.confidence * 22)
                else:
                    pts = int(plate.confidence * 8)
                plate_score = max(plate_score, min(40, pts))

        # ── 2. Vehicle visibility (0–35) ──────────────────────────────────────
        vis_score = 0
        for v in violations:
            x1, y1, x2, y2 = v.bbox
            bbox_area  = max((x2 - x1) * (y2 - y1), 1)
            ratio      = bbox_area / frame_area
            # Larger vehicle in frame = clearer evidence
            pts        = min(35, int(ratio / self.MIN_BBOX_RATIO * 5))
            vis_score  = max(vis_score, pts)

        # Check for image blur in violation ROI
        if violations:
            x1, y1, x2, y2 = violations[0].bbox
            roi        = frame[y1:y2, x1:x2]
            if roi.size > 0:
                blur   = cv2.Laplacian(
                    cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var()
                if blur < 40:           # very blurry ROI
                    vis_score = int(vis_score * 0.6)

        # ── 3. Detection clarity (0–25) ───────────────────────────────────────
        # High confidence + few overlapping boxes = unambiguous violation
        max_conf    = max(v.confidence for v in violations)
        conf_pts    = int(max_conf * 22)

        # Penalise if too many simultaneous violations (ambiguous scene)
        overlap_pen = max(0, (len(violations) - 3) * 2)
        clarity_score = max(0, min(25, conf_pts - overlap_pen))

        # ── Total + grade ─────────────────────────────────────────────────────
        total    = plate_score + vis_score + clarity_score

        # ── Minimum grade override for self-evident violations ────────────────
        # Helmet Non-Compliance and Triple Riding are visually self-evident
        # from the image — the violation IS the image. These should always
        # generate a challan if the model confidence is reasonable (≥0.30),
        # regardless of whether the plate is fully readable.
        visual_violation_types = {"Helmet Non-Compliance", "Triple Riding"}
        has_visual_violation   = any(
            v.violation_type in visual_violation_types for v in violations
        )
        max_conf_val = max(v.confidence for v in violations)
        if has_visual_violation and max_conf_val >= 0.30 and total < 50:
            # Boost to minimum Grade B
            total          = 50
            plate_score    = max(plate_score, 10)   # partial plate credit

        if total >= 75:
            grade, court_ready = "A", True
            discard_reason     = ""
        elif total >= 50:
            grade, court_ready = "B", True
            discard_reason     = ""
        else:
            grade, court_ready = "C", False
            reasons = []
            if plate_score  < 15: reasons.append("plate unreadable")
            if vis_score    < 12: reasons.append("vehicle too small/blurry")
            if clarity_score < 8: reasons.append("low detection confidence")
            discard_reason = "; ".join(reasons)

        # ── Fine amount (highest applicable) ─────────────────────────────────
        fine = max(
            (FINE_AMOUNTS.get(v.violation_type, 500) for v in violations),
            default=500,
        )

        return EvidenceScore(
            total            = total,
            grade            = grade,
            court_ready      = court_ready,
            plate_score      = plate_score,
            visibility_score = vis_score,
            clarity_score    = clarity_score,
            fine_amount      = fine,
            discard_reason   = discard_reason,
        )

    def annotate_score(
        self,
        frame : np.ndarray,
        score : EvidenceScore,
    ) -> np.ndarray:
        """Draw the evidence grade badge on the frame."""
        color = {
            "A": (0, 220, 0),
            "B": (0, 165, 255),
            "C": (0, 0, 200),
        }.get(score.grade, (150, 150, 150))

        h, w = frame.shape[:2]
        label = score.label()
        cv2.rectangle(frame, (w - 310, h - 38), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, label, (w - 305, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if not score.court_ready and score.discard_reason:
            cv2.putText(frame, f"DISCARDED: {score.discard_reason}",
                        (w - 305, h - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 255), 1)
        return frame