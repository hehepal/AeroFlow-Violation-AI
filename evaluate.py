"""
evaluate.py  —  AeroFlow Violation AI
Performance evaluation module.

Theme 3 Task 8 — Performance Evaluation:
  - Accuracy, Precision, Recall, F1-score per violation class
  - mAP (mean Average Precision) for object detection
  - FPS (frames per second) — computational efficiency
  - Confusion matrix across violation types

Two modes:
  A) Runtime mode  — track FPS + confidence distribution while prototype runs
  B) Offline mode  — given ground-truth labels, compute full detection metrics
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


# ── Dataclass for one detection's PR record ───────────────────────────────────

@dataclass
class DetectionRecord:
    """Stores one detection for offline evaluation."""
    predicted_class : str
    true_class      : str    # "" if false positive
    confidence      : float
    iou             : float  # IoU with matched ground-truth box (0 if FP)


# ── Runtime FPS tracker ────────────────────────────────────────────────────────

class FPSTracker:
    """Rolling FPS counter using a sliding window of frame timestamps."""

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Call once per processed frame."""
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return round((len(self._times) - 1) / max(elapsed, 1e-6), 1)

    @property
    def avg_ms(self) -> float:
        """Average milliseconds per frame."""
        fps = self.fps
        return round(1000 / fps, 1) if fps > 0 else 0.0


# ── Per-class metrics ─────────────────────────────────────────────────────────

@dataclass
class ClassMetrics:
    label           : str
    tp              : int   = 0
    fp              : int   = 0
    fn              : int   = 0
    confidence_sum  : float = 0.0
    confidence_count: int   = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def avg_confidence(self) -> float:
        return (self.confidence_sum / self.confidence_count
                if self.confidence_count else 0.0)


# ── Main evaluator ────────────────────────────────────────────────────────────

class PerformanceEvaluator:
    """
    Tracks runtime performance AND computes offline detection metrics.

    Runtime usage (called every frame by prototype.py):
        evaluator = PerformanceEvaluator()
        evaluator.fps_tracker.tick()
        evaluator.record_frame_violations(violations)

    Offline usage (when ground-truth labels are available):
        evaluator.add_detection(DetectionRecord(...))
        report = evaluator.compute_metrics()
    """

    IOU_THRESHOLD = 0.45   # IoU threshold for TP classification

    VIOLATION_CLASSES = [
        "Helmet Non-Compliance",
        "Triple Riding",
        "Wrong-Side Driving",
        "Stop-Line Violation",
        "Red-Light Violation",
        "Illegal Parking",
    ]

    def __init__(self):
        self.fps_tracker = FPSTracker(window=60)

        # Runtime stats
        self._frame_count        = 0
        self._total_violations   = 0
        self._violations_per_type: dict[str, int] = defaultdict(int)
        self._confidence_scores  : list[float]    = []

        # Offline detection records
        self._records: list[DetectionRecord] = []

        # Per-class metrics for offline eval
        self._class_metrics: dict[str, ClassMetrics] = {
            cls: ClassMetrics(label=cls) for cls in self.VIOLATION_CLASSES
        }

    # ── Runtime tracking ──────────────────────────────────────────────────────

    def record_frame(self, violations: list) -> None:
        """
        Call once per frame with the list of ViolationResult objects.
        Updates FPS and running violation counts.
        """
        self.fps_tracker.tick()
        self._frame_count += 1
        self._total_violations += len(violations)
        for v in violations:
            self._violations_per_type[v.violation_type] += 1
            self._confidence_scores.append(v.confidence)

    def runtime_summary(self) -> dict:
        """Fast summary dict for live display overlay."""
        avg_conf = (sum(self._confidence_scores) / len(self._confidence_scores)
                    if self._confidence_scores else 0.0)
        return {
            "fps"              : self.fps_tracker.fps,
            "avg_ms_per_frame" : self.fps_tracker.avg_ms,
            "frames_processed" : self._frame_count,
            "total_violations" : self._total_violations,
            "violations_per_type": dict(self._violations_per_type),
            "avg_confidence"   : round(avg_conf, 3),
        }

    # ── Offline evaluation ────────────────────────────────────────────────────

    def add_detection(self, record: DetectionRecord) -> None:
        """Add one detection result for offline metric computation."""
        self._records.append(record)
        cls = record.predicted_class
        if cls not in self._class_metrics:
            self._class_metrics[cls] = ClassMetrics(label=cls)
        m = self._class_metrics[cls]
        m.confidence_sum   += record.confidence
        m.confidence_count += 1
        if record.true_class == cls and record.iou >= self.IOU_THRESHOLD:
            m.tp += 1
        else:
            m.fp += 1
            if record.true_class:
                true_m = self._class_metrics.get(record.true_class)
                if true_m:
                    true_m.fn += 1

    def compute_metrics(self) -> dict:
        """
        Compute full detection metrics from added records.

        Returns:
            {
              "per_class": {class_name: {precision, recall, f1, avg_confidence}},
              "macro_precision": float,
              "macro_recall": float,
              "macro_f1": float,
              "mAP": float,
              "total_detections": int,
            }
        """
        per_class = {}
        precisions, recalls, f1s = [], [], []

        for cls, m in self._class_metrics.items():
            if m.tp + m.fp + m.fn == 0:
                continue
            per_class[cls] = {
                "precision"       : round(m.precision, 4),
                "recall"          : round(m.recall,    4),
                "f1"              : round(m.f1,        4),
                "avg_confidence"  : round(m.avg_confidence, 4),
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
            }
            precisions.append(m.precision)
            recalls.append(m.recall)
            f1s.append(m.f1)

        n = len(precisions)
        macro_p   = sum(precisions) / n if n else 0.0
        macro_r   = sum(recalls)    / n if n else 0.0
        macro_f1  = sum(f1s)        / n if n else 0.0
        map_score = self._compute_map()

        return {
            "per_class"        : per_class,
            "macro_precision"  : round(macro_p,   4),
            "macro_recall"     : round(macro_r,   4),
            "macro_f1"         : round(macro_f1,  4),
            "mAP"              : round(map_score, 4),
            "total_detections" : len(self._records),
            "iou_threshold"    : self.IOU_THRESHOLD,
        }

    def _compute_map(self) -> float:
        """
        Compute mAP@IOU_THRESHOLD across all classes.
        Uses 11-point interpolated AP approximation.
        """
        if not self._records:
            return 0.0

        aps = []
        for cls, m in self._class_metrics.items():
            if m.tp + m.fp == 0:
                continue
            # Sort records for this class by confidence descending
            cls_recs = sorted(
                [r for r in self._records if r.predicted_class == cls],
                key=lambda r: r.confidence,
                reverse=True,
            )
            if not cls_recs:
                continue

            cum_tp, cum_fp = 0, 0
            prec_list, rec_list = [], []
            total_gt = m.tp + m.fn

            for rec in cls_recs:
                if (rec.true_class == cls and
                        rec.iou >= self.IOU_THRESHOLD):
                    cum_tp += 1
                else:
                    cum_fp += 1
                p = cum_tp / (cum_tp + cum_fp)
                r = cum_tp / total_gt if total_gt else 0.0
                prec_list.append(p)
                rec_list.append(r)

            # 11-point interpolation
            ap = 0.0
            for t in np.linspace(0, 1, 11):
                p_vals = [p for p, r in zip(prec_list, rec_list) if r >= t]
                ap += max(p_vals) / 11 if p_vals else 0.0
            aps.append(ap)

        return sum(aps) / len(aps) if aps else 0.0

    def print_report(self) -> None:
        """Print formatted performance report to terminal."""
        rs  = self.runtime_summary()
        met = self.compute_metrics() if self._records else None

        print("\n" + "=" * 60)
        print("  AEROFLOW AI — PERFORMANCE REPORT")
        print("=" * 60)
        print(f"  FPS                  : {rs['fps']}")
        print(f"  Avg ms / frame       : {rs['avg_ms_per_frame']}")
        print(f"  Frames processed     : {rs['frames_processed']}")
        print(f"  Total violations     : {rs['total_violations']}")
        print(f"  Avg detection conf   : {rs['avg_confidence']}")

        if met:
            print(f"\n  mAP@{self.IOU_THRESHOLD}            : {met['mAP']}")
            print(f"  Macro Precision      : {met['macro_precision']}")
            print(f"  Macro Recall         : {met['macro_recall']}")
            print(f"  Macro F1             : {met['macro_f1']}")
            print("\n  PER-CLASS METRICS:")
            for cls, m in met["per_class"].items():
                print(f"    {cls:<35s} "
                      f"P={m['precision']:.3f}  "
                      f"R={m['recall']:.3f}  "
                      f"F1={m['f1']:.3f}")
        print("=" * 60 + "\n")
