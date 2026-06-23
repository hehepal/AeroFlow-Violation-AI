"""
license_plate_reader.py  —  AeroFlow Violation AI
License plate detection and OCR for Indian number plates.

Strategy:
  1. Crop vehicle bounding box from frame
  2. Detect plate region using edge + contour analysis
  3. Run EasyOCR on the cropped plate region
  4. Post-process text to match Indian plate format (e.g. DL 3C AB 1234)

EasyOCR model download: ~200MB on first run (automatic).
Supports Indian number plates in English characters.

Indian plate format examples:
  DL 3C AB 1234    (private vehicle)
  DL 1 PA 0001     (government)
  HR 26 BJ 2317    (Haryana)
  MH 12 AB 1234    (Maharashtra)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PlateResult:
    plate_text      : str
    confidence      : float
    plate_bbox      : tuple | None   # (x1, y1, x2, y2) within vehicle crop
    raw_text        : str = ""


class LicensePlateReader:
    """Lazy-loads EasyOCR on first call to avoid startup cost."""

    # Indian plate format: 2 letters + 1-2 digits + 1-2 letters + 4 digits
    _PLATE_PATTERN = re.compile(
        r"[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,2}\s?\d{4}"
    )

    def __init__(self, gpu: bool = False):
        self._gpu    = gpu
        self._reader = None   # lazy-loaded

    def _load_reader(self):
        if self._reader is None:
            try:
                import easyocr
                print("[LPR] Loading EasyOCR (first run downloads ~200MB)...")
                self._reader = easyocr.Reader(["en"], gpu=self._gpu,
                                              verbose=False)
                print("[LPR] EasyOCR ready.")
            except ImportError:
                print("[LPR] EasyOCR not installed. Run: pip install easyocr")
                self._reader = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def read_plate(
        self,
        frame       : np.ndarray,
        vehicle_bbox: tuple[int, int, int, int],
        expand_px   : int = 8,
    ) -> PlateResult:
        """
        Detect and read the license plate from a vehicle's bounding box.

        Args:
            frame        : full BGR frame
            vehicle_bbox : (x1, y1, x2, y2) of the detected vehicle
            expand_px    : pixels to expand bbox before cropping

        Returns:
            PlateResult with plate text and confidence.
        """
        self._load_reader()
        if self._reader is None:
            return PlateResult("OCR_UNAVAILABLE", 0.0, None)

        h, w  = frame.shape[:2]
        x1, y1, x2, y2 = vehicle_bbox
        x1 = max(0, x1 - expand_px)
        y1 = max(0, y1 - expand_px)
        x2 = min(w, x2 + expand_px)
        y2 = min(h, y2 + expand_px)

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return PlateResult("", 0.0, None)

        # Detect plate region within crop
        plate_crop, plate_bbox = self._detect_plate_region(vehicle_crop)

        # Run OCR on best available crop
        ocr_target = plate_crop if plate_crop is not None else vehicle_crop
        ocr_target = self._preprocess_for_ocr(ocr_target)

        try:
            ocr_results = self._reader.readtext(ocr_target, detail=1)
        except Exception as e:
            print(f"[LPR] OCR error: {e}")
            return PlateResult("", 0.0, plate_bbox)

        return self._parse_ocr_results(ocr_results, plate_bbox)

    # ── Plate region detection ────────────────────────────────────────────────

    def _detect_plate_region(
        self, vehicle_crop: np.ndarray
    ) -> tuple[np.ndarray | None, tuple | None]:
        """
        Locate the rectangular license plate within a vehicle crop.

        Uses edge detection + contour filtering:
          - Plate is a rectangle with aspect ratio ~4.5:1 (Indian plates)
          - Typically in the lower 40% of the vehicle bounding box
        """
        h, w = vehicle_crop.shape[:2]
        # Focus on lower portion where plates usually appear
        lower = vehicle_crop[int(h * 0.5):, :]

        gray    = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blur, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_crop  = None
        best_bbox  = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0:
                continue
            aspect = cw / ch
            # Indian plate: ~4–5.5 aspect ratio; allow generous range
            if not (2.5 <= aspect <= 7.0):
                continue
            # Prefer large, well-shaped rectangles
            rect_score = area * min(aspect / 4.5, 4.5 / aspect)
            if rect_score > best_score:
                best_score = rect_score
                best_bbox  = (x, int(h * 0.5) + y, x + cw, int(h * 0.5) + y + ch)
                best_crop  = vehicle_crop[
                    int(h * 0.5) + y: int(h * 0.5) + y + ch,
                    x: x + cw
                ]

        return best_crop, best_bbox

    # ── OCR preprocessing ─────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
        """
        Sharpen and binarize image for better OCR accuracy on number plates.
        """
        if img is None or img.size == 0:
            return img
        # Upscale small crops for better OCR
        h, w = img.shape[:2]
        if h < 32 or w < 80:
            scale = max(32 / h, 80 / w, 2.0)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)

        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold to handle uneven lighting on plate
        binary   = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # ── Result parsing ────────────────────────────────────────────────────────

    def _parse_ocr_results(
        self,
        ocr_results : list,
        plate_bbox  : tuple | None,
    ) -> PlateResult:
        """
        Combine all OCR text blocks, clean, and try to match Indian plate format.
        """
        if not ocr_results:
            return PlateResult("", 0.0, plate_bbox)

        texts       = []
        confidences = []
        for _, text, conf in ocr_results:
            texts.append(text.strip().upper())
            confidences.append(conf)

        raw_combined = " ".join(texts)
        clean        = re.sub(r"[^A-Z0-9\s]", "", raw_combined)
        avg_conf     = sum(confidences) / len(confidences) if confidences else 0.0

        # Try to match Indian plate pattern
        match = self._PLATE_PATTERN.search(clean)
        if match:
            plate_text = match.group().replace(" ", "")
            # Reformat: DL3CAB1234 → DL 3C AB 1234
            plate_text = self._format_plate(plate_text)
            return PlateResult(plate_text, round(avg_conf, 3), plate_bbox, raw_combined)

        # No pattern match — return raw cleaned text
        candidate = clean.strip()
        return PlateResult(
            candidate if len(candidate) >= 4 else "",
            round(avg_conf * 0.5, 3),   # penalise unformatted
            plate_bbox,
            raw_combined,
        )

    @staticmethod
    def _format_plate(raw: str) -> str:
        """Format raw plate string to standard Indian format: DL 3C AB 1234"""
        raw = raw.replace(" ", "").upper()
        # Typical structure: 2L + 1-2D + 1-2L + 4D
        m = re.match(r"([A-Z]{2})(\d{1,2})([A-Z]{1,2})(\d{4})", raw)
        if m:
            return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"
        return raw
