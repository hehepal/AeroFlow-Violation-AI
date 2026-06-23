"""
preprocess.py  —  AeroFlow Violation AI
Image enhancement pipeline before YOLO inference.

Tasks (Theme 3 Task 1):
  - Enhance image quality and normalize inputs
  - Handle low light, rain, shadows, and motion blur
  - CLAHE contrast enhancement on luminance channel
  - Non-local means denoising
  - Unsharp masking for edge sharpening

All functions are stateless and accept / return BGR numpy arrays.
"""
import cv2
import numpy as np


# ── Condition detection ───────────────────────────────────────────────────────

def detect_conditions(frame: np.ndarray) -> dict[str, bool]:
    """
    Detect challenging imaging conditions in the frame.
    Returns a dict of flags used to decide which enhancements to apply.
    """
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_lum  = float(np.mean(gray))
    blur_score= float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "low_light"   : mean_lum < 60,
        "overexposed" : mean_lum > 210,
        "blurry"      : blur_score < 80,
        "noisy"       : bool(np.std(gray) < 25),
    }


# ── Enhancement functions ─────────────────────────────────────────────────────

def apply_clahe(frame: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) on L channel.
    Dramatically improves visibility in low-light and shadow conditions.
    """
    lab      = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_eq     = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def denoise(frame: np.ndarray, strength: int = 7) -> np.ndarray:
    """
    Fast Non-Local Means denoising — reduces noise while preserving edges.
    Used when frame appears grainy (e.g. night CCTV, heavy rain).
    strength: filter intensity (higher = more denoising, more blur)
    """
    return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


def sharpen(frame: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """
    Unsharp masking — enhances edges for better plate and badge detection.
    amount: sharpening strength (0–1 recommended)
    """
    blurred  = cv2.GaussianBlur(frame, (0, 0), sigmaX=3)
    return cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)


def gamma_correct(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Gamma correction to brighten very dark frames."""
    inv_gamma = 1.0 / gamma
    table     = np.array([((i / 255.0) ** inv_gamma) * 255
                           for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)


def normalize_brightness(frame: np.ndarray,
                          target_mean: float = 120.0) -> np.ndarray:
    """Scale frame brightness to a target mean luminance value."""
    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current    = float(np.mean(gray))
    if current < 1:
        return frame
    scale      = target_mean / current
    scaled     = np.clip(frame.astype(np.float32) * scale, 0, 255)
    return scaled.astype(np.uint8)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline — auto-selects enhancements based on
    detected imaging conditions.

    Args:
        frame : raw BGR frame from camera

    Returns:
        enhanced BGR frame ready for YOLO inference
    """
    if frame is None or frame.size == 0:
        return frame

    cond      = detect_conditions(frame)
    enhanced  = frame.copy()

    # Low-light: gamma lift first, then CLAHE
    if cond["low_light"]:
        enhanced = gamma_correct(enhanced, gamma=1.6)
        enhanced = apply_clahe(enhanced, clip_limit=3.0)

    # Overexposed: softer CLAHE to reduce blown highlights
    elif cond["overexposed"]:
        enhanced = apply_clahe(enhanced, clip_limit=1.5)

    # Normal light: standard CLAHE pass
    else:
        enhanced = apply_clahe(enhanced, clip_limit=2.0)

    # Denoise if noisy (e.g. night camera)
    if cond["noisy"]:
        enhanced = denoise(enhanced, strength=6)

    # Sharpen for blurry frames (motion blur, low-quality cameras)
    if cond["blurry"]:
        enhanced = sharpen(enhanced, amount=0.5)

    return enhanced


def draw_condition_overlay(frame: np.ndarray,
                            cond: dict[str, bool]) -> np.ndarray:
    """Draw a small condition indicator in top-right corner of frame."""
    h, w  = frame.shape[:2]
    active = [k for k, v in cond.items() if v]
    label  = "Cond: " + (", ".join(active) if active else "Normal")
    cv2.putText(frame, label, (w - 320, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    return frame
