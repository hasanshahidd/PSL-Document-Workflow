"""Detect 90/180/270-degree rotation using Tesseract OSD and correct it.

Scans pulled off a flatbed or photographed sideways come in 90 degrees off.
A regular skew-deskew (small-angle) can't recover from this. Tesseract's
OSD ('orientation and script detection') reports the dominant rotation.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
import pytesseract


@dataclass
class OrientationResult:
    rotation: int          # degrees rotated to correct (0, 90, 180, 270)
    confidence: float      # OSD confidence
    script: str | None     # detected writing system (Latin, etc.)


def detect_and_correct(img: np.ndarray) -> tuple[np.ndarray, OrientationResult]:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    try:
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError:
        return img, OrientationResult(rotation=0, confidence=0.0, script=None)
    rotation = int(osd.get("rotate", 0)) % 360
    confidence = float(osd.get("orientation_conf", 0.0))
    script = osd.get("script")
    if rotation == 0 or confidence < 1.0:
        return img, OrientationResult(rotation=0, confidence=confidence, script=script)
    if rotation == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated = img
    return rotated, OrientationResult(rotation=rotation, confidence=confidence, script=script)
