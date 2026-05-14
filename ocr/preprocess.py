"""Image preprocessing for OCR.

Every transformation here is designed to improve Tesseract's accuracy on
scanned legal documents. Each function is pure (numpy in -> numpy out) so
the pipeline can compose them and report which stages ran.

Pipeline order (when applied):
  1. ensure grayscale
  2. upscale if low-DPI
  3. shadow / illumination correction
  4. denoise (non-local means or bilateral, picked by noise estimate)
  5. CLAHE contrast normalisation
  6. deskew via projection profile + Hough fallback
  7. border trim
  8. binarize (Otsu, Sauvola, adaptive — caller picks)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from skimage.filters import threshold_sauvola


# --- helpers ----------------------------------------------------------------
def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def estimate_noise(img: np.ndarray) -> float:
    """Return a rough noise estimate in [0,1]; based on Laplacian variance."""
    gray = _ensure_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_skew_projection(img: np.ndarray) -> float:
    """Projection profile based skew estimate. Robust on dense body text."""
    gray = _ensure_gray(img)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best_angle, best_score = 0.0, -1.0
    h, w = bin_.shape
    for angle in np.arange(-5.0, 5.1, 0.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(bin_, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderValue=0)
        proj = rotated.sum(axis=1).astype(np.float64)
        score = float(np.var(proj))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def estimate_skew_hough(img: np.ndarray) -> float | None:
    """Hough line based skew estimate. Better on sparse documents (cover
    pages, exhibits) where projection profile flattens out.

    Returns None when no confident dominant angle is detected so the caller
    can fall back to the projection profile estimate.
    """
    gray = _ensure_gray(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 720, threshold=200)
    if lines is None:
        return None
    angles: list[float] = []
    for rho_theta in lines[:50]:
        _, theta = rho_theta[0]
        deg = (theta * 180.0 / np.pi) - 90.0
        if -10.0 <= deg <= 10.0:  # only consider lines close to horizontal
            angles.append(float(deg))
    if not angles:
        return None
    # use median for robustness against outliers
    return float(np.median(angles))


def estimate_skew(img: np.ndarray) -> float:
    """Combined estimate. Try projection profile first, cross check with
    Hough when the two agree within 1 degree we trust the projection
    result, otherwise prefer the Hough estimate (sparse layouts).
    """
    proj = estimate_skew_projection(img)
    hough = estimate_skew_hough(img)
    if hough is None:
        return proj
    if abs(proj - hough) <= 1.0:
        return proj
    return hough


# --- transformations --------------------------------------------------------
def upscale_if_low_res(img: np.ndarray, min_height: int = 1500) -> tuple[np.ndarray, bool]:
    h = img.shape[0]
    if h >= min_height:
        return img, False
    scale = min_height / h
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC), True


def remove_shadows(img: np.ndarray) -> np.ndarray:
    """Knock out uneven illumination using a morphological background subtract."""
    gray = _ensure_gray(img)
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def denoise(img: np.ndarray, noise_score: float | None = None) -> np.ndarray:
    """Pick a denoiser based on estimated noise — speed/quality tradeoff."""
    gray = _ensure_gray(img)
    noise = noise_score if noise_score is not None else estimate_noise(gray)
    if noise < 50:  # already clean
        return cv2.bilateralFilter(gray, d=5, sigmaColor=20, sigmaSpace=20)
    return cv2.fastNlMeansDenoising(gray, h=10)


def clahe_contrast(img: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(img: np.ndarray, angle: float | None = None) -> tuple[np.ndarray, float]:
    """Rotate the page so text lines are horizontal."""
    gray = _ensure_gray(img)
    angle = angle if angle is not None else estimate_skew(gray)
    if abs(angle) < 0.2:
        return gray, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def trim_borders(img: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Crop the largest content-bounding rectangle (drops scan borders)."""
    gray = _ensure_gray(img)
    _, bin_ = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(bin_)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    pad = 5
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, gray.shape[1])
    y1 = min(y + h + pad, gray.shape[0])
    return gray[y0:y1, x0:x1]


def binarize_otsu(img: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(img)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_


def binarize_sauvola(img: np.ndarray, window: int = 25) -> np.ndarray:
    gray = _ensure_gray(img)
    t = threshold_sauvola(gray, window_size=window)
    return ((gray > t) * 255).astype(np.uint8)


def binarize_adaptive(img: np.ndarray) -> np.ndarray:
    gray = _ensure_gray(img)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )


# --- composed pipelines -----------------------------------------------------
@dataclass
class PreprocessReport:
    upscaled: bool = False
    shadow_removed: bool = False
    denoised: bool = False
    clahe: bool = False
    deskew_angle: float = 0.0
    trimmed: bool = False
    binariser: str | None = None
    noise_estimate: float = 0.0
    stages: list[str] = field(default_factory=list)


def standard_pipeline(img: np.ndarray, binariser: str = "otsu") -> tuple[np.ndarray, PreprocessReport]:
    """The default 'always-on' pipeline for scanned legal pages."""
    report = PreprocessReport()
    img = _ensure_gray(img)
    img, up = upscale_if_low_res(img)
    if up:
        report.upscaled = True
        report.stages.append("upscale")
    img = remove_shadows(img); report.shadow_removed = True; report.stages.append("shadow")
    noise = estimate_noise(img); report.noise_estimate = round(noise, 2)
    img = denoise(img, noise_score=noise); report.denoised = True; report.stages.append("denoise")
    img = clahe_contrast(img); report.clahe = True; report.stages.append("clahe")
    img, angle = deskew(img); report.deskew_angle = round(angle, 2)
    if angle != 0.0:
        report.stages.append(f"deskew({angle:.2f}deg)")
    img = trim_borders(img); report.trimmed = True; report.stages.append("trim")
    if binariser == "otsu":
        img = binarize_otsu(img)
    elif binariser == "sauvola":
        img = binarize_sauvola(img)
    elif binariser == "adaptive":
        img = binarize_adaptive(img)
    report.binariser = binariser
    report.stages.append(f"binarize:{binariser}")
    return img, report
