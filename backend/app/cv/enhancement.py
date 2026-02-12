import cv2
import numpy as np

def adjust_brightness_contrast(img_bgr: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    # alpha: contrast, beta: brightness
    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return out

def histogram_equalization_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def clahe_gray(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    out = clahe.apply(gray)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
