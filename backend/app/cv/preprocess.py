import cv2
import numpy as np

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def resize_keep_aspect(img_bgr: np.ndarray, max_side: int = 768) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img_bgr
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def normalize_uint8(gray: np.ndarray) -> np.ndarray:
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return g.astype(np.uint8)
