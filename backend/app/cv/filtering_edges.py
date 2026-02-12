import cv2
import numpy as np

def gaussian_blur(img_bgr: np.ndarray, k: int = 5) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.GaussianBlur(img_bgr, (k, k), 0)

def median_blur(img_bgr: np.ndarray, k: int = 5) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.medianBlur(img_bgr, k)

def sharpen(img_bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(img_bgr, -1, kernel)

def sobel_edges(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

def canny(img_bgr: np.ndarray, t1: int = 60, t2: int = 140) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
