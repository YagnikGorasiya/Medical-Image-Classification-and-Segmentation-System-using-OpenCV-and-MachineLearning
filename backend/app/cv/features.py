import cv2
import numpy as np

def harris_corners(img_bgr: np.ndarray, block_size: int = 2, ksize: int = 3, k: float = 0.04) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    out = img_bgr.copy()
    out[dst > 0.01 * dst.max()] = [0, 0, 255]
    return out

def orb_keypoints(img_bgr: np.ndarray, nfeatures: int = 500) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)
    out = cv2.drawKeypoints(img_bgr, kp, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    return out

def hog_visualization_like(img_bgr: np.ndarray) -> np.ndarray:
    # Simple gradient magnitude map (lightweight substitute demo for HOG concept)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
