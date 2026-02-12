import cv2
import numpy as np
from joblib import load

def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    # Simple classical pipeline: resize + HOG-like gradients + histogram
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    hist = cv2.calcHist([gray], [0], None, [32], [0,256]).flatten()
    mag_hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [32], [0,256]).flatten()

    feat = np.concatenate([hist, mag_hist]).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat.reshape(1, -1)

def predict_classical(img_bgr: np.ndarray, model_path: str) -> dict:
    clf = load(model_path)
    x = extract_features(img_bgr)
    pred = clf.predict(x)[0]
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)[0]
        classes = clf.classes_
        return {"predicted_class": str(pred), "all_probs": {str(classes[i]): float(proba[i]) for i in range(len(classes))}}
    return {"predicted_class": str(pred)}
