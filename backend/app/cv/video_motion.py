import cv2
import numpy as np

def optical_flow_lk(video_path: str, max_frames: int = 120) -> dict:
    cap = cv2.VideoCapture(video_path)
    ok, old_frame = cap.read()
    if not ok:
        raise ValueError("Cannot read video")

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)

    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    frame_count = 0
    total_tracks = 0

    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if p0 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, 200, 0.3, 7, blockSize=7)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            total_tracks += len(good_new)
            p0 = good_new.reshape(-1,1,2)

        old_gray = frame_gray.copy()
        frame_count += 1

    cap.release()
    return {"frames_processed": frame_count, "tracked_points_total": int(total_tracks)}
