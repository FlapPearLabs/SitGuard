# -*- coding: utf-8 -*-
"""Frame forensics: pure-zero frames vs dark-room noise."""
import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("isOpened:", cap.isOpened())
print("exposure:", cap.get(cv2.CAP_PROP_EXPOSURE))
print("fps prop:", cap.get(cv2.CAP_PROP_FPS))
print("brightness prop:", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("gain prop:", cap.get(cv2.CAP_PROP_GAIN))

for i in range(6):
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"frame#{i}: grab failed")
        continue
    mn, mx = int(frame.min()), int(frame.max())
    mean, std = float(frame.mean()), float(frame.std())
    nonzero = int(np.count_nonzero(frame))
    print(f"frame#{i}: min={mn} max={mx} mean={mean:.4f} std={std:.4f} "
          f"nonzero_pixels={nonzero}/{frame.size}")
    if i == 5:
        cv2.imwrite(r'D:\未完成项目\SitGuard\scripts\camera_diag\last_frame.png', frame)
        # brightness-boosted version to reveal faint content
        boosted = cv2.convertScaleAbs(frame, alpha=8.0, beta=0)
        cv2.imwrite(r'D:\未完成项目\SitGuard\scripts\camera_diag\last_frame_boost8x.png', boosted)
        print("saved last_frame.png and last_frame_boost8x.png")
cap.release()
