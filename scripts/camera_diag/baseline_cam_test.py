# -*- coding: utf-8 -*-
"""Baseline camera test outside SitGuard: DSHOW vs default."""
import time

import cv2

print("cv2:", cv2.__version__)

for label, args in [("CAP_DSHOW", (0, cv2.CAP_DSHOW)), ("DEFAULT", (0,))]:
    cap = cv2.VideoCapture(*args)
    print(f"\n[{label}] isOpened={cap.isOpened()}")
    if not cap.isOpened():
        continue
    means = []
    t0 = time.time()
    for i in range(12):
        ok, frame = cap.read()
        m = float(frame.mean()) if ok and frame is not None else -1.0
        means.append(round(m, 1))
    print(f"[{label}] 12 frame means: {means}  ({time.time()-t0:.1f}s)")
    cap.release()
    time.sleep(1.0)
