# -*- coding: utf-8 -*-
"""Live camera monitor: log frame brightness every ~0.5s for 120s.
User presses F10 / Fn+F10 during the run; a working toggle shows mean jumping from 0."""
import cv2, time, sys, os

LOG = r'D:\未完成项目\SitGuard\scripts\camera_diag\live_monitor.log'

def log(msg):
    line = time.strftime('%H:%M:%S') + ' ' + msg
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

os.makedirs(os.path.dirname(LOG), exist_ok=True)
with open(LOG, 'w', encoding='utf-8') as f:
    f.write('=== live monitor start ===\n')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
log(f'DSHOW isOpened={cap.isOpened()}')
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    log(f'DEFAULT isOpened={cap.isOpened()}')

t0 = time.time()
n = 0
last_state = None
while time.time() - t0 < 120:
    ok, frame = cap.read()
    if not ok or frame is None:
        state = 'READ-FAIL'
        mean = -1.0
    else:
        mean = float(frame.mean())
        state = 'BLACK' if mean < 1.0 else 'LIVE'
    n += 1
    # log every frame while state changes, otherwise every 4th
    if state != last_state:
        log(f'*** STATE CHANGE -> {state} (mean={mean:.1f}) frame#{n}')
        last_state = state
    elif n % 4 == 0:
        log(f'{state} mean={mean:.1f} frame#{n}')
    time.sleep(0.4)

log(f'=== done, {n} frames in 120s ===')
cap.release()
