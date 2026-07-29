# -*- coding: utf-8 -*-
"""Deep probe: try combinations of backend x resolution x fourcc x exposure settings."""
import cv2, time

def test(label, backend, w=None, h=None, fourcc=None, manual_exposure=False):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print(f'{label}: OPEN FAILED')
        return
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if w:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if manual_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual mode (DSHOW convention)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    best = -1.0
    for i in range(10):
        ok, f = cap.read()
        if ok and f is not None:
            m = float(f.mean())
            best = max(best, m)
        time.sleep(0.1)
    aw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ah = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    print(f'{label}: best_mean={best:.1f} actual={int(aw)}x{int(ah)} exposure={exp} gain={gain}')
    cap.release()
    time.sleep(0.5)

print('--- DSHOW default ---')
test('DSHOW/default', cv2.CAP_DSHOW)
print('--- DSHOW 640x480 ---')
test('DSHOW/640x480', cv2.CAP_DSHOW, 640, 480)
print('--- DSHOW 640x480 MJPG ---')
test('DSHOW/640x480/MJPG', cv2.CAP_DSHOW, 640, 480, 'MJPG')
print('--- DSHOW 1280x720 YUY2 ---')
test('DSHOW/720p/YUY2', cv2.CAP_DSHOW, 1280, 720, 'YUY2')
print('--- DSHOW manual exposure ---')
test('DSHOW/manual-exposure', cv2.CAP_DSHOW, 640, 480, None, True)
print('--- camera index 1 (if second cam exists) ---')
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print('index1 isOpened =', cap.isOpened())
if cap.isOpened():
    for i in range(5):
        ok, f = cap.read()
        if ok and f is not None:
            print('  index1 mean =', float(f.mean()))
        time.sleep(0.2)
cap.release()
