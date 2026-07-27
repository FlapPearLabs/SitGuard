"""Dump a single raw camera frame to disk, with no detection — pure 'what does the webcam actually see'.

Saves 'dryrun_raw_frame.png' so we can SEE the camera output.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Wait briefly so the camera warms up (some cams need 1-2 seconds)
time.sleep(1.5)

ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    print("ERROR: cap.read() returned False / None")
    sys.exit(1)

print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
print(f"Mean brightness (0=black, 255=white): {frame.mean():.1f}")
print(f"Min: {frame.min()}, Max: {frame.max()}")

out = Path(__file__).parent / "dryrun_raw_frame.png"
cv2.imwrite(str(out), frame)
print(f"\nSaved: {out}")
print("Open it in your image viewer to see what the camera actually captured.")