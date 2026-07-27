"""Try multiple camera backends to bypass any exclusive lock (e.g. WeChat)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import time

# OpenCV backend constants
BACKENDS = [
    ("CAP_ANY", cv2.CAP_ANY),
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_V4L2", cv2.CAP_V4L2),  # not relevant on Windows but harmless
]

out_dir = Path(__file__).parent

for name, backend in BACKENDS:
    print(f"\n--- Trying backend {name} ---")
    cap = cv2.VideoCapture(0 + backend)
    if not cap.isOpened():
        print(f"  {name}: open() returned False")
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Try reading 5 frames — give camera time to warm up
    last_brightness = 0.0
    for attempt in range(5):
        ok, frame = cap.read()
        if ok and frame is not None:
            brightness = frame.mean()
            print(f"  attempt {attempt+1}: brightness = {brightness:.1f}")
            last_brightness = brightness
            if brightness > 5:
                # Got a real frame — save it
                path = out_dir / f"dryrun_frame_{name}.png"
                cv2.imwrite(str(path), frame)
                print(f"  ✓ saved: {path}")
                cap.release()
                sys.exit(0)
        time.sleep(0.5)

    cap.release()
    print(f"  {name}: 5 attempts, last brightness = {last_brightness:.1f}")

print("\n\n=== All backends returned black frames ===")
print("Most likely cause: another app (WeChat video call, Zoom, etc.) is holding the camera.")
print("Try closing those apps and re-running this script.")