"""Camera backend diagnostic — find out WHY frames are black.

Works on any Windows machine with opencv-python installed. Run it, read the
output, and send the output to whoever is debugging.

What it does:
  For each OpenCV capture backend (DirectShow, Media Foundation, auto), it
  opens the default camera, grabs 15 frames, and reports the mean brightness.
  A backend that returns non-black frames is the one SitGuard should use.

Usage:
    python scripts/camera_diagnose.py
"""
from __future__ import annotations

import sys

import cv2
import numpy as np


def probe(index: int, backend_name: str, backend_flag) -> None:
    print(f"\n=== Backend: {backend_name} ===")
    try:
        if backend_flag is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend_flag)
    except Exception as exc:  # noqa: BLE001
        print(f"  could not construct capture: {exc}")
        return

    if not cap.isOpened():
        print("  isOpened = False  -> backend cannot open this camera")
        cap.release()
        return

    print("  isOpened = True")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)) if fourcc else "?"
    print(f"  reported resolution: {w}x{h}  codec: {codec!r}")

    means: list[float] = []
    for _ in range(15):
        ok, frame = cap.read()
        if not ok or frame is None:
            means.append(float("nan"))
            continue
        means.append(float(np.mean(frame)))

    valid = [m for m in means if m == m]  # drop NaN
    if not valid:
        print("  RESULT: no frames could be grabbed")
    else:
        lo, hi, last = min(valid), max(valid), valid[-1]
        print(f"  brightness over 15 reads: min={lo:.2f} max={hi:.2f} last={last:.2f}")
        if hi < 5:
            print("  RESULT: BLACK (this backend returns all-zero frames)")
        else:
            print("  RESULT: OK (non-black frames -> this backend works!)")
    cap.release()


def main() -> int:
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {sys.platform}")
    index = 0

    candidates = [
        ("CAP_DSHOW", getattr(cv2, "CAP_DSHOW", None)),
        ("CAP_MSMF", getattr(cv2, "CAP_MSMF", None)),
        ("CAP_ANY (auto)", None),
        ("CAP_VFW", getattr(cv2, "CAP_VFW", None)),
    ]
    for name, flag in candidates:
        if flag is None and name != "CAP_ANY (auto)":
            continue
        probe(index, name, flag)

    print("\n--- Summary ---")
    print("If a backend shows OK above, SitGuard should open the camera with that")
    print("backend flag. If ALL backends are BLACK, the camera itself is returning")
    print("black frames at the driver level (hardware / UVC firmware issue), not a")
    print("software problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
