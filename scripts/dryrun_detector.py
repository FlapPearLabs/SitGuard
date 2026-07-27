"""Dry-run smoke test for the detection pipeline.

Runs the actual YuNet detector against the real webcam for ~10 seconds,
prints every detection to stdout. Lets you verify the core engine
works end-to-end without launching the full Qt UI.

Usage:
    python scripts/dryrun_detector.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `sitguard` importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from sitguard.config import Baseline, Thresholds, load  # noqa: E402
from sitguard.detector import CAMERA_OK, SittingDetector  # noqa: E402


def fmt(det) -> str:
    if det.face_box is None:
        return "no face"
    x, y, w, h = det.face_box
    return (
        f"face bbox=({x},{y},{w}x{h}) "
        f"size={det.face_size_ratio:.3f} "
        f"yaw={(det.yaw_deg or 0):+.1f}° "
        f"pitch={(det.pitch_deg or 0):+.1f}° "
        f"conf={det.score:.2f}"
    )


def main() -> int:
    cfg = load()
    detector = SittingDetector(
        baseline=cfg.baseline,
        thresholds=cfg.thresholds,
        camera=cfg.camera_index,
    )
    if not detector.open_camera():
        print(f"ERROR opening camera: {detector.camera_error}")
        print(f"status = {detector.camera_status}")
        return 1

    print(f"Camera opened OK at index {detector.camera_index}")
    print(f"Polling at ~2s intervals for 20 seconds.")
    print("Sit in front of the camera — you should see face box data here.\n")

    start = time.time()
    frames = 0
    faces_seen = 0
    try:
        while time.time() - start < 20.0:
            frame = detector.read_frame()
            if frame is None:
                print(f"[t+{time.time()-start:5.1f}s] no frame")
                time.sleep(0.5)
                continue
            frames += 1
            det = detector.process(frame)
            state = detector.state
            seen = "FACE " if det.face_box else "empty"
            if det.face_box:
                faces_seen += 1
            print(
                f"[t+{time.time()-start:5.1f}s] {seen}  "
                f"state={state.value:14s}  "
                f"{fmt(det)}"
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\n--- summary ---")
    print(f"frames read:  {frames}")
    print(f"frames w/face: {faces_seen}")
    print(f"final state:  {detector.state.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())