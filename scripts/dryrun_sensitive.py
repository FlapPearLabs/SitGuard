"""Ultra-sensitive dryrun — lower thresholds, faster poll, dump frames.

This script runs YuNet with score_threshold=0.2 (vs 0.6 default) and
samples every 200ms. It saves the first detected face to disk and
prints the raw detection values so we can see exactly what the model
sees.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from sitguard.detector import _find_model_path, _LANDMARKS_3D


def main() -> int:
    model_path = _find_model_path()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: can't open camera 0")
        return 1

    # Use VERY low threshold so we see anything
    detector = cv2.FaceDetectorYN.create(
        str(model_path), "", (320, 320),
        score_threshold=0.2,
        nms_threshold=0.3,
        top_k=10,
    )

    print(f"Camera opened. YuNet model = {model_path}")
    print(f"Sampling every 200ms for 15 seconds at low threshold (0.2)")
    print(f"Sit in front of the camera NOW. Press Ctrl+C to stop.\n")

    out_dir = Path(__file__).parent
    saved_frame = False
    start = time.time()
    try:
        i = 0
        while time.time() - start < 15.0:
            ok, frame = cap.read()
            if not ok:
                print(f"[t+{time.time()-start:5.1f}s] read fail")
                time.sleep(0.2)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, raw = detector.detect(frame)

            i += 1
            if raw is None or len(raw) == 0:
                if i % 5 == 0:  # don't spam
                    print(f"[t+{time.time()-start:5.1f}s] no face (mean brightness={frame.mean():.0f})")
            else:
                best = max(raw, key=lambda r: r[14])
                x, y, bw, bh = best[0:4]
                landmarks = best[4:14].reshape(5, 2)
                score = float(best[14])
                cx_norm = (x + bw / 2) / w
                cy_norm = (y + bh / 2) / h
                size_ratio = (bw * bh) / (w * h)

                # Pose from landmarks
                focal = float(w)
                center = (w / 2.0, h / 2.0)
                cm = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
                dc = np.zeros((4, 1))
                ok2, rvec, _ = cv2.solvePnP(
                    _LANDMARKS_3D, landmarks.astype(np.float64), cm, dc,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok2:
                    rmat, _ = cv2.Rodrigues(rvec)
                    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
                    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
                    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
                else:
                    pitch = yaw = 0.0

                print(
                    f"[t+{time.time()-start:5.1f}s] FACE  "
                    f"score={score:.2f} "
                    f"size={size_ratio:.3f} "
                    f"center=({cx_norm:.2f},{cy_norm:.2f}) "
                    f"yaw={yaw:+.1f}° pitch={pitch:+.1f}°"
                )

                # Save the first face frame to disk so we can SEE it
                if not saved_frame:
                    # Annotate it
                    cv2.rectangle(frame,
                                  (int(x), int(y)),
                                  (int(x + bw), int(y + bh)),
                                  (124, 92, 255), 2)
                    for (lx, ly) in landmarks:
                        cv2.circle(frame, (int(lx), int(ly)), 3, (124, 92, 255), -1)
                    out_path = out_dir / "dryrun_first_face.png"
                    cv2.imwrite(str(out_path), frame)
                    print(f"  → saved annotated frame: {out_path}")
                    saved_frame = True

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\nframes read: {i}")
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())