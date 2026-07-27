"""Simulate the detector pipeline against synthetic face geometry.

Why this exists:
  The user's webcam isn't delivering frames (hardware / privacy issue),
  so we can't dryrun the real YuNet pipeline. This script generates
  realistic face-box + landmark trajectories for 4 posture states and
  feeds them through compute_posture(). It tells us:
    - Does our classifier behave sanely across the state space?
    - Are the default thresholds sensible, or do they need tuning?
    - What's the false-positive / false-negative behavior near boundaries?

Each scenario:
  - Sets a baseline (good posture)
  - Generates 10 frames of varied good/bad geometry
  - Reports what compute_posture() classifies
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sitguard.config import Baseline, Thresholds
from sitguard.detector import Detection, PostureState, SittingDetector


def make_det(face_center, face_size, yaw=0.0, pitch=0.0, score=0.95):
    """Build a synthetic Detection — no real face detection, just geometry."""
    return Detection(
        face_box=(0, 0, 100, 100),  # bbox doesn't matter for posture
        face_center_norm=face_center,
        face_size_ratio=face_size,
        yaw_deg=yaw,
        pitch_deg=pitch,
        score=score,
        timestamp=0.0,
    )


def scenario(name, baseline, frames, expected_states):
    """Run a sequence of frames through the classifier and report."""
    detector = SittingDetector(baseline, Thresholds())
    print(f"\n=== {name} ===")
    print(f"  baseline:  c=({baseline.face_center_x:.2f}, {baseline.face_center_y:.2f})  "
          f"size={baseline.face_size_ratio:.3f}  yaw={baseline.yaw_deg:+.1f}°  pitch={baseline.pitch_deg:+.1f}°")
    print(f"  {'frame':>5}  {'cx':>5} {'cy':>5} {'size':>5}  {'yaw':>6} {'pitch':>6}  {'classified':>12}  {'expected':>12}  {'match':>5}")
    correct = 0
    for i, (frame, expected) in enumerate(zip(frames, expected_states)):
        cls = detector.compute_posture(frame)
        match = "✓" if cls == expected else "✗"
        if cls == expected:
            correct += 1
        c, s = frame.face_center_norm, frame.face_size_ratio
        print(f"  {i+1:>5}  {c[0]:>5.2f} {c[1]:>5.2f} {s:>5.3f}  "
              f"{frame.yaw_deg:>+6.1f} {frame.pitch_deg:>+6.1f}  "
              f"{cls.value:>12}  {expected.value:>12}  {match:>5}")
    print(f"  accuracy: {correct}/{len(frames)}")


def main():
    # ===== Scenario 1: User stays in GOOD posture =====
    baseline = Baseline(
        calibrated_at="2026-06-11",
        face_center_x=0.50, face_center_y=0.45, face_size_ratio=0.25,
        yaw_deg=0.0, pitch_deg=0.0,
    )
    frames = [
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),
        make_det((0.51, 0.46), 0.252, 1.0, -1.0),  # tiny jitter
        make_det((0.49, 0.44), 0.248, -0.5, 0.5),
        make_det((0.52, 0.47), 0.255, 2.0, -2.0),  # user leaned a bit
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),
    ]
    scenario("1. GOOD posture (tiny natural jitter)", baseline, frames,
             [PostureState.GOOD]*5)

    # ===== Scenario 2: User slouches forward (face gets bigger) =====
    frames = [
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),      # good
        make_det((0.50, 0.45), 0.270, 0.0, 5.0),      # +8% size
        make_det((0.50, 0.45), 0.290, 0.0, 8.0),      # +16% size → SLOUCHING
        make_det((0.50, 0.45), 0.310, 0.0, 10.0),     # +24% size → SLOUCHING
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),      # back to good
    ]
    scenario("2. SLOUCHING (face grows when leaning forward)", baseline, frames,
             [PostureState.GOOD, PostureState.GOOD, PostureState.SLOUCHING,
              PostureState.SLOUCHING, PostureState.GOOD])

    # ===== Scenario 3: User looks down at phone (pitch increases) =====
    frames = [
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),       # good
        make_det((0.50, 0.45), 0.250, 2.0, 10.0),     # +10° pitch
        make_det((0.50, 0.45), 0.250, 3.0, 22.0),     # +22° → HEAD_DOWN
        make_det((0.50, 0.45), 0.250, 1.0, 30.0),     # +30° → HEAD_DOWN
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),       # back
    ]
    scenario("3. HEAD_DOWN (pitch increases)", baseline, frames,
             [PostureState.GOOD, PostureState.GOOD, PostureState.HEAD_DOWN,
              PostureState.HEAD_DOWN, PostureState.GOOD])

    # ===== Scenario 4: User leans to the side =====
    frames = [
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),
        make_det((0.55, 0.45), 0.250, 5.0, 0.0),     # +5% x
        make_det((0.65, 0.45), 0.250, 10.0, 0.0),    # +15% x → LEANING
        make_det((0.65, 0.45), 0.250, 35.0, 0.0),    # +35° yaw → LEANING
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),
    ]
    scenario("4. LEANING (face drifts sideways or head turns)", baseline, frames,
             [PostureState.GOOD, PostureState.GOOD, PostureState.LEANING,
              PostureState.LEANING, PostureState.GOOD])

    # ===== Scenario 5: User leans back in chair (face gets smaller) =====
    frames = [
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),
        make_det((0.50, 0.45), 0.240, 0.0, -3.0),    # -4% size
        make_det((0.50, 0.45), 0.220, 0.0, -5.0),    # -12% size
        make_det((0.50, 0.45), 0.200, 0.0, -7.0),    # -20% size → LEAN_BACK
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),
    ]
    scenario("5. LEAN_BACK (face shrinks when leaning away)", baseline, frames,
             [PostureState.GOOD, PostureState.GOOD, PostureState.GOOD,
              PostureState.LEAN_BACK, PostureState.GOOD])

    # ===== Scenario 6: Edge case — slight forward lean with downward gaze =====
    # User has head down AND slightly forward. Priority: HEAD_DOWN wins.
    frames = [
        make_det((0.50, 0.45), 0.250, 0.0, 0.0),
        make_det((0.50, 0.45), 0.260, 0.0, 15.0),   # +4% size, +15° pitch
        make_det((0.50, 0.45), 0.270, 0.0, 25.0),   # +8% size, +25° pitch
    ]
    scenario("6. Combined: slight forward + head down (priority order)", baseline, frames,
             [PostureState.GOOD, PostureState.GOOD, PostureState.HEAD_DOWN])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
If you see mostly ✓ marks, the default thresholds are sane.
✗ marks tell us where to tune:
  - If SLOUCHING misfires (✗ on +5% size), tighten posture_size_close
  - If HEAD_DOWN requires >30° pitch, lower posture_pitch_down
  - If LEANING triggers on tiny drift, raise posture_x_drift / posture_y_drift
""")


if __name__ == "__main__":
    main()