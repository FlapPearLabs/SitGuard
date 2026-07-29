"""Webcam-based sitting detector.

Pipeline (per frame):
  1. YuNet face detection → bbox + 5 landmarks
  2. Compute size, position, yaw, pitch from landmarks
  3. Track face across frames (IoU-based)
  4. Five-signal fusion → presence state

YuNet model: face_detection_yunet_2023mar.onnx (~228 KB).
MoveNet Lightning (~3 MB) is reserved for posture scoring in v0.2.

This module owns no timers or UI — pure perception.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .config import Baseline, Thresholds
from .devices import CameraDevice


class PresenceState(str, Enum):
    NOT_PRESENT = "not_present"
    PRESENT_IDLE = "present_idle"   # face found, but not facing screen
    PRESENT_FOCUSED = "present_focused"  # sitting at desk, counting


@dataclass
class Detection:
    """Single-frame detection result."""

    face_box: tuple[int, int, int, int] | None  # (x, y, w, h) in pixels
    face_center_norm: tuple[float, float] | None  # (x, y) 0..1
    face_size_ratio: float | None  # bbox area / frame area
    yaw_deg: float | None
    pitch_deg: float | None
    score: float  # confidence 0..1
    timestamp: float


# Why camera init failed (when open_camera returns False).
CAMERA_OK = "ok"
CAMERA_NO_DEVICE = "no_device"           # no index responded
CAMERA_PERMISSION_DENIED = "permission_denied"  # Win privacy setting off
CAMERA_IN_USE = "in_use"                 # another app has it locked


class PostureState(str, Enum):
    """Sitting posture classification, computed every frame from face geometry."""

    UNKNOWN = "unknown"        # no face OR no baseline yet
    GOOD = "good"              # within tolerance of baseline
    HEAD_DOWN = "head_down"     # pitch too high (looking down at phone / book)
    SLOUCHING = "slouching"     # face larger than baseline (leaned forward)
    LEANING = "leaning"         # face center drifted (sideways or up/down)
    LEAN_BACK = "lean_back"     # face smaller than baseline (leaned back)


@dataclass
class PresenceSignals:
    """Weighted signals for the fusion vote."""

    face_present: float = 0.0
    face_size_ok: float = 0.0
    face_stable: float = 0.0
    head_facing: float = 0.0
    recent_motion: float = 0.0

    @property
    def total(self) -> float:
        # Weights sum to 1.0
        return (
            0.30 * self.face_present
            + 0.25 * self.face_size_ok
            + 0.15 * self.face_stable
            + 0.20 * self.head_facing
            + 0.10 * self.recent_motion
        )


# --- YuNet model loader -----------------------------------------------------

_DEFAULT_MODEL_NAME = "face_detection_yunet_2023mar.onnx"

# 3D reference face model for solvePnP (YuNet's 5 landmarks)
# (right_eye, left_eye, nose, right_mouth, left_mouth) — generic adult
_FACE_3D = np.array(
    [
        (0.0, 0.0, 0.0),       # placeholder, replaced below
    ],
    dtype=np.float64,
)

# Standard 5-point face model in mm (used by OpenCV face module docs)
_LANDMARKS_3D = np.array(
    [
        [-30.0, -30.0, -30.0],   # right eye corner
        [30.0, -30.0, -30.0],    # left eye corner
        [0.0, 0.0, 0.0],         # nose tip
        [-25.0, 30.0, -20.0],    # right mouth corner
        [25.0, 30.0, -20.0],     # left mouth corner
    ],
    dtype=np.float64,
)


def _find_model_path(models_dir: Path | None = None) -> Path:
    """Locate the YuNet ONNX model. Downloads if missing (small, ~228 KB)."""
    candidates: list[Path] = []
    if models_dir is not None:
        candidates.append(models_dir / _DEFAULT_MODEL_NAME)
    candidates.append(Path(__file__).resolve().parent.parent / "models" / _DEFAULT_MODEL_NAME)
    candidates.append(Path.home() / ".sitguard" / "models" / _DEFAULT_MODEL_NAME)

    for c in candidates:
        if c.exists():
            return c

    # Not found — raise with helpful hint
    raise FileNotFoundError(
        "YuNet model not found. Place face_detection_yunet_2023mar.onnx in:\n"
        f"  {candidates[1]}\n"
        "Download from: https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )


# --- Main detector ----------------------------------------------------------


class SittingDetector:
    """Webcam-based sitting presence detector.

    Usage:
        detector = SittingDetector(baseline, thresholds)
        while True:
            frame = read_frame()
            detection = detector.process(frame)
            state = detector.presence_state()
    """

    def __init__(
        self,
        baseline: Baseline,
        thresholds: Thresholds,
        models_dir: Path | None = None,
        camera: CameraDevice | int = 0,
    ) -> None:
        """Create a detector.

        `camera` may be either a CameraDevice (preferred — knows its name)
        or a plain int index (fallback for tests / legacy callers).
        """
        self.baseline = baseline
        self.thresholds = thresholds
        if isinstance(camera, CameraDevice):
            self.camera_index = camera.index
            self.camera_name = camera.name
        else:
            self.camera_index = int(camera)
            self.camera_name = ""

        model_path = _find_model_path(models_dir)
        # cv2.FaceDetectorYN.create returns a detector object (OpenCV 4.7+)
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,
        )

        # Camera is opened lazily so the constructor stays cheap / testable
        self._cap: cv2.VideoCapture | None = None
        self._camera_status: str = CAMERA_OK
        self._camera_error: str = ""

        # Tracking buffers
        self._recent_centers: deque[tuple[float, float]] = deque(maxlen=30)  # ~2s at 15fps
        self._last_face_seen: float = 0.0
        self._last_motion_time: float = 0.0

        # State machine
        self._state = PresenceState.NOT_PRESENT
        self._state_entered_at: float = 0.0
        self._focused_started_at: float | None = None

    # -- camera --

    def open_camera(self) -> bool:
        """Open the webcam. Returns True on success.

        On failure, populates `camera_status` and `camera_error` with a
        structured reason the UI can act on.
        """
        if self._cap is not None and self._cap.isOpened():
            return True

        # Probe a couple of indices if our preferred one doesn't open
        indices_to_try = [self.camera_index]
        for i in range(3):
            if i not in indices_to_try:
                indices_to_try.append(i)

        for idx in indices_to_try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                grabbed, _ = cap.read()
                if grabbed:
                    self._cap = cap
                    self.camera_index = idx
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._camera_status = CAMERA_OK
                    self._camera_error = ""
                    return True
                # Opened but can't read → likely in-use by another app
                cap.release()
                self._camera_status = CAMERA_IN_USE
                self._camera_error = (
                    f"Camera {idx} is open but unreadable. "
                    "Is another app (Zoom, Teams, OBS) using it? Close it and retry."
                )
                continue
            cap.release()

        # Nothing worked. Decide between "no device" and "permission denied".
        from .devices import check_privacy_allowed, discover_cameras

        if not check_privacy_allowed():
            self._camera_status = CAMERA_PERMISSION_DENIED
            self._camera_error = (
                "Windows is blocking desktop apps from accessing the camera. "
                "Open Settings → Privacy & security → Camera, then turn on "
                "'Let desktop apps access your camera'."
            )
        else:
            devs = discover_cameras()
            if devs.devices:
                self._camera_status = CAMERA_IN_USE
                names = ", ".join(d.name for d in devs.devices)
                self._camera_error = (
                    f"Found {len(devs.devices)} camera(s) ({names}) but none "
                    "could be opened. Another app may have them locked."
                )
            else:
                self._camera_status = CAMERA_NO_DEVICE
                self._camera_error = (
                    "No camera devices detected. Plug in a webcam or check "
                    "Device Manager → Cameras for disabled devices."
                )
        return False

    @property
    def camera_status(self) -> str:
        """One of CAMERA_OK / CAMERA_NO_DEVICE / CAMERA_PERMISSION_DENIED / CAMERA_IN_USE."""
        return self._camera_status

    @property
    def camera_error(self) -> str:
        """Human-readable explanation when camera_status != CAMERA_OK."""
        return self._camera_error

    def close_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> np.ndarray | None:
        """Grab one frame. Returns None on failure."""
        if self._cap is None:
            if not self.open_camera():
                return None
        ok, frame = self._cap.read()
        if not ok:
            return None
        # Mirror so behavior matches user expectation (selfie-style)
        return cv2.flip(frame, 1)

    # -- core --

    def process(self, frame: np.ndarray) -> Detection:
        """Run detection on a single frame. Updates internal state."""
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))

        # YuNet returns (num_faces, 15) — each row: x,y,w,h, 5 landmarks (x,y), score
        _, raw = self._detector.detect(frame)
        now = time.time()

        if raw is None or len(raw) == 0:
            self._recent_centers.clear()
            return Detection(
                face_box=None,
                face_center_norm=None,
                face_size_ratio=None,
                yaw_deg=None,
                pitch_deg=None,
                score=0.0,
                timestamp=now,
            )

        # Pick the highest-confidence face
        best = max(raw, key=lambda r: r[14])
        x, y, bw, bh = best[0:4]
        landmarks = best[4:14].reshape(5, 2)
        score = float(best[14])

        # Normalize to 0..1
        cx_norm = (x + bw / 2) / w
        cy_norm = (y + bh / 2) / h
        size_ratio = (bw * bh) / (w * h)

        # Pose from solvePnP
        yaw, pitch = self._estimate_head_pose(landmarks, w, h)

        det = Detection(
            face_box=(int(x), int(y), int(bw), int(bh)),
            face_center_norm=(cx_norm, cy_norm),
            face_size_ratio=size_ratio,
            yaw_deg=yaw,
            pitch_deg=pitch,
            score=score,
            timestamp=now,
        )

        self._recent_centers.append((cx_norm, cy_norm))
        self._last_face_seen = now

        # Cheap motion proxy: if the face is moving a little, count as motion
        if len(self._recent_centers) >= 2:
            dx = self._recent_centers[-1][0] - self._recent_centers[-2][0]
            dy = self._recent_centers[-1][1] - self._recent_centers[-2][1]
            if math.hypot(dx, dy) > 0.005:
                self._last_motion_time = now

        self._update_state(det)
        return det

    def _estimate_head_pose(
        self,
        landmarks: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        """Compute yaw & pitch from the 5-point landmarks."""
        # Camera matrix approximation (no calibration — good enough for pose binning)
        focal_length = float(frame_w)
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))

        image_points = landmarks.astype(np.float64)
        ok, rvec, _ = cv2.solvePnP(
            _LANDMARKS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        # Decompose to euler angles
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(-rmat[2, 0], sy)
            yaw = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[2, 0], sy)
            yaw = 0.0
        return math.degrees(yaw), math.degrees(pitch)

    # -- five-signal fusion --

    def compute_signals(self, det: Detection) -> PresenceSignals:
        sigs = PresenceSignals()
        now = time.time()

        # 1. Face present
        sigs.face_present = 1.0 if det.face_box is not None else 0.0

        # 2. Face size within tolerance of baseline
        if det.face_size_ratio is not None and self.baseline.face_size_ratio > 0:
            ratio = det.face_size_ratio / self.baseline.face_size_ratio
            sigs.face_size_ok = max(0.0, 1.0 - abs(1.0 - ratio) / self.thresholds.size_tolerance)
        elif det.face_size_ratio is not None:
            # No baseline yet — accept any reasonable size
            sigs.face_size_ok = 1.0 if 0.02 < det.face_size_ratio < 0.6 else 0.0

        # 3. Position stability (low variance over last 2s)
        if len(self._recent_centers) >= 10:
            arr = np.array(self._recent_centers)
            var = float(np.mean(np.var(arr, axis=0)))
            # var < 0.0005 → fully stable → 1.0; var > 0.01 → fully unstable → 0.0
            sigs.face_stable = max(0.0, 1.0 - var / 0.01)

        # 4. Head facing (yaw + pitch within tolerance)
        if det.yaw_deg is not None and det.pitch_deg is not None:
            yaw_ok = abs(det.yaw_deg) < self.thresholds.yaw_tolerance_deg
            pitch_ok = abs(det.pitch_deg) < self.thresholds.pitch_tolerance_deg
            sigs.head_facing = 1.0 if (yaw_ok and pitch_ok) else 0.0

        # 5. Recent motion
        sigs.recent_motion = 1.0 if (now - self._last_motion_time) < 60.0 else 0.0

        return sigs

    # -- state machine --

    def _update_state(self, det: Detection) -> None:
        now = time.time()
        sigs = self.compute_signals(det)
        score = sigs.total
        face_seen = det.face_box is not None

        previous = self._state
        next_state = previous
        dwell = now - self._state_entered_at

        if not face_seen:
            # Lost the face
            if (now - self._last_face_seen) > self.thresholds.absent_timeout_seconds:
                next_state = PresenceState.NOT_PRESENT
        else:
            if score >= 0.70:
                next_state = PresenceState.PRESENT_FOCUSED
            elif score >= 0.40:
                next_state = PresenceState.PRESENT_IDLE

        # Hysteresis: state must hold for dwell time before transition
        if next_state != previous:
            if next_state == PresenceState.PRESENT_FOCUSED:
                required = self.thresholds.focused_dwell_seconds
            elif next_state == PresenceState.PRESENT_IDLE and previous == PresenceState.NOT_PRESENT:
                required = self.thresholds.present_dwell_seconds
            else:
                required = 0.5  # fast exit

            if dwell >= required:
                self._state = next_state
                self._state_entered_at = now
                if next_state == PresenceState.PRESENT_FOCUSED and self._focused_started_at is None:
                    self._focused_started_at = now

        # Reset focused streak if we leave focused for > leave_reset_seconds
        if (
            self._focused_started_at is not None
            and self._state != PresenceState.PRESENT_FOCUSED
            and (now - self._state_entered_at) > self.thresholds.leave_reset_seconds
        ):
            self._focused_started_at = None

    @property
    def state(self) -> PresenceState:
        return self._state

    @property
    def focused_session_start(self) -> float | None:
        """When the current focused streak started, or None if not focused."""
        return self._focused_started_at

    # -- posture --------------------------------------------------------------

    def compute_posture(self, det: Detection) -> PostureState:
        """Classify the user's sitting posture from a single Detection.

        Compares the current face geometry against the calibrated baseline.
        Returns PostureState.UNKNOWN if no face OR no baseline yet.
        """
        if det.face_box is None or det.face_center_norm is None:
            return PostureState.UNKNOWN
        if not self.baseline.calibrated_at:
            return PostureState.UNKNOWN

        th = self.thresholds
        bl = self.baseline

        # 1. Position drift
        dx = det.face_center_norm[0] - bl.face_center_x
        dy = det.face_center_norm[1] - bl.face_center_y
        x_drift = abs(dx)
        y_drift = abs(dy)

        # 2. Size ratio change — larger = closer (slouching forward)
        if bl.face_size_ratio > 0 and det.face_size_ratio is not None:
            size_ratio = det.face_size_ratio / bl.face_size_ratio
        else:
            size_ratio = 1.0

        # 3. Pitch / yaw
        pitch = det.pitch_deg or 0.0
        yaw = det.yaw_deg or 0.0

        # Priority order matters: head_down takes precedence over slouching
        # because if you're looking down, your face may also be a bit closer.
        if pitch > th.posture_pitch_down:
            return PostureState.HEAD_DOWN
        if size_ratio > 1.0 + th.posture_size_close:
            return PostureState.SLOUCHING
        if size_ratio < 1.0 - th.posture_size_close:
            return PostureState.LEAN_BACK
        if x_drift > th.posture_x_drift or y_drift > th.posture_y_drift:
            return PostureState.LEANING
        if abs(yaw) > th.posture_yaw_turned:
            return PostureState.LEANING
        return PostureState.GOOD

    def posture_score(self, state: PostureState) -> float:
        """0..100 score: 100 = perfect, 0 = unknown/away.

        Used for daily report / progress charts.
        """
        return {
            PostureState.GOOD: 100.0,
            PostureState.HEAD_DOWN: 40.0,
            PostureState.SLOUCHING: 50.0,
            PostureState.LEANING: 65.0,
            PostureState.LEAN_BACK: 70.0,
            PostureState.UNKNOWN: 0.0,
        }.get(state, 0.0)


__all__ = [
    "Detection",
    "PresenceSignals",
    "PresenceState",
    "PostureState",
    "SittingDetector",
    "CAMERA_OK",
    "CAMERA_NO_DEVICE",
    "CAMERA_PERMISSION_DENIED",
    "CAMERA_IN_USE",
]