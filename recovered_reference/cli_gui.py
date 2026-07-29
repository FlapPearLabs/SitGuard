"""`sitguard gui` — launch the Qt UI directly.

Useful for development and for users who prefer the rich UI over the
plain tray daemon. Internally creates a QApplication, wires up the
detector + session manager, and runs the event loop.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ..config import Config, load, save
from ..detector import (
    CAMERA_IN_USE,
    CAMERA_NO_DEVICE,
    CAMERA_OK,
    CAMERA_PERMISSION_DENIED,
    PostureState,
    PresenceState,
    SittingDetector,
)
from ..devices import CameraDevice, discover_cameras, open_camera_privacy_settings
from ..qt import apply_app_theme
from ..qt.tray import SitGuardApp
from ..qt.windows import BreakOverlay
from ..session import SessionManager
from ..stats import Event, append_event
from . import __version__

log = logging.getLogger("sitguard.gui")


def main() -> int:
    cfg = load()

    app = QApplication(sys.argv)
    app.setApplicationName("SitGuard")
    app.setApplicationVersion(__version__)
    apply_app_theme(app)

    # Probe camera
    camera = CameraDevice(
        index=cfg.camera_index,
        name=cfg.camera_name or f"Camera {cfg.camera_index}",
        is_default=(cfg.camera_index == 0),
    )
    detector = SittingDetector(cfg.baseline, cfg.thresholds, camera=camera)
    if not detector.open_camera():
        log.error("Could not open webcam: %s", detector.camera_error)
        if detector.camera_status == CAMERA_PERMISSION_DENIED:
            open_camera_privacy_settings()
        # Keep the UI running so the user can fix it from the tray menu

    session = SessionManager(cfg.thresholds)

    sg = SitGuardApp(
        cfg,
        on_pause=lambda m: _on_pause(cfg, m),
        on_skip=lambda: _on_skip(cfg),
        on_quit=lambda: app.quit(),
        on_settings_saved=lambda new_cfg: _on_settings_saved(new_cfg),
        cameras_for_settings=[camera.name] if camera.name else [],
    )

    # First-run wizard
    if not cfg.baseline.calibrated_at:
        devices = discover_cameras()
        sg.show_first_run_wizard(devices)

    # Periodic detector tick
    paused_until = 0.0
    last_state = PresenceState.NOT_PRESENT
    last_posture = PostureState.UNKNOWN
    last_posture_started = time.time()

    def tick() -> None:
        nonlocal last_state, paused_until, last_posture, last_posture_started
        now = time.time()
        if now < paused_until:
            session.tick(PresenceState.NOT_PRESENT, PostureState.UNKNOWN)
            return
        frame = detector.read_frame()
        if frame is None:
            return
        det = detector.process(frame)
        state = detector.state
        posture = detector.compute_posture(det)

        if state != last_state:
            if state == PresenceState.PRESENT_FOCUSED:
                append_event(Event(now, "focus_started", {}))
            elif last_state == PresenceState.PRESENT_FOCUSED:
                append_event(Event(now, "focus_ended", {}))
            last_state = state

        if posture != last_posture:
            append_event(Event(now, "posture_state", {"state": posture.value}))
            last_posture = posture
            last_posture_started = now

        session.tick(state, posture)

        if session.should_remind(state):
            overlay = BreakOverlay(cfg.reminder_title, cfg.reminder_body)
            overlay.snoozed.connect(lambda m: _on_pause(cfg, m))
            overlay.dismissed.connect(lambda: _on_skip(cfg))
            overlay.tookBreak.connect(lambda: _on_skip(cfg))
            overlay.show_fullscreen()
            session.mark_reminded()

        if session.should_remind_posture(posture):
            _show_posture_reminder(posture)
            session.mark_posture_reminded()

        mins = session.minutes_to_reminder(state)
        sg.update_state(
            state_label=_state_label(state),
            state_subtext=_state_subtext(state, mins),
            dot_color=_dot_color(state),
            minutes_to_reminder=mins,
            threshold_minutes=cfg.thresholds.sitting_threshold_minutes,
            posture_label=_posture_label(posture),
            posture_color=_posture_color(posture),
        )

    timer = QTimer()
    timer.setInterval(2000)
    timer.timeout.connect(tick)
    timer.start()

    return app.exec()


# --- Helpers -----------------------------------------------------------------


def _on_pause(cfg: Config, minutes: int) -> None:
    from ..stats import append_event, Event as _Event
    append_event(_Event(time.time(), "pause", {"minutes": minutes}))
    log.info("Paused for %d min", minutes)


def _on_skip(cfg: Config) -> None:
    log.info("Skip next reminder")


def _on_settings_saved(new_cfg: Config) -> None:
    save(new_cfg)
    log.info("Settings saved")


def _state_label(state: PresenceState) -> str:
    return {
        PresenceState.NOT_PRESENT: "AWAY",
        PresenceState.PRESENT_IDLE: "IDLE",
        PresenceState.PRESENT_FOCUSED: "FOCUSED",
    }.get(state, "")


def _state_subtext(state: PresenceState, minutes_remaining: float) -> str:
    if state == PresenceState.NOT_PRESENT:
        return "Camera can't see you."
    if state == PresenceState.PRESENT_IDLE:
        return "Looking away from screen."
    if minutes_remaining > 30:
        return "You're in the zone."
    if minutes_remaining > 10:
        return "Getting closer."
    if minutes_remaining > 1:
        return "Wrap up soon."
    return "Time to stand up."


def _dot_color(state: PresenceState) -> str:
    if state == PresenceState.NOT_PRESENT:
        return "#86868b"
    if state == PresenceState.PRESENT_IDLE:
        return "#ff9f0a"
    return "#30d158"


def _posture_label(posture: PostureState) -> str:
    return {
        PostureState.GOOD: "GOOD",
        PostureState.HEAD_DOWN: "HEAD DOWN",
        PostureState.SLOUCHING: "SLOUCHING",
        PostureState.LEANING: "LEANING",
        PostureState.LEAN_BACK: "LEAN BACK",
        PostureState.UNKNOWN: "—",
    }.get(posture, "—")


def _posture_color(posture: PostureState) -> str:
    return {
        PostureState.GOOD: "#30d158",
        PostureState.HEAD_DOWN: "#ff9f0a",
        PostureState.SLOUCHING: "#ff9f0a",
        PostureState.LEANING: "#ffd60a",
        PostureState.LEAN_BACK: "#5ac8fa",
        PostureState.UNKNOWN: "#86868b",
    }.get(posture, "#86868b")


def _show_posture_reminder(posture: PostureState) -> None:
    """Pop a small overlay when the user has been in bad posture too long."""
    from .qt.windows import BreakOverlay

    titles = {
        PostureState.HEAD_DOWN: "Head down",
        PostureState.SLOUCHING: "You're slouching",
        PostureState.LEANING: "Sit up straight",
        PostureState.LEAN_BACK: "Come closer",
    }
    bodies = {
        PostureState.HEAD_DOWN: "Looking down for a while. Lift your head, soften your shoulders.",
        PostureState.SLOUCHING: "You leaned closer to the screen. Sit back — your spine will thank you.",
        PostureState.LEANING: "Your head is off-center. Straighten up.",
        PostureState.LEAN_BACK: "You drifted back. Sit forward again so your eyes stay level with the screen.",
    }
    title = titles.get(posture, "Posture check")
    body = bodies.get(posture, "Take a breath, realign, and continue.")
    overlay = BreakOverlay(title, body, auto_dismiss_seconds=20)
    overlay.show_fullscreen()


if __name__ == "__main__":
    sys.exit(main())