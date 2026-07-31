"""SitGuard entry point.

Three subcommands:
  sitguard              → run the daemon (default)
  sitguard calibrate    → re-run the 30s baseline calibration
  sitguard report       → open today's report in the default browser
  sitguard --version    → print version
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import Config, load, save, sessions_dir, app_data_dir
from .detector import (
    CAMERA_IN_USE,
    CAMERA_NO_DEVICE,
    CAMERA_OK,
    CAMERA_PERMISSION_DENIED,
    PresenceState,
    SittingDetector,
)
from .devices import (
    CameraDevice,
    discover_cameras,
    format_device_list,
    save_device_list,
    load_device_list,
)
from .session import SessionManager
from .stats import Event, append_event, daily_report, weekly_report
from .ui import (
    TrayController,
    notify,
    open_camera_privacy_settings,
    open_report,
    show_camera_error_dialog,
    show_camera_setup_guide,
    show_first_run_wizard,
    verify_open_source,
)

log = logging.getLogger("sitguard")


# --- Subcommands ------------------------------------------------------------


def cmd_run(cfg: Config) -> int:
    """Main daemon loop."""
    # First-run: show the wizard so the user understands what to do
    if not cfg.baseline.calibrated_at:
        log.info("First run detected — opening setup wizard")
        try:
            show_first_run_wizard()
        except FileNotFoundError as e:
            log.warning("Wizard asset missing: %s", e)

    # Build CameraDevice from config, falling back to a synthesized one
    camera = CameraDevice(
        index=cfg.camera_index,
        name=cfg.camera_name or f"Camera {cfg.camera_index}",
        is_default=(cfg.camera_index == 0),
    )
    detector = SittingDetector(cfg.baseline, cfg.thresholds, camera=camera)
    if not detector.open_camera():
        log.error("Could not open webcam: %s", detector.camera_error)
        notify("SitGuard", detector.camera_error)
        # Always open the HTML guide — it has full step-by-step instructions.
        try:
            show_camera_setup_guide()
        except FileNotFoundError as e:
            log.warning("Guide asset missing: %s", e)
        # For permission issues, also pop the Win settings page directly
        if detector.camera_status == CAMERA_PERMISSION_DENIED:
            show_camera_error_dialog(
                detector.camera_error
                + "\n\nClick OK to open the Camera privacy settings."
            )
            open_camera_privacy_settings()
        else:
            show_camera_error_dialog(detector.camera_error)
        return 1

    session = SessionManager(cfg.thresholds)
    session.start()

    paused_until = 0.0
    skip_next = False
    running = True

    def on_pause(minutes: int) -> None:
        nonlocal paused_until
        paused_until = time.time() + minutes * 60
        append_event(Event(time.time(), "pause", {"minutes": minutes}))
        log.info("Paused for %d min", minutes)

    def on_skip() -> None:
        nonlocal skip_next
        skip_next = True
        log.info("Skip next reminder")

    def on_open_report() -> None:
        try:
            open_report(daily_report(session.stats), weekly_report())
        except Exception as e:  # pragma: no cover
            log.warning("Could not open report: %s", e)

    def on_quit() -> None:
        nonlocal running
        running = False
        log.info("Quit requested")

    tray = TrayController(
        cfg,
        on_pause,
        on_skip,
        on_open_report,
        on_quit,
        on_show_privacy_guide=lambda: _safe_call(show_camera_setup_guide),
        on_about=lambda: _safe_call(_show_about),
    )
    tray.start()

    last_state = PresenceState.NOT_PRESENT
    log.info("SitGuard running. Press Ctrl+C to stop.")
    try:
        while running:
            now = time.time()
            if now < paused_until:
                session.tick(PresenceState.NOT_PRESENT)
                time.sleep(1.0)
                continue

            frame = detector.read_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            detection = detector.process(frame)
            state = detector.state

            # Log focus transitions
            if state != last_state:
                if state == PresenceState.PRESENT_FOCUSED:
                    append_event(Event(now, "focus_started", {}))
                elif last_state == PresenceState.PRESENT_FOCUSED:
                    append_event(Event(now, "focus_ended", {}))
                last_state = state

            session.tick(state)

            # Break reminder
            if session.should_remind(state):
                if skip_next:
                    skip_next = False
                else:
                    body = cfg.reminder_body.format(
                        minutes=cfg.thresholds.sitting_threshold_minutes
                    )
                    notify(cfg.reminder_title, body)
                    append_event(Event(time.time(), "reminder_fired", {}))
                    session.mark_reminded()

            # Update tray tooltip
            mins = session.minutes_to_reminder(state)
            tray.update_tooltip(f"SitGuard — {mins:.0f} min to next reminder")

            time.sleep(2.0)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        detector.close_camera()
        tray.stop()
        append_event(Event(time.time(), "session_stopped", {}))

    return 0


def cmd_calibrate(cfg: Config) -> int:
    """30-second calibration wizard."""
    detector = SittingDetector(cfg.baseline, cfg.thresholds)
    if not detector.open_camera():
        print("ERROR: Could not open webcam.", file=sys.stderr)
        return 1

    print("Sit comfortably in front of the screen.")
    print("Look at the screen like you normally would.")
    print("Calibration will run for 30 seconds. Press Ctrl+C to abort.\n")

    samples: list[dict] = []
    start = time.time()
    try:
        while time.time() - start < 30.0:
            frame = detector.read_frame()
            if frame is None:
                time.sleep(0.5)
                continue
            det = detector.process(frame)
            if det.face_center_norm is not None and det.face_size_ratio is not None:
                samples.append(
                    {
                        "cx": det.face_center_norm[0],
                        "cy": det.face_center_norm[1],
                        "size": det.face_size_ratio,
                        "yaw": det.yaw_deg or 0.0,
                        "pitch": det.pitch_deg or 0.0,
                    }
                )
                remaining = int(30 - (time.time() - start))
                print(f"\r  capturing... {remaining:2d}s ({len(samples)} samples) ", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 1
    finally:
        detector.close_camera()

    print()
    if len(samples) < 10:
        print("ERROR: Too few valid samples. Try again with better lighting.", file=sys.stderr)
        return 1

    # Median across samples (robust to outliers)
    import statistics

    cfg.baseline.face_center_x = statistics.median(s["cx"] for s in samples)
    cfg.baseline.face_center_y = statistics.median(s["cy"] for s in samples)
    cfg.baseline.face_size_ratio = statistics.median(s["size"] for s in samples)
    cfg.baseline.yaw_deg = statistics.median(s["yaw"] for s in samples)
    cfg.baseline.pitch_deg = statistics.median(s["pitch"] for s in samples)
    cfg.baseline.calibrated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    save(cfg)
    print(f"Calibration saved → {Path(cfg_path_str := str(_config_path(cfg))).__class__.__name__}")
    print()
    print("Baseline:")
    print(f"  Face center: ({cfg.baseline.face_center_x:.2f}, {cfg.baseline.face_center_y:.2f})")
    print(f"  Face size:   {cfg.baseline.face_size_ratio:.3f}")
    print(f"  Yaw:         {cfg.baseline.yaw_deg:+.1f}°")
    print(f"  Pitch:       {cfg.baseline.pitch_deg:+.1f}°")
    return 0


def _config_path(cfg: Config) -> Path:
    """Tiny helper for the calibrate print message."""
    from .config import config_path
    return config_path()


def cmd_report(_cfg: Config) -> int:
    """Render and open the HTML report."""
    rep = daily_report()
    week = weekly_report()
    open_report(rep, week)
    print(f"Report opened in browser. ({rep['focused_minutes']:.0f} min focused today)")
    return 0


def cmd_cameras(cfg: Config) -> int:
    """Discover available cameras, show them, and persist the choice.

    Updates `cfg.camera_index` / `cfg.camera_name` to the default camera
    if nothing was set previously.
    """
    print("Scanning for cameras...\n")
    devices = discover_cameras()
    print(format_device_list(devices))
    print()

    cache_path = app_data_dir() / "cameras.json"
    save_device_list(devices, cache_path)

    if not devices.devices:
        print("Nothing to select. Fix the issue above and re-run.")
        if devices.privacy_blocked:
            print("Opening Camera privacy settings...")
            open_camera_privacy_settings()
        return 1

    # If user hasn't picked yet, default to the first one
    if not cfg.camera_name or any(d.index == cfg.camera_index for d in []):
        chosen = devices.devices[0]
        cfg.camera_index = chosen.index
        cfg.camera_name = chosen.name
        save(cfg)
        print(f"Default camera set to: [{chosen.index}] {chosen.name}")
        print(f"Saved → {cfg.camera_name}")
    else:
        print(f"Current selection: [{cfg.camera_index}] {cfg.camera_name}")

    if devices.privacy_blocked:
        print("\nNOTE: privacy is currently BLOCKED. Opening Settings...")
        open_camera_privacy_settings()
    return 0


def cmd_verify(_cfg: Config) -> int:
    """Scan the local SitGuard source for any network call patterns.

    Helps users independently confirm the open-source privacy claims.
    """
    src_dir = Path(__file__).resolve().parent
    print(f"Scanning {src_dir} for network call patterns...\n")
    result = verify_open_source(src_dir)

    print(f"  Files scanned: {result['file_count']}")
    if result["clean"]:
        print("  ✓ No suspicious imports found (urllib, requests, http, etc.).")
        print("\nSitGuard source is verified to make zero outbound network calls.")
        return 0
    else:
        print(f"  ✗ Found {len(result['findings'])} suspicious import(s):\n")
        for f in result["findings"]:
            print(f"    {f['file']}:{f['line']}  {f['pattern']}")
            print(f"      → {f['snippet']}")
        return 1


def cmd_gui(_cfg: Config) -> int:
    """Launch the Qt-based UI (Mac-style dark theme)."""
    from .cli_gui import main as gui_main
    return gui_main()


def _config_path(cfg: Config) -> Path:
    """Tiny helper used only by the calibrate command for printing the path."""
    from .config import config_path
    return config_path()


def _safe_call(fn) -> None:
    """Run a UI callback, logging any exception instead of crashing the tray."""
    try:
        fn()
    except Exception as e:  # pragma: no cover
        log.exception("UI callback failed: %s", e)


def _show_about() -> None:
    """Open the camera setup guide — it doubles as the 'About / privacy' page."""
    from .ui import open_source_on_github

    show_camera_setup_guide()
    open_source_on_github()


# --- argparse ---------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sitguard",
        description="Open-source webcam-based sedentary reminder for Windows.",
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("run", help="Run the daemon (default)").set_defaults(func=cmd_run)
    sub.add_parser("calibrate", help="Recalibrate baseline posture").set_defaults(func=cmd_calibrate)
    sub.add_parser("report", help="Open today's report in your browser").set_defaults(func=cmd_report)
    sub.add_parser(
        "cameras", help="Discover cameras and check Windows privacy settings"
    ).set_defaults(func=cmd_cameras)
    sub.add_parser(
        "verify",
        help="Scan the installed source for any network call patterns",
    ).set_defaults(func=cmd_verify)
    sub.add_parser(
        "gui",
        help="Launch the Qt-based UI (Mac-style dark theme)",
    ).set_defaults(func=cmd_gui)

    p.add_argument("--version", action="version", version=f"sitguard {__version__}")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    cfg = load()
    handler = getattr(args, "func", None)
    if handler is None:
        # Default: run the daemon
        handler = cmd_run
    return handler(cfg)


if __name__ == "__main__":
    sys.exit(main())