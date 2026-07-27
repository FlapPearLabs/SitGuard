"""Smoke test: boot Qt, create main window, take a screenshot.

Run with: python scripts/preview_main_window.py
Closes after 3 seconds. Saves screenshot to scripts/main_window.png so we
can see it without a display server.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

# Allow `python scripts/preview.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sitguard.qt import apply_app_theme  # noqa: E402
from sitguard.qt.windows import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    apply_app_theme(app)

    win = MainWindow()
    win.show()

    # Simulate the daemon pushing some state
    win.update_state(
        state_label="FOCUSED",
        state_subtext="You're in the zone.",
        dot_color="#30d158",
        minutes_to_reminder=23.5,
        threshold_minutes=45,
    )

    # Position in the top-right corner of the primary screen
    screen = app.primaryScreen().availableGeometry()
    win.move(screen.right() - win.width() - 24, screen.top() + 24)

    out = Path(__file__).parent / "main_window.png"

    def take_screenshot_and_quit() -> None:
        pixmap = win.grab()
        pixmap.save(str(out), "PNG")
        print(f"Saved {out} ({pixmap.width()}x{pixmap.height()})")
        app.quit()

    QTimer.singleShot(1500, take_screenshot_and_quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())