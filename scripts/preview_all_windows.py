"""Preview all 4 main Qt windows side-by-side. Saves a screenshot."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRectF, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sitguard.config import Config  # noqa: E402
from sitguard.devices import DeviceList, CameraDevice  # noqa: E402
from sitguard.qt import apply_app_theme  # noqa: E402
from sitguard.qt.windows import (  # noqa: E402
    FirstRunWizard,
    MainWindow,
    ReportWindow,
    SettingsWindow,
)


def main() -> int:
    app = QApplication(sys.argv)
    apply_app_theme(app)

    cfg = Config()

    # --- Main window ---
    main_win = MainWindow()
    main_win.update_state(
        state_label="FOCUSED",
        state_subtext="You're in the zone.",
        dot_color="#30d158",
        minutes_to_reminder=23.5,
        threshold_minutes=45,
    )
    main_win.show()
    main_win.move(40, 40)

    # --- Settings ---
    settings = SettingsWindow(
        cfg,
        cameras=[
            "USB2.0 HD UVC WebCam",
            "Integrated Camera (disabled)",
        ],
    )
    settings.show()
    settings.move(440, 40)

    # --- Report ---
    report = ReportWindow()
    report.show()
    report.move(40, 580)

    # --- First-run wizard ---
    devices = DeviceList(
        devices=[
            CameraDevice(
                index=0, name="USB2.0 HD UVC WebCam",
                is_default=True, resolution=(1280, 720), status="ok",
            ),
        ],
        privacy_blocked=False,
    )
    wizard = FirstRunWizard(cfg, devices)
    wizard.show()
    wizard.move(960, 40)

    out_dir = Path(__file__).parent
    state = {"captured": 0}

    def grab(widget, name):
        pix: QPixmap = widget.grab()
        path = out_dir / f"preview_{name}.png"
        pix.save(str(path), "PNG")
        print(f"Saved {path} ({pix.width()}x{pix.height()})")
        state["captured"] += 1
        if state["captured"] >= 4:
            QTimer.singleShot(100, app.quit)

    QTimer.singleShot(1200, lambda: grab(main_win, "main"))
    QTimer.singleShot(1500, lambda: grab(settings, "settings"))
    QTimer.singleShot(1800, lambda: grab(report, "report"))
    QTimer.singleShot(2100, lambda: grab(wizard, "wizard"))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())