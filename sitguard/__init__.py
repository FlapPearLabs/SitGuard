"""SitGuard - webcam-based sedentary reminder for Windows.

This package is recovered from the shipped SitGuard.exe: the bulk of the
modules were restored as Python 3.12 bytecode (.pyc) and only the package
init, the entry point and the camera layer (devices.py) are kept as source
so the OpenCV backend fix (CAP_DSHOW-first) is applied.
"""

__version__ = "0.1.0"

__all__ = [
    "config",
    "detector",
    "session",
    "stats",
    "ui",
    "cli",
    "cli_gui",
    "devices",
]
