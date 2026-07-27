# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SitGuard — Qt-based UI edition.

Two important things to know:
  1. We exclude most of PySide6's optional modules (QML, WebEngine, PDF,
     Multimedia, etc.) to keep the binary small.
  2. After `pyinstaller sitguard.spec`, run `python scripts/strip_qt.py`
     to remove Qt translations and unused plugins. This is what brings
     the binary from ~120 MB down to ~50 MB.
"""
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 (provided by PyInstaller)
MODELS_DIR = ROOT / "models"

a = Analysis(
    [str(ROOT / "sitguard" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(MODELS_DIR / "face_detection_yunet_2023mar.onnx"), "models"),
        (str(ROOT / "sitguard" / "assets"), "assets"),
    ] if (MODELS_DIR / "face_detection_yunet_2023mar.onnx").exists() else [
        (str(ROOT / "sitguard" / "assets"), "assets"),
    ],
    hiddenimports=[
        # Qt
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # === Stdlib bloat ===
        "tkinter",
        "test",
        "unittest",
        "pydoc",
        "doctest",
        "lib2to3",
        "email",
        "xml",
        "xmlrpc",
        "pdb",
        "pipes",
        "pty",
        "tty",
        "curses",
        "curses.panel",
        "readline",
        "rlcompleter",
        "dbm",
        "sqlite3",
        "ensurepip",
        "venv",
        "zipfile",
        "tarfile",
        "py_compile",
        "compileall",
        "multiprocessing",
        "concurrent",
        "asyncio",
        "wsgiref",
        "html",
        "http",
        "urllib",
        "logging.config",
        "pstats",
        "cProfile",
        "profile",

        # === 3rd-party bloat ===
        "matplotlib",
        "pandas",
        "numpy.tests",
        "scipy",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "pystray",
        "PIL",
        "winotify",

        # === PySide6 modules we don't use ===
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtBluetooth",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtNetwork",
        "PySide6.QtNetworkAuth",
        "PySide6.QtNfc",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtPositioning",
        "PySide6.QtPrintSupport",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickControls2",
        "PySide6.QtQuickWidgets",
        "PySide6.QtRemoteObjects",
        "PySide6.QtScxml",
        "PySide6.QtSensors",
        "PySide6.QtSerialBus",
        "PySide6.QtSerialPort",
        "PySide6.QtSql",
        "PySide6.QtSvg",
        "PySide6.QtSvgWidgets",
        "PySide6.QtTest",
        "PySide6.QtTextToSpeech",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngine",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebSockets",
        "PySide6.QtXml",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="SitGuard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "sitguard.ico") if (ROOT / "assets" / "sitguard.ico").exists() else None,
)