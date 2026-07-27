# SitGuard

<p align="left">
  <img src="assets/logo.svg" width="64" alt="SitGuard logo">
</p>

<p align="left">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Windows%2010%2F11-blue.svg" alt="Windows 10/11">
  <a href="https://github.com/<owner>/sitguard"><img src="https://img.shields.io/badge/source-GitHub-black.svg" alt="Source on GitHub"></a>
  <img src="https://img.shields.io/badge/network%20calls-zero-brightgreen.svg" alt="Zero network calls">
</p>

**An open-source webcam-based sedentary reminder for Windows.**

> **You don't have to trust us.** SitGuard is MIT-licensed source. Read it,
> grep it, run `sitguard verify` to confirm it makes zero outbound network
> calls, or audit the build yourself. [View source on GitHub →](https://github.com/<owner>/sitguard)

SitGuard lives in your system tray, watches you via the built-in webcam, and
gently nudges you to stand up when you've been sitting too long. The moment
you walk away, the timer pauses. No accounts, no cloud, no telemetry.

```text
   ▄▄▄       ▄▄▄▄▄▄▄
   ██▀▀██▄  ▀▀██▀▀  ██   ← a single ring of attention
      ▀███   ██  ██  ██
   ██▄▄██▀   ██▄▄██▄██
   ▀▀▀▀       ▀▀▀▀▀▀▀
   S I T G U A R D       v0.1.0
```

## Install

### Option A — pip (Python users)

```bash
pip install sitguard
sitguard cameras   # one-time: discover your webcam + check privacy
sitguard
```

Requires Python 3.11+ on Windows 10/11.

### Option B — single executable (everyone else)

1. Download `SitGuard.exe` from the [Releases](../../releases) page (~25 MB).
2. Double-click to run. No installer, no admin rights.
3. First launch runs a 30-second calibration to learn your normal posture.

## Camera discovery & Windows privacy

Windows 10/11 ships with **desktop app camera access disabled by default**.
If `sitguard cameras` shows "No cameras detected" or SitGuard fails to start:

1. Open **Settings → Privacy & security → Camera**
2. Turn on **Camera access**
3. Turn on **Let apps access your camera**
4. Turn on **Let desktop apps access your camera** ← *this is the one that blocks us*
5. Re-run `sitguard cameras`

SitGuard detects this automatically and pops the Settings page directly when
the privacy toggle is off.

`sitguard cameras` enumerates every connected webcam via PowerShell + WMI
(gets friendly names like "Integrated Camera" or "USB2.0 HD UVC WebCam"),
then probes each via OpenCV to confirm it's actually openable, and persists
the choice to `%APPDATA%\SitGuard\config.toml`.

**Webcam shows up in `discover_cameras()` but reads all-black frames?**
That's almost always a broken UVC stream, not SitGuard. See
[docs/CAMERA_TROUBLESHOOTING.md](docs/CAMERA_TROUBLESHOOTING.md) for the
full 10-step diagnostic we ran during development.

## How it works

SitGuard runs a five-signal fusion detector every two seconds:

| Signal | What it measures | Why it matters |
|--------|------------------|----------------|
| Face present | YuNet face detector fires | Someone is in front of the camera |
| Face size | Bbox area vs. frame area | You're not too far / too close |
| Position stability | Variance over 2s window | Not a passer-by triggering false counts |
| Head pose | Yaw / pitch from solvePnP | You're roughly facing the screen |
| Recent motion | Frame-diff in ROI | You're not frozen / camera not stuck |

When the weighted vote crosses 0.70, you're in **focused** state and the
sitting timer ticks. Drop below 0.40 and the timer pauses. After your
configured threshold (default 45 min), a Windows toast pops up:

> *"You've been sitting for 45 minutes. Stand up, stretch, and look out
> the window for a bit."*

The moment you walk away, the timer resets.

## What's in the box

```
SitGuard/
├── sitguard/
│   ├── config.py       # TOML settings + baseline dataclasses
│   ├── detector.py     # YuNet + 5-signal fusion + 3-state machine
│   ├── session.py      # Timer logic + reminder cooldown
│   ├── stats.py        # Daily/weekly JSON event log
│   ├── ui.py           # Windows tray + native toast + HTML report
│   ├── cli.py          # `sitguard run / calibrate / report`
│   └── tests/          # pytest suite (no camera required)
├── docs/plans/         # Design docs
├── models/             # YuNet ONNX (~228 KB)
├── scripts/build.py    # PyInstaller wrapper
├── .github/workflows/  # CI: lint + test + build
└── assets/             # Logo + hero
```

## Design philosophy

- **Local-first.** No frames ever leave your machine. Verify with
  `grep -r "http" sitguard/` — should be empty.
- **Lightweight.** Single ~25 MB exe. YuNet is 228 KB; the Python runtime is
  embedded via PyInstaller. UPX compresses ~40%.
- **No magic.** Every threshold lives in `config.toml`. Override anything.
- **Open-source everything.** MIT licensed. YuNet, OpenCV, MoveNet are all
  permissively licensed upstream.

## Configuration

Config file lives at `%APPDATA%\SitGuard\config.toml`. Generated on first
launch; edit and save while SitGuard is closed.

```toml
[thresholds]
sitting_threshold_minutes = 45
break_minutes = 5
reminder_cooldown_minutes = 5
yaw_tolerance_deg = 30.0
pitch_tolerance_deg = 25.0

[baseline]
face_center_x = 0.5
face_center_y = 0.45
face_size_ratio = 0.25

reminder_title = "Time to stand up"
reminder_body = "You've been sitting for {minutes} minutes. Stand up, stretch, look out the window."
```

## Commands

```bash
sitguard              # run the daemon (default)
sitguard cameras      # discover webcams + check Windows privacy
sitguard calibrate    # re-run 30s baseline wizard
sitguard report       # open today's report in your browser
sitguard verify       # audit the source for network calls
sitguard --version    # 0.1.0
sitguard -v           # verbose (debug) logging
```

## Building from source

```bash
git clone https://github.com/<you>/sitguard.git
cd sitguard
pip install -e ".[build]"

# Place YuNet model in models/
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# Build single-file exe (with UPX on PATH)
python scripts/build.py
# → dist/SitGuard.exe
```

## Privacy

| Data | Where it goes | Retention |
|------|---------------|-----------|
| Camera frames | RAM only (cv2.Mat) | Dropped after detection (~2s) |
| Baseline | `%APPDATA%\SitGuard\config.toml` | Until you reset |
| Daily events | `%APPDATA%\SitGuard\sessions\YYYY-MM-DD.json` | Auto-pruned after 30 days |
| Network | **None.** Zero outbound connections. | n/a |

SitGuard never records video, never sends telemetry, never phones home.

## Verifying it's actually open source

Don't take our word for it — SitGuard ships with a built-in self-audit:

```bash
sitguard verify
# Scanning .../sitguard/ for network call patterns...
#   Files scanned: 8
#   ✓ No suspicious imports found (urllib, requests, http, etc.).
# SitGuard source is verified to make zero outbound network calls.
```

You can also do the same audit yourself with grep:

```bash
# 1. No network libraries
grep -rE "^\s*(import|from)\s+(urllib|requests|http|aiohttp|httpx|websockets|socket)" sitguard/
# → should print nothing

# 2. No image-write / display calls in the detector
grep -nE "imwrite|imshow|VideoWriter" sitguard/detector.py
# → should print nothing (only VideoCapture / read is allowed)

# 3. Check declared dependencies
cat pyproject.toml | grep -A 5 "^dependencies"
# → only opencv-python-headless, pywin32, winotify, tomli
```

The full setup wizard is also available any time from the tray menu
(**Camera setup guide**) — it has the same audits inline with screenshots.

## Roadmap

- **v0.1** — Sitting timer + Windows toast + daily report ← **you are here**
- **v0.2** — Weekly report + posture score + custom reminder messages
- **v0.3** — macOS port (PyObjC)
- **v0.4** — Tray-icon live preview (tiny face-mesh overlay)
- **v1.0** — Public launch (HN, Product Hunt)

## Contributing

PRs welcome. See `CONTRIBUTING.md`. The five-signal fusion in
`sitguard/detector.py` is the most interesting area to play with — try
adding a sixth signal (microphone activity, keyboard cadence, etc.).

## Acknowledgements

- [YuNet](https://github.com/ShiqiYu/libfacedetection) — face detection (228 KB, MIT)
- [OpenCV 5.0](https://opencv.org/) — DNN engine
- [MoveNet](https://tfhub.dev/google/movenet/) — pose estimation (planned for v0.2)
- Inspired by [Posturr](https://www.appinn.com/posturr/), [Workrave](https://www.workrave.org/),
  and every developer's neck pain.

## License

MIT. See `LICENSE`.