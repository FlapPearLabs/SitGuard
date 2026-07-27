# SitGuard — Design Document

**Date:** 2026-06-11
**Status:** Approved for MVP implementation
**Author:** Mavis (assistant) + user

---

## 1. Problem Statement

Modern knowledge workers spend 8–12 hours daily at a computer. Prolonged sitting
is a well-documented health hazard (cardiovascular disease, cervical spondylosis,
metabolic syndrome). Existing tools fall into two camps:

- **Timer-only** (Workrave, BreakTimer): light, no camera, but cannot tell if
  the user is actually sitting at the desk — a forgotten laptop still counts.
- **Camera + posture** (Posturr, hunchback-detection): focused on bad posture,
  not accumulated sitting time.

**SitGuard's angle:** the *primary* signal is "are you actually sitting in front
of the computer right now?" Secondary signal is "are you sitting well?" The
moment you leave the desk, the timer pauses.

## 2. Goals & Non-Goals

### Goals
- Single-screen Windows app that runs in the background, uses the built-in
  webcam, and breaks your streak when you've been sitting too long.
- Two install paths:
  - `pip install sitguard` for Python users — `sitguard` command.
  - Single `.exe` (~25 MB after UPX) for everyone else — double-click and run.
- Daily / weekly sitting report (longest streak, rest compliance rate).
- Personal baseline calibration on first run (handles different heights /
  desk setups).
- Open source, MIT licensed, GitHub-ready.

### Non-Goals (MVP)
- Cross-platform (macOS / Linux) — Windows-only for v1.
- Detailed posture coaching (head / shoulder / spine angle) — only basic
  "are you facing the screen" check.
- Multi-user / cloud sync — strictly local, single machine.
- Webcam recording / streaming — frames are processed in-memory and dropped.
- Screen lock / forced rest (no `wjbgis/Sedentary-reminder`-style aggressive
  blocking) — SitGuard *suggests*, it does not *force*.

## 3. Target User

| Persona | Workflow | Why SitGuard |
|---------|----------|--------------|
| Solo developer / writer | Long focused sessions | Gentle accountability |
| Remote team member | Wants a "health buddy" | Privacy-respecting, local |
| Open-source contributor | Wants to tinker | MIT, clean code, GitHub-friendly |

## 4. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    sitguard (entry)                       │
└────────────────┬─────────────────────────┬───────────────┘
                 │                         │
        ┌────────▼─────────┐      ┌────────▼────────┐
        │  detector.py     │      │   ui.py          │
        │  (vision core)   │◄────►│  (tray / popup)  │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
        ┌────────▼─────────┐      ┌────────▼────────┐
        │  session.py      │      │   config.py     │
        │  (state machine) │◄────►│  (TOML config)  │
        └────────┬─────────┘      └─────────────────┘
                 │
        ┌────────▼─────────┐      ┌─────────────────┐
        │  stats.py        │      │  cli.py         │
        │  (JSON records)  │      │  (pip entry)    │
        └──────────────────┘      └─────────────────┘
```

### 4.1 Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.11+ | Richest CV ecosystem |
| Vision | OpenCV 5.0 headless + YuNet (228 KB) + MoveNet Lightning (~3 MB) | Tiny, fast, no MediaPipe bloat |
| UI | `pywin32` (tray + native dialogs) + `winotify` (toast) | Native Win, no Qt |
| Config | TOML via `tomllib` (stdlib in 3.11) | Git-friendly |
| Stats | JSON files in `%APPDATA%\SitGuard\` | No DB needed |
| Package | `pyproject.toml` + setuptools | Standard |
| Build | PyInstaller + UPX | Single exe |

### 4.2 Why NOT MediaPipe

MediaPipe `solutions` brings ~30 MB of TFLite runtime, graph orchestration,
and graphics dependencies. For SitGuard's needs (face box + 17 body keypoints
once per ~2 seconds), YuNet + MoveNet Lightning under OpenCV5's new DNN
engine deliver equivalent accuracy at a fraction of the footprint.

## 5. Detection Pipeline

### 5.1 Five-Signal Fusion

`is_sitting_at_desk(now)` is a weighted vote across five signals:

| Signal | Source | Weight |
|--------|--------|--------|
| Face present | YuNet detector | 0.30 |
| Face size in frame (proxy for distance) | YuNet bbox | 0.25 |
| Face position stability (no false detection from passers-by) | Tracking variance over 2s window | 0.15 |
| Head yaw/pitch within ±30° / ±25° | YuNet 5 landmarks + `cv2.solvePnP` | 0.20 |
| Recent motion in last 60s | Frame-diff on downsampled ROI | 0.10 |

Threshold = 0.55. Above → `PRESENT_FOCUSED`. Below → `PRESENT_IDLE` or
`NOT_PRESENT` based on which signals fired.

### 5.2 Three-State Machine

```
              ┌──────────────────┐
              │   NOT_PRESENT    │◄────────────────┐
              │  (seat empty)    │                 │
              └──────┬───────────┘                 │
                     │ face_found_for_2s           │ absent_for_5min
                     ▼                             │
              ┌──────────────────┐                 │
              │  PRESENT_IDLE    │                 │
              │  (face found,    │                 │
              │   not facing)    │                 │
              └──────┬───────────┘                 │
                     │ yaw_ok & size_ok for 3s     │
                     ▼                             │
              ┌──────────────────┐                 │
              │ PRESENT_FOCUSED  │─────────────────┘
              │  (counting)      │   size_drift > 30% or face_lost_for_10s
              └──────────────────┘
```

Hysteresis: state transitions require the new condition to hold for N
consecutive seconds (2s, 3s, 10s above). This kills flickering from
single-frame glitches.

### 5.3 Calibration

On first run (and accessible via `sitguard calibrate`):
1. Run detection for 30 seconds.
2. Record baseline: face center (x, y), face size (w/h ratio), yaw, pitch.
3. Save baseline to `config.toml`.
4. Detection thresholds are computed as `baseline ± tolerance`.

This solves the "6'5" person sits far away" vs. "5'2" person sits close"
problem without user-tunable knobs.

## 6. Sitting Timer

```python
# Pseudocode
threshold_minutes = config.threshold      # default 45
break_minutes     = config.break_duration  # default 5

while running:
    state = detector.poll()  # every 2s
    if state == PRESENT_FOCUSED:
        sitting_seconds += 2
    elif state == PRESENT_IDLE:
        sitting_seconds += 0   # paused
    else:  # NOT_PRESENT
        sitting_seconds += 0   # paused; reset if absent > break_minutes

    if sitting_seconds >= threshold_minutes * 60:
        ui.notify_break()
        sitting_seconds = 0
```

After the threshold is reached, SitGuard plays a soft Windows toast and (if
the user enables "remind again in N min") re-arms.

## 7. UI / UX

### 7.1 Tray Icon
- Right-click menu: Pause 30m / Skip this break / Open report / Settings / Quit.
- Icon color reflects state: green (focused), yellow (idle), grey (paused),
  red (camera error).

### 7.2 Break Notification
- Windows toast (small, bottom-right) with default text:
  *"You've been sitting for 45 minutes. Stand up, stretch, look out the
  window for a bit."*
- Click toast → opens report window.
- Right-click → "Snooze 5 min" / "I'm already standing".

### 7.3 Report Window (Notion / Verso minimalist style)
- Single page, three sections:
  1. **Today**: longest sitting streak (mm:ss), breaks taken, rest
     compliance %.
  2. **This week**: bar chart of daily sitting minutes (monospace font,
     SVG-rendered, no JS framework).
  3. **Posture**: average yaw / pitch / face position drift — single
     number, "Posture score 78 / 100".

Design tokens:
- Background `#0a0a0a`
- Surface `#141414`
- Text `#ededed`
- Muted `#8a8a8a`
- Accent (single) `#7c5cff` (electric indigo — used sparingly)
- Font: Geist Sans (UI) + Geist Mono (numbers)
- Corner radius: 12px (cards), 8px (buttons), 0px (data table)

## 8. Persistence

All data in `%APPDATA%\SitGuard\`:
- `config.toml` — settings + baseline
- `sessions/YYYY-MM-DD.json` — daily session log
- `report-cache.json` — precomputed weekly aggregates

No SQLite. JSON is fine at this scale (one record per ~2s ≈ 30k records/day,
~3 MB / day, never grows beyond 30 days via auto-pruning).

## 9. Distribution

### 9.1 Python user
```bash
pip install sitguard
sitguard                # start
sitguard calibrate      # re-run baseline
sitguard report         # open today's report
sitguard --version
```

### 9.2 Windows user
Download `SitGuard.exe` (~25 MB) from GitHub Releases. Double-click to run.
First launch opens a 30-second calibration wizard.

Build pipeline: `python scripts/build.py` → PyInstaller with `--upx-dir` →
artifact in `dist/`.

## 10. Privacy

- **No frames ever written to disk.** Camera frames live in `cv2.Mat` and
  are overwritten on the next frame.
- **No network calls.** SitGuard makes zero outbound connections. Verify
  via `pip download sitguard && grep -r "http" sitguard/` — should be empty.
- **No analytics / telemetry.**
- **Webcam light is the only externally visible signal.** User can disable
  camera access at any time via tray menu.

## 11. Roadmap

| Version | Scope |
|---------|-------|
| **v0.1 (MVP)** | Single user, Windows, sitting timer + toast + daily report |
| v0.2 | Weekly report + posture score + custom reminder messages |
| v0.3 | macOS port (PyObjC native dialogs) |
| v0.4 | Tray-icon live preview (tiny face-mesh overlay) |
| v1.0 | Public release announcement on HN / Product Hunt |

## 12. References

- YuNet: Wu et al., "YuNet: A Tiny Millisecond-level Face Detector" (MIR 2023)
- MoveNet: Google Research, TensorFlow Hub (2021)
- SitPose: Jin et al., arXiv:2412.12216 (2024-12) — ensemble design inspiration
- OpenCV 5.0 release notes (2026-06-07)