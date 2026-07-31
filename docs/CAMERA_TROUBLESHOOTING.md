# Camera Troubleshooting — when SitGuard shows "no face detected"

This document records an actual debugging session where the webcam was
detected by Windows but delivered black frames. If you hit the same
issue, this is the checklist.

## TL;DR

If `python -c "import cv2; cap=cv2.VideoCapture(0); ok, f=cap.read(); print(f.mean() if ok else None)"`
prints `0.0` (or any near-zero value) when you can see yourself in
front of the camera, **your webcam UVC stream is broken**, not the
software.

## What we did (in order)

| # | Step | Result | Conclusion |
|---|------|--------|------------|
| 1 | `Get-PnpDevice -Class Camera` | 2 cameras, both `OK` | Hardware enumerated |
| 2 | `Get-Process wechat/zoom/teams` | All clear | No app lock |
| 3 | `HKCU:\...\webcam\NonPackaged` | `Value=Allow` | Privacy OK |
| 4 | `cap.isOpened()` | True | Driver accepts the handle |
| 5 | `cap.read()` on index 0 | Returns all-zero BGR array | **No data in stream** |
| 6 | Try `CAP_DSHOW`, `CAP_MSMF`, `CAP_V4L2` | MSMF refused, others all black | Backend not the issue |
| 7 | `Disable-PnpDevice` + `Enable-PnpDevice` on the camera | No change | Driver reset didn't help |
| 8 | Disable + enable both USB 3.0/3.2 host controllers | USB bus fully reset, still black | USB subsystem OK |
| 9 | Open Microsoft Camera app | Also shows black | **OS-level UVC stream is dead** |
| 10 | `shutdown /s /t 0` (full power off) | _not yet tried_ | _only remaining fix_ |

## What this means

The webcam sensor has either:

1. **Entered a "suspended" UVC state** that requires a full power cycle
   (not a software restart) to recover. Common on laptop built-in
   webcams after another app crashed while holding the device.
2. **Hardware failure** — cable / sensor dead. Replace or service.

## Fixes, in order of likelihood

1. **Full shutdown** (`shutdown /s /t 0` in admin PowerShell, or hold
   the power button 10 seconds). Power back on. This clears the
   UVC firmware suspended state on most laptops.
2. **BIOS reset**: Reboot into BIOS, save & exit without changes.
   Reboot back. This reinitializes the embedded controller.
3. **If the laptop has a "webcam kill switch"** in the BIOS, make
   sure it's enabled. Some Lenovo / Dell models have this.
4. **Driver reinstall** in Device Manager → Camera → Uninstall
   device → restart. Windows will reinstall the UVC driver.
5. **External USB webcam** as a last-resort workaround. SitGuard's
   `discover_cameras()` will pick it up automatically.
6. **Service the laptop** if it's still under warranty — the sensor
   cable may be loose from the motherboard (common after drops).

## Verifying the fix

After any of the above, run:

```powershell
cd D:\SitGuard
python scripts\dump_frame.py
```

You should see:

```
Frame shape: (720, 1280, 3), dtype: uint8
Mean brightness (0=black, 255=white): ~80-200
Min: ~10, Max: ~250
```

If `Mean brightness` is in that range, the camera is healthy. If
still 0, the hardware is dead — try a different webcam.

Then run the full dryrun to get real face geometry data:

```powershell
python scripts\dryrun_detector.py
```

That output will let you tune `Thresholds.posture_*` for your actual
camera + seating distance (current defaults are conservative).

## Diagnostic scripts (in `scripts/`)

| Script | Purpose |
|--------|---------|
| `dump_frame.py` | Single frame, brightness check |
| `dump_frame_all_backends.py` | Try DSHOW / MSMF / ANY / V4L2 |
| `dryrun_detector.py` | 20s real YuNet pipeline test |
| `dryrun_sensitive.py` | 15s with `score_threshold=0.2` (very forgiving) |
| `simulate_posture.py` | Synthetic face geometry, validates `compute_posture()` logic without a camera |

## When it's not the camera

If `dump_frame.py` returns a non-black frame but `dryrun_detector.py`
still says "no face" for 20s straight:

- Check lighting — YuNet fails in near-dark
- Check focus / lens cap (you'd be surprised)
- Try `score_threshold=0.3` instead of `0.6` in `SittingDetector`

## Status (2026-07-16)

`discover_cameras()` reports 1 camera (USB2.0 HD UVC WebCam, 1280×720),
privacy allowed. `read()` returns all-zero BGR frame. **UVC stream is
broken at the OS level.** User elected to pause work on camera features
until a working environment is available. Posture detection was
validated on synthetic data (23/23 scenarios pass).

The rest of SitGuard — UI, settings, stats, break reminders, system
tray, single-file 42.3 MB exe — is unaffected.
