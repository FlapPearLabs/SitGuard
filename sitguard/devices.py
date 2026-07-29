"""Camera device discovery on Windows.

Two strategies:
  1. **PowerShell + WMI** — lists every PnP imaging device, including the
     device path and friendly name. Works without opening the camera.
  2. **OpenCV probe** — opens indices 0..N until one succeeds. Maps each
     successful index to a friendly name (best-effort match to WMI list).

Win10/11 privacy settings must allow desktop apps to access the camera
(Settings → Privacy & security → Camera → "Let desktop apps access your
camera") before any of this works. We surface a clear error otherwise.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# WMI class GUIDs (per MSDN)
_CAMERA_CLASS_GUID = "{ca3e7ab9-b4c3-4ae6-8251-579ef933890f}"


# --- Capture backend selection (the fix for the black-screen bug) ----------
#
# On Windows, OpenCV's default capture backend is Media Foundation (CAP_MSMF).
# MSMF is notorious for OPENING successfully but returning all-zero (black)
# frames for a large class of UVC webcams — exactly the "camera opens but the
# screen is black" symptom. The far more compatible backend for UVC cameras
# is DirectShow (CAP_DSHOW). So we always try DSHOW first and only fall back to
# MSMF / auto if DSHOW cannot open the device.
#
# This was the root cause of SitGuard rendering a black preview on machines
# that otherwise work fine in other camera apps: open_camera used
# `cv2.VideoCapture(index)` with no backend flag, so OpenCV silently picked
# the broken MSMF path.


def _backend_order() -> list[tuple[str, int | None]]:
    """Preferred capture backends, most-compatible first.

    Returns a list of (name, flag) where flag is the cv2 backend constant or
    None for the auto/default path.
    """
    import cv2

    order: list[tuple[str, int | None]] = []
    dshow = getattr(cv2, "CAP_DSHOW", None)
    if dshow is not None:
        order.append(("DirectShow", dshow))
    msmf = getattr(cv2, "CAP_MSMF", None)
    if msmf is not None:
        order.append(("MediaFoundation", msmf))
    order.append(("Auto", None))
    return order


def _try_open_camera(index: int, backend_flag: int | None):
    """Open a camera with one specific backend. Returns the capture or None."""
    import cv2

    try:
        cap = (
            cv2.VideoCapture(index, backend_flag)
            if backend_flag is not None
            else cv2.VideoCapture(index)
        )
    except Exception:  # noqa: BLE001 - any failure means "this backend can't open it"
        return None
    if not cap.isOpened():
        cap.release()
        return None
    return cap



@dataclass
class CameraDevice:
    """A discovered video capture device."""

    index: int                        # OpenCV index — pass to cv2.VideoCapture(idx)
    name: str                         # "Integrated Camera", "USB2.0 HD UVC WebCam", etc.
    pnp_device_id: str = ""           # full PnP path (e.g. USB\\VID_...)
    status: str = "unknown"           # "ok", "disabled", "in_use", "permission_denied"
    is_default: bool = False          # first camera in the system
    resolution: tuple[int, int] | None = None  # probed (width, height)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.resolution is not None:
            d["resolution"] = list(self.resolution)
        return d


@dataclass
class DeviceList:
    """All discovered cameras + diagnostic info."""

    devices: list[CameraDevice] = field(default_factory=list)
    wmi_error: str | None = None
    privacy_blocked: bool = False     # Windows privacy setting is off
    probe_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [d.to_dict() for d in self.devices],
            "wmi_error": self.wmi_error,
            "privacy_blocked": self.privacy_blocked,
            "probe_summary": self.probe_summary,
        }


# --- Strategy 1: WMI / PowerShell enumeration ------------------------------


def _wmi_enumerate_raw() -> list[dict[str, str]]:
    """Run PowerShell to enumerate imaging devices via WMI. Returns list of
    {Name, DeviceID, Status, ClassGUID} dicts. Empty list on failure."""
    if not IS_WINDOWS:
        return []
    pwsh = shutil.which("powershell") or shutil.which("pwsh")
    if not pwsh:
        return []

    script = (
        "Get-PnpDevice -Class Camera -ErrorAction SilentlyContinue"
        " | Select-Object -Property FriendlyName, InstanceId, Status"
        " | ConvertTo-Json -Depth 2"
    )
    try:
        result = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        log.warning("PowerShell WMI enumeration failed: %s", e)
        return []

    if result.returncode != 0 or not result.stdout.strip():
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = [data]
    return [
        {
            "name": d.get("FriendlyName", ""),
            "device_id": d.get("InstanceId", ""),
            "status": d.get("Status", "Unknown"),
        }
        for d in data
        if isinstance(d, dict)
    ]


# --- Strategy 2: OpenCV probe ------------------------------------------------


def _probe_opencv_indices(max_index: int = 5) -> list[int]:
    """Try opening cameras at indices 0..max_index. Return those that succeed."""
    import cv2  # local import — package stays importable without opencv installed

    # OpenCV writes scary ERROR lines to stderr when an index is out of range
    # or when the obsensor backend probes nonexistent hardware. That's expected
    # and not actionable — suppress it so our CLI output stays clean.
    import os
    import contextlib

    # 1) Quiet OpenCV's own logger
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)  # OpenCV 5+
    except AttributeError:
        # Older OpenCV: just disable the INFO logs
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass

    # 2) Also redirect fd-level stderr in case the backend bypasses Python logging
    devnull = open(os.devnull, "w")
    ok_indices: list[int] = []
    try:
        for i in range(max_index + 1):
            with contextlib.redirect_stderr(devnull):
                cap = None
                for _name, flag in _backend_order():
                    cap = _try_open_camera(i, flag)
                    if cap is not None:
                        break
                opened = cap is not None
                grabbed = False
                if opened:
                    grabbed, _ = cap.read()
                    cap.release()
            if opened and grabbed:
                ok_indices.append(i)
    finally:
        devnull.close()
    return ok_indices


def _probe_resolution(index: int) -> tuple[int, int] | None:
    import cv2

    cap = _try_open_camera(index, getattr(cv2, "CAP_DSHOW", None))
    if cap is None:
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h) if w and h else None


def open_capture(index: int = 0, warmup_frames: int = 5):
    """Open the camera with the most-compatible backend and verify it streams.

    Tries backends in order (DirectShow -> Media Foundation -> auto), reads a
    few warm-up frames, and returns the capture only if it yields at least one
    non-black frame. This avoids the classic OpenCV-on-Windows black-screen bug
    where the default MSMF backend opens successfully but returns all-zero
    frames for many UVC webcams.

    Returns the opened ``cv2.VideoCapture`` (caller must release it) or ``None``
    if no backend can produce a usable frame.
    """
    import cv2
    import numpy as np

    for _name, flag in _backend_order():
        cap = _try_open_camera(index, flag)
        if cap is None:
            continue
        # Warm up: some cameras emit black frames for the first few reads.
        good = False
        for _ in range(warmup_frames):
            ok, frame = cap.read()
            if ok and frame is not None and float(np.mean(frame)) > 1.0:
                good = True
                break
        if good:
            return cap
        cap.release()
    return None


def open_camera(index: int = 0, **kwargs):
    """Convenience wrapper around :func:`open_capture` with a stable name.

    The detection / UI layer should call THIS instead of constructing
    ``cv2.VideoCapture`` directly, so the backend fallback always applies and
    the black-screen bug cannot return.
    """
    return open_capture(index, **kwargs)



# --- Privacy setting detection ---------------------------------------------


def check_privacy_allowed() -> bool:
    """Probe whether Windows currently allows desktop apps to access the camera.

    Best-effort: checks the registry value that backs the "Let desktop apps
    access your camera" toggle. Returns True if the value is 1, False if 0,
    None (treated as allowed) if the key is missing.
    """
    if not IS_WINDOWS:
        return True
    pwsh = shutil.which("powershell")
    if not pwsh:
        return True

    script = (
        "$path = 'HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\"
        "CapabilityAccessManager\\ConsentStore\\webcam\\NonPackaged';"
        "if (Test-Path $path) {"
        "  Get-ItemProperty -Path $path -Name 'Value' -ErrorAction SilentlyContinue"
        "  | Select-Object -ExpandProperty Value"
        "} else { 'missing' }"
    )
    try:
        result = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return True
    out = result.stdout.strip().lower()
    if out == "deny" or out == "0":
        return False
    return True


# --- Public entry points ----------------------------------------------------


def discover_cameras(max_index: int = 5) -> DeviceList:
    """Enumerate every video capture device accessible to SitGuard.

    Strategy:
      1. Pull PnP device names via PowerShell + WMI (no camera access needed).
      2. Probe OpenCV indices 0..max_index to find ones that actually open.
      3. Match probe results to WMI names by order (index 0 = first device).

    Returns a DeviceList. If nothing is found, `devices` is empty and
    `privacy_blocked` may be True (telling the user what to fix).
    """
    wmi_list = _wmi_enumerate_raw()
    opencv_indices = _probe_opencv_indices(max_index)

    devices: list[CameraDevice] = []
    for i, idx in enumerate(opencv_indices):
        # Best-effort: pair by position. WMI may have extra entries (disabled
        # devices, virtual cameras, etc.) so we just take the i-th match.
        wmi = wmi_list[i] if i < len(wmi_list) else {}
        devices.append(
            CameraDevice(
                index=idx,
                name=wmi.get("name") or f"Camera {idx}",
                pnp_device_id=wmi.get("device_id", ""),
                status="ok" if wmi.get("status") in {"OK", ""} else wmi.get("status", "unknown"),
                is_default=(idx == opencv_indices[0]),
                resolution=_probe_resolution(idx),
            )
        )

    privacy_ok = check_privacy_allowed()
    summary = (
        f"Found {len(devices)} camera(s) via OpenCV; "
        f"{len(wmi_list)} imaging device(s) via WMI; "
        f"privacy={'allowed' if privacy_ok else 'BLOCKED'}"
    )

    return DeviceList(
        devices=devices,
        wmi_error=None if wmi_list or not IS_WINDOWS else "PowerShell WMI query returned nothing",
        privacy_blocked=not privacy_ok,
        probe_summary=summary,
    )


def format_device_list(devices: DeviceList) -> str:
    """Human-friendly multi-line summary of discovered devices."""
    if not devices.devices:
        msg = "No cameras detected."
        if devices.privacy_blocked:
            msg += (
                "\n\nWindows is blocking desktop apps from accessing the camera.\n"
                "To fix this:\n"
                "  1. Open Settings → Privacy & security → Camera\n"
                "  2. Turn on 'Camera access'\n"
                "  3. Turn on 'Let apps access your camera'\n"
                "  4. Turn on 'Let desktop apps access your camera'\n"
                "  5. Re-run SitGuard"
            )
        elif devices.wmi_error:
            msg += f"\nWMI query failed: {devices.wmi_error}"
        return msg

    lines = [f"{len(devices.devices)} camera(s) detected:"]
    for d in devices.devices:
        res = f" {d.resolution[0]}x{d.resolution[1]}" if d.resolution else ""
        star = " *" if d.is_default else "  "
        lines.append(f"  {star}[{d.index}] {d.name}{res}  ({d.status})")
    if devices.privacy_blocked:
        lines.append(
            "\nWARNING: Windows privacy settings currently BLOCK desktop apps from "
            "using the camera. SitGuard will fail until you allow it in Settings."
        )
    return "\n".join(lines)


def save_device_list(devices: DeviceList, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(devices.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_device_list(path: Path) -> DeviceList | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    devs = []
    for d in data.get("devices", []):
        res = d.get("resolution")
        devs.append(
            CameraDevice(
                index=int(d["index"]),
                name=d.get("name", f"Camera {d['index']}"),
                pnp_device_id=d.get("pnp_device_id", ""),
                status=d.get("status", "unknown"),
                is_default=bool(d.get("is_default", False)),
                resolution=tuple(res) if res else None,
            )
        )
    return DeviceList(
        devices=devs,
        wmi_error=data.get("wmi_error"),
        privacy_blocked=bool(data.get("privacy_blocked", False)),
        probe_summary=data.get("probe_summary", ""),
    )


__all__ = [
    "CameraDevice",
    "DeviceList",
    "discover_cameras",
    "format_device_list",
    "check_privacy_allowed",
    "save_device_list",
    "load_device_list",
    "open_capture",
    "open_camera",
]