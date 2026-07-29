"""SitGuard — webcam-based sedentary reminder for Windows."""

__version__ = "0.1.0"
__all__ = ["__version__"]


def _install_camera_backend_fix():
    """Globally patch cv2.VideoCapture: DSHOW-first + warmup + non-black check.

    On Windows, OpenCV defaults to MSMF which opens many UVC webcams
    "successfully" but yields all-black frames. Every bare
    cv2.VideoCapture(index) call in this package (e.g. detector.open_camera)
    gets rerouted: try CAP_DSHOW first, validate with warmup frames and a
    non-zero brightness check, then fall back to MSMF / ANY.

    Full diagnostic logging goes to:
        %LOCALAPPDATA%\\SitGuard\\camera_debug.log
    """
    import io
    import os
    import sys
    import time
    import traceback

    # ---------- logging ----------
    try:
        _log_dir = os.path.join(
            os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or ".",
            "SitGuard",
        )
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, "camera_debug.log")
    except Exception:
        _log_path = None

    def _camlog(msg):
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        try:
            if _log_path:
                with io.open(_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass
        try:
            sys.stderr.write("[SitGuard-cam] " + line + "\n")
        except Exception:
            pass

    try:
        import cv2
    except Exception as exc:
        _camlog("FATAL: cv2 import failed: %r" % (exc,))
        return

    if getattr(cv2, "_sitguard_dshow_patch", False):
        return

    _orig = cv2.VideoCapture
    _backend_names = {}
    for _n in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY", "CAP_VFW"):
        _v = getattr(cv2, _n, None)
        if _v is not None:
            _backend_names[_v] = _n

    _camlog("=== camera fix active === cv2=%s python=%s"
            % (getattr(cv2, "__version__", "?"), sys.version.split()[0]))

    # ---------- capture proxy: logs runtime frame brightness ----------
    class _CapProxy(object):
        def __init__(self, cap, tag):
            object.__setattr__(self, "_sg_cap", cap)
            object.__setattr__(self, "_sg_tag", tag)
            object.__setattr__(self, "_sg_n", 0)

        def read(self, *a, **kw):
            cap = object.__getattribute__(self, "_sg_cap")
            n = object.__getattribute__(self, "_sg_n")
            ok, frame = cap.read(*a, **kw)
            if n < 30 or n % 300 == 0:
                try:
                    mean = float(frame.mean()) if (ok and frame is not None) else -1.0
                except Exception:
                    mean = -2.0
                _camlog("%s read#%d ok=%s mean=%.1f shape=%s"
                        % (object.__getattribute__(self, "_sg_tag"), n, ok,
                           mean, getattr(frame, "shape", None)))
            object.__setattr__(self, "_sg_n", n + 1)
            return ok, frame

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_sg_cap"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_sg_cap"), name, value)

    # ---------- warmup validation ----------
    def _frames_ok(cap, backend_name, warm=8):
        grabbed_any = False
        best = -1.0
        for i in range(warm):
            try:
                grabbed, frame = cap.read()
            except Exception as exc:
                _camlog("  %s warmup#%d read raised: %r" % (backend_name, i, exc))
                return False
            if not grabbed or frame is None:
                _camlog("  %s warmup#%d grabbed=False" % (backend_name, i))
                continue
            grabbed_any = True
            try:
                mean = float(frame.mean())
            except Exception:
                mean = -1.0
            if mean > best:
                best = mean
            _camlog("  %s warmup#%d mean=%.1f shape=%s"
                    % (backend_name, i, mean, getattr(frame, "shape", None)))
            if mean > 1.0:
                _camlog("  %s VALIDATED (mean=%.1f > 1.0)" % (backend_name, mean))
                return True
        _camlog("  %s REJECTED (grabbed_any=%s best_mean=%.1f)"
                % (backend_name, grabbed_any, best))
        return False

    # ---------- patched open ----------
    def _open(*args, **kwargs):
        if kwargs or len(args) != 1 or not isinstance(args[0], int):
            _camlog("passthrough VideoCapture(args=%r kwargs=%r)" % (args, kwargs))
            return _orig(*args, **kwargs)
        index = args[0]
        _camlog("intercept VideoCapture(%d) — trying DSHOW first" % index)
        order = []
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            val = getattr(cv2, name, None)
            if val is not None and val not in order:
                order.append(val)
        for backend in order:
            bname = _backend_names.get(backend, str(backend))
            try:
                cap = _orig(index, backend)
            except Exception as exc:
                _camlog("  %s constructor raised: %r" % (bname, exc))
                continue
            try:
                opened = cap.isOpened()
                _camlog("  %s isOpened=%s" % (bname, opened))
                if opened and _frames_ok(cap, bname):
                    _camlog("SELECTED backend=%s for index %d" % (bname, index))
                    return _CapProxy(cap, "cam%d/%s" % (index, bname))
                cap.release()
            except Exception as exc:
                _camlog("  %s validation raised: %r\n%s"
                        % (bname, exc, traceback.format_exc()))
                try:
                    cap.release()
                except Exception:
                    pass
        _camlog("ALL BACKENDS FAILED for index %d — falling back to default open"
                % index)
        return _CapProxy(_orig(index), "cam%d/DEFAULT" % index)

    cv2.VideoCapture = _open
    cv2._sitguard_dshow_patch = True


try:
    _install_camera_backend_fix()
except Exception:
    pass
