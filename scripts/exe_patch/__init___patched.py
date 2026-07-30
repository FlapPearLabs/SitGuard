"""SitGuard — webcam-based sedentary reminder for Windows."""

__version__ = "0.1.0"
__all__ = ["__version__"]


def _install_camera_backend_fix():
    """Globally patch cv2.VideoCapture: DSHOW-first + warmup + non-black check.

    v4 changes (startup speed):
      * backend cache: the backend that worked last time is stored in
        %LOCALAPPDATA%\\SitGuard\\backend_cache.json and tried first with a
        1-frame fast check — near-zero probe time on healthy machines.
      * warmup reduced 8 -> 3 frames, plus a 3-second hard timeout per
        backend, so a firmware-blanked camera no longer stalls startup for
        15-20 s.
      * leaner logging: probe results + selected backend + first 5 runtime
        frames only. Log: %LOCALAPPDATA%\\SitGuard\\camera_debug.log
    """
    import io
    import json
    import os
    import sys
    import time
    import traceback

    # ---------- paths ----------
    try:
        _app_dir = os.path.join(
            os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or ".",
            "SitGuard",
        )
        os.makedirs(_app_dir, exist_ok=True)
        _log_path = os.path.join(_app_dir, "camera_debug.log")
        _cache_path = os.path.join(_app_dir, "backend_cache.json")
    except Exception:
        _log_path = None
        _cache_path = None

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

    _camlog("=== camera fix v4 active === cv2=%s python=%s"
            % (getattr(cv2, "__version__", "?"), sys.version.split()[0]))

    # ---------- backend cache ----------
    def _cache_load():
        try:
            if _cache_path and os.path.isfile(_cache_path):
                with io.open(_cache_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if isinstance(d, dict):
                    return d
        except Exception:
            pass
        return {}

    def _cache_save(d):
        try:
            if _cache_path:
                with io.open(_cache_path, "w", encoding="utf-8") as f:
                    json.dump(d, f)
        except Exception:
            pass

    def _cache_get(index):
        v = _cache_load().get(str(index))
        return v if isinstance(v, int) else None

    def _cache_put(index, backend):
        d = _cache_load()
        d[str(index)] = int(backend)
        _cache_save(d)

    def _cache_drop(index):
        d = _cache_load()
        if str(index) in d:
            del d[str(index)]
            _cache_save(d)

    # ---------- capture proxy: logs first few runtime frames ----------
    class _CapProxy(object):
        def __init__(self, cap, tag):
            object.__setattr__(self, "_sg_cap", cap)
            object.__setattr__(self, "_sg_tag", tag)
            object.__setattr__(self, "_sg_n", 0)

        def read(self, *a, **kw):
            cap = object.__getattribute__(self, "_sg_cap")
            n = object.__getattribute__(self, "_sg_n")
            ok, frame = cap.read(*a, **kw)
            if n < 5:
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

    # ---------- warmup validation (3 frames max, 3 s hard timeout) ----------
    def _frames_ok(cap, backend_name, warm=3, timeout=3.0):
        deadline = time.monotonic() + timeout
        best = -1.0
        for i in range(warm):
            if time.monotonic() > deadline:
                _camlog("  %s TIMEOUT after %d frame(s) (%.1fs budget)"
                        % (backend_name, i, timeout))
                return False
            try:
                grabbed, frame = cap.read()
            except Exception as exc:
                _camlog("  %s warmup#%d read raised: %r" % (backend_name, i, exc))
                return False
            if not grabbed or frame is None:
                continue
            try:
                mean = float(frame.mean())
            except Exception:
                mean = -1.0
            if mean > best:
                best = mean
            if mean > 1.0:
                _camlog("  %s VALIDATED warmup#%d mean=%.1f" % (backend_name, i, mean))
                return True
        _camlog("  %s REJECTED (best_mean=%.1f after %d frames)"
                % (backend_name, best, warm))
        return False

    def _try_backend(index, backend, warm, timeout):
        bname = _backend_names.get(backend, str(backend))
        try:
            cap = _orig(index, backend) if backend is not None else _orig(index)
        except Exception as exc:
            _camlog("  %s constructor raised: %r" % (bname, exc))
            return None
        try:
            if cap.isOpened() and _frames_ok(cap, bname, warm=warm, timeout=timeout):
                return cap
            cap.release()
        except Exception as exc:
            _camlog("  %s validation raised: %r\n%s"
                    % (bname, exc, traceback.format_exc()))
            try:
                cap.release()
            except Exception:
                pass
        return None

    # ---------- patched open ----------
    def _open(*args, **kwargs):
        if kwargs or len(args) != 1 or not isinstance(args[0], int):
            return _orig(*args, **kwargs)
        index = args[0]
        t0 = time.monotonic()

        # 1) cached backend: 1-frame fast path
        cached = _cache_get(index)
        if cached is not None:
            bname = _backend_names.get(cached, str(cached))
            _camlog("cam%d cache hit: %s — fast check" % (index, bname))
            cap = _try_backend(index, cached, warm=1, timeout=2.0)
            if cap is not None:
                _camlog("cam%d SELECTED %s via cache (%.2fs)"
                        % (index, bname, time.monotonic() - t0))
                return _CapProxy(cap, "cam%d/%s" % (index, bname))
            _camlog("cam%d cache stale (%s failed) — full probe" % (index, bname))
            _cache_drop(index)

        # 2) full probe chain: DSHOW -> MSMF -> ANY
        _camlog("cam%d full probe: DSHOW -> MSMF -> ANY" % index)
        order = []
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            val = getattr(cv2, name, None)
            if val is not None and val not in order:
                order.append(val)
        for backend in order:
            bname = _backend_names.get(backend, str(backend))
            cap = _try_backend(index, backend, warm=3, timeout=3.0)
            if cap is not None:
                _camlog("cam%d SELECTED %s (%.2fs) — cached for next start"
                        % (index, bname, time.monotonic() - t0))
                _cache_put(index, backend)
                return _CapProxy(cap, "cam%d/%s" % (index, bname))

        _camlog("cam%d ALL BACKENDS FAILED (%.2fs) — default open"
                % (index, time.monotonic() - t0))
        return _CapProxy(_orig(index), "cam%d/DEFAULT" % index)

    cv2.VideoCapture = _open
    cv2._sitguard_dshow_patch = True


try:
    _install_camera_backend_fix()
except Exception:
    pass
