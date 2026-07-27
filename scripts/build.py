"""Build SitGuard.exe via PyInstaller.

Usage:
    python scripts/build.py

Produces dist/SitGuard.exe. To compress further, install UPX and place
upx.exe on PATH or in venv/Scripts/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "scripts" / "sitguard.spec"
ENTRY = ROOT / "sitguard" / "__main__.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SitGuard.exe")
    parser.add_argument("--onefile", action="store_true", default=True)
    parser.add_argument("--no-upx", action="store_true")
    parser.add_argument("--clean", action="store_true", default=True)
    args = parser.parse_args()

    # Make sure pyinstaller is installed
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found. Install with: pip install pyinstaller", file=sys.stderr)
        return 1

    # Clean previous build artifacts
    if args.clean:
        for d in ("build", "dist"):
            target = ROOT / d
            if target.exists():
                print(f"Cleaning {target}...")
                shutil.rmtree(target, ignore_errors=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        return rc

    out = ROOT / "dist" / "SitGuard.exe"
    if not out.exists():
        print(f"ERROR: expected {out} not found", file=sys.stderr)
        return 1

    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n✓ Built {out} ({size_mb:.1f} MB)")
    if size_mb > 50:
        print(f"⚠ Size {size_mb:.1f} MB exceeds 50 MB target. Consider enabling UPX.")
    return 0


if __name__ == "__main__":
    sys.exit(main())