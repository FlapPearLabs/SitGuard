# Contributing to SitGuard

Thanks for caring about your neck. So do we.

## Setup

```bash
git clone https://github.com/<you>/sitguard.git
cd sitguard
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e ".[dev]"

# Get the YuNet model
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Workflow

1. Branch off `main`: `git checkout -b feat/your-thing`
2. Make changes. Run tests: `pytest`
3. Lint: `ruff check sitguard scripts`
4. Types: `mypy sitguard`
5. Open a PR.

## Project layout

```
sitguard/
├── config.py     # TOML + dataclasses — pure data, no I/O where possible
├── detector.py   # Vision core — owns no timers / UI
├── session.py    # Pure timer logic — owns no vision / UI
├── stats.py      # JSON event log + report aggregation
├── ui.py         # All Windows-specific stuff lives here
├── cli.py        # argparse entry point
└── tests/        # pytest, no real camera needed
```

## Where to add things

- **New detection signal?** → `sitguard/detector.py::compute_signals`. Update
  the weights in the docstring and add a test.
- **New reminder copy?** → `Config.reminder_body` (template-string) or
  `Config.reminder_title`. Don't hardcode strings in code.
- **New platform (macOS, Linux)?** → Add a backend module under `ui.py`
  (e.g., `_mac.py`), dispatch in `TrayController` / `notify()`. Keep the
  public API identical.
- **New report view?** → Extend `stats.weekly_report()` to return more
  fields, then update `ui._REPORT_HTML`. Don't fork the template — keep one.

## Code style

- Python 3.11+ syntax (`from __future__ import annotations` is required).
- Type hints everywhere. `mypy --strict` is enforced in CI.
- `ruff` line-length 100. Imports sorted.
- No `print()` in library code — use `logging`. `cli.py` is the only file
  that prints to stdout.
- Prefer `pathlib.Path` over `os.path`. Prefer dataclasses over dicts.

## Testing

Tests run without a real camera — they construct `Detection` and
`PresenceSignals` directly. See `tests/test_detector.py` for examples.

```bash
pytest                       # run all
pytest tests/test_session.py # one file
pytest --cov=sitguard        # with coverage
```

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(detector): add microphone activity as a 6th signal
fix(session): cooldown now resets on focus interruption
docs: clarify YuNet license in README
```

## Releasing

Bumps go in `sitguard/__init__.py::__version__`. CI builds and uploads
`SitGuard.exe` to GitHub Releases automatically on `main` pushes.