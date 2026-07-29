"""Main window: a single 360x440 card with the ring, a status line, and action buttons.

This is what the user sees when they click the tray icon and pick "Show window"
(or it's auto-shown on first run). It deliberately fits in the corner of a
laptop screen — no maximize, no menu bar, just the focused state of "sit guard".
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from ...detector import PostureState
from ..theme import THEME
from ..widgets import RingProgress


class MainWindow(QMainWindow):
    """The single most important window in SitGuard."""

    pauseRequested = Signal(int)        # minutes
    skipRequested = Signal()
    openReportRequested = Signal()
    openSettingsRequested = Signal()
    quitRequested = Signal()

    def __init__(self, tray: QSystemTrayIcon | None = None) -> None:
        super().__init__()
        self._tray = tray

        # Frameless, always-on-top, narrow fixed size
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(360, 480)

        # State
        self._minutes_to_reminder: float = 0.0
        self._threshold_minutes: int = 45
        self._state_label: str = "FOCUSED"

        self._build_ui()
        self._install_drag()

        # Auto-refresh "live" feel — update the ring each second
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

    # -- UI construction -----------------------------------------------------

    def _build_ui(self) -> None:
        # Outer wrapper to enable rounded corners
        outer = QWidget(self)
        outer.setObjectName("outer")
        outer.setStyleSheet(
            f"""
            QWidget#outer {{
                background-color: {THEME.surface};
                border: 1px solid {THEME.border};
                border-radius: {THEME.radius_lg}px;
            }}
            """
        )

        root = QVBoxLayout(outer)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        # --- Header row ---
        header = QHBoxLayout()
        header.setSpacing(8)

        self._title = QLabel("SitGuard")
        self._title.setProperty("role", "title")

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(f"color: {THEME.success}; font-size: 10px;")

        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self._status_dot)

        # --- Ring ---
        ring_holder = QHBoxLayout()
        ring_holder.setAlignment(Qt.AlignCenter)
        self._ring = RingProgress(size=240, ring_width=10)
        self._ring.set_center_text("--:--")
        self._ring.set_center_subtext("to next reminder")
        ring_holder.addWidget(self._ring)
        ring_frame = QFrame()
        ring_frame.setLayout(ring_holder)
        ring_frame.setStyleSheet("background: transparent; border: none;")

        # --- Status card ---
        status_card = QFrame()
        status_card.setProperty("surface", "inset")
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(0)

        self._status_label = QLabel("FOCUSED")
        self._status_label.setProperty("role", "caption")
        self._status_label.setStyleSheet(f"color: {THEME.success};")
        self._state_subtext = QLabel("You're in the zone.")
        self._state_subtext.setProperty("role", "muted")
        status_layout.addWidget(self._status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self._state_subtext)

        # --- Posture line (sits just under the status card) ---
        posture_card = QFrame()
        posture_card.setProperty("surface", "inset")
        posture_layout = QHBoxLayout(posture_card)
        posture_layout.setContentsMargins(16, 10, 16, 10)
        posture_layout.setSpacing(0)

        posture_caption = QLabel("Posture")
        posture_caption.setProperty("role", "caption")
        self._posture_dot = QLabel("●")
        self._posture_dot.setStyleSheet(f"color: {THEME.success}; font-size: 10px;")
        self._posture_value = QLabel("GOOD")
        self._posture_value.setStyleSheet(
            f"font-size: 12px; font-weight: 500; color: {THEME.success};"
        )
        posture_layout.addWidget(posture_caption)
        posture_layout.addStretch(1)
        posture_layout.addWidget(self._posture_value)
        posture_layout.addSpacing(6)
        posture_layout.addWidget(self._posture_dot)

        # --- Actions ---
        actions = QHBoxLayout()
        actions.setSpacing(8)

        btn_pause = QPushButton("Pause")
        btn_pause.setProperty("variant", "secondary")
        btn_pause.clicked.connect(lambda: self.pauseRequested.emit(30))

        btn_skip = QPushButton("Skip")
        btn_skip.setProperty("variant", "ghost")
        btn_skip.clicked.connect(self.skipRequested.emit)

        btn_report = QPushButton("Report")
        btn_report.setProperty("variant", "ghost")
        btn_report.clicked.connect(self.openReportRequested.emit)

        actions.addWidget(btn_pause)
        actions.addWidget(btn_skip)
        actions.addWidget(btn_report)

        root.addLayout(header)
        root.addWidget(ring_frame, 1, alignment=Qt.AlignCenter)
        root.addWidget(status_card)
        root.addWidget(posture_card)
        root.addLayout(actions)

        self.setCentralWidget(outer)

    # -- Dragging (frameless window) -----------------------------------------

    def _install_drag(self) -> None:
        self._drag_pos: tuple[int, int] | None = None

    def mousePressEvent(self, event) -> None:  # noqa: D401
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: D401
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: D401
        self._drag_pos = None

    # -- Public API ----------------------------------------------------------

    def update_state(
        self,
        state_label: str,
        state_subtext: str,
        dot_color: str,
        minutes_to_reminder: float,
        threshold_minutes: int,
        posture_label: str = "GOOD",
        posture_color: str = THEME.success,
    ) -> None:
        """Update everything in one shot — called by the daemon loop."""
        self._state_label = state_label
        self._state_subtext.setText(state_subtext)
        self._status_label.setText(state_label)
        self._status_label.setStyleSheet(f"color: {dot_color};")
        self._status_dot.setStyleSheet(f"color: {dot_color}; font-size: 10px;")
        self._minutes_to_reminder = minutes_to_reminder
        self._threshold_minutes = threshold_minutes
        self._posture_label = posture_label
        self._posture_dot.setStyleSheet(f"color: {posture_color}; font-size: 10px;")
        self._posture_value.setText(posture_label)
        self._posture_value.setStyleSheet(
            f"font-size: 12px; font-weight: 500; color: {posture_color};"
        )
        # Subtle posture ring color change
        self._ring.set_colors(
            track=THEME.surface_3,
            progress=posture_color if posture_label != "GOOD" else THEME.accent,
        )
        self._refresh_ring()
        self._refresh_state()

    def _refresh_ring(self) -> None:
        m = int(self._minutes_to_reminder)
        s = int((self._minutes_to_reminder - m) * 60)
        self._ring.set_center_text(f"{m:02d}:{s:02d}")
        # Progress: 0 = full ring (just started), 1 = done
        if self._threshold_minutes > 0:
            frac = max(0.0, min(1.0, 1.0 - self._minutes_to_reminder / self._threshold_minutes))
        else:
            frac = 0.0
        self._ring.set_progress_animated(frac)

    def _refresh_state(self) -> None:
        m = self._minutes_to_reminder
        if m > 30:
            sub = "Plenty of time."
        elif m > 10:
            sub = "Getting closer."
        elif m > 1:
            sub = "Wrap up soon."
        else:
            sub = "Time to stand up."
        # Only override if state is FOCUSED; otherwise let caller drive
        if self._state_label == "FOCUSED":
            self._state_subtext.setText(sub)

    def _on_tick(self) -> None:
        # Decrement the visible time once per second for the "live" feel
        if self._minutes_to_reminder > 0:
            self._minutes_to_reminder = max(0.0, self._minutes_to_reminder - 1 / 60)
            self._refresh_ring()
            self._refresh_state()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: D401
        # Hide instead of quit — tray handles real exit
        event.ignore()
        self.hide()