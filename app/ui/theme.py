"""Type ramp and state colours.

Controls are drawn by Qt's native windows11 style, which follows the system light/dark
mode and accent colour. A style sheet rule on a control would force Qt's generic
fallback renderer for it, so the rules here only touch labels, by object name.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel

# Tray and status colours, one per Controller.State.
IDLE = "#8b919e"
RECORDING = "#e0435c"
PROCESSING = "#3ec98a"
ERROR = "#e0a343"

# Windows type ramp: title 28, subtitle 20, display 40, caption 12.
_DISPLAY = 'font-family: "Segoe UI Variable Display", "Segoe UI"; font-weight: 600;'

STYLESHEET = f"""
#PageTitle {{ {_DISPLAY} font-size: 28px; }}
#SectionTitle {{ {_DISPLAY} font-size: 20px; }}
#Hero {{ {_DISPLAY} font-size: 40px; }}
#Caption {{ font-size: 12px; color: palette(placeholder-text); }}
"""


def apply(app: QApplication) -> None:
    # Qt takes the 9pt Win32 message font; Windows 11's own body text is 14px.
    font = app.font()
    font.setFamilies(["Segoe UI Variable Text", "Segoe UI"])
    font.setPointSizeF(10.5)
    app.setFont(font)
    base = app.style().name()
    app.setStyleSheet(STYLESHEET)

    # On a light/dark switch, Qt 6.11 updates the app palette but leaves existing
    # widgets painted in the old one. Setting the style again repolishes them all.
    def restyle() -> None:
        app.setStyle(base)
        app.setStyleSheet(STYLESHEET)

    app.styleHints().colorSchemeChanged.connect(lambda _: QTimer.singleShot(0, restyle))


def label(text: str, role: str = "", wrap: bool = False) -> QLabel:
    """A QLabel in one of the ramp roles above: PageTitle, SectionTitle, Hero, Caption."""
    widget = QLabel(text)
    widget.setObjectName(role)
    widget.setWordWrap(wrap)
    return widget
