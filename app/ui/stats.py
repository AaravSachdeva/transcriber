"""Stats screen: time saved as the headline, the rest in a sentence, and a chart of
words dictated per day.

QtCharts ships in the PySide6 wheel (via PySide6-Addons), so the chart costs no extra
dependency.
"""

from __future__ import annotations

from datetime import date

from PySide6.QtCharts import (
    QBarSeries, QBarSet, QCategoryAxis, QChart, QChartView, QValueAxis,
)
from PySide6.QtCore import Qt, QMargins
from PySide6.QtGui import QPalette, QPainter, QPen
from PySide6.QtWidgets import QFrame, QVBoxLayout, QWidget

from . import theme
from ..store import Store, TYPING_WPM


def _pretty_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


class StatsScreen(QWidget):
    def __init__(self, store: Store) -> None:
        super().__init__()
        self._store = store

        title = theme.label("Stats", "PageTitle")
        self._saved = theme.label("", "Hero")
        saved_caption = theme.label(
            f"saved versus typing the same words at {TYPING_WPM:.0f} WPM", "Caption")
        self._summary = theme.label("", wrap=True)
        chart_title = theme.label("Words per day, last 30 days", "SectionTitle")

        self._chart = QChart()
        self._chart.setBackgroundVisible(False)
        self._chart.setPlotAreaBackgroundVisible(False)
        self._chart.legend().setVisible(False)
        self._chart.setMargins(QMargins(0, 0, 0, 0))

        self._chart_view = QChartView(self._chart)
        self._chart_view.setRenderHint(QPainter.Antialiasing)
        self._chart_view.setFrameShape(QFrame.NoFrame)
        self._chart_view.viewport().setAutoFillBackground(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(4)
        layout.addWidget(title)
        layout.addSpacing(16)
        layout.addWidget(self._saved)
        layout.addWidget(saved_caption)
        layout.addSpacing(12)
        layout.addWidget(self._summary)
        layout.addSpacing(28)
        layout.addWidget(chart_title)
        layout.addWidget(self._chart_view, 1)

        self.reload()

    def reload(self) -> None:
        stats = self._store.stats()
        self._saved.setText(_pretty_duration(stats.seconds_saved))
        if stats.count:
            self._summary.setText(
                f"{stats.words:,} words across {stats.count:,} dictations, spoken at "
                f"{stats.speaking_wpm:.0f} WPM. {stats.llm_share:.0%} were long enough "
                "to go through the LLM."
            )
        else:
            self._summary.setText("Nothing dictated yet. Hold right Ctrl in any app and speak.")
        self._rebuild_chart()

    def _rebuild_chart(self) -> None:
        """Colours are read from the palette on every rebuild, so a light/dark switch
        shows up the next time this screen is opened."""
        palette = self.palette()
        dim = palette.color(QPalette.PlaceholderText)
        rule = palette.color(QPalette.Mid)

        daily = self._store.daily_words(days=30)
        self._chart.removeAllSeries()
        for axis in list(self._chart.axes()):
            self._chart.removeAxis(axis)

        bars = QBarSet("Words")
        bars.setColor(palette.color(QPalette.Accent))
        bars.setBorderColor(palette.color(QPalette.Accent))
        bars.append([words for _day, words in daily])
        series = QBarSeries()
        series.setBarWidth(0.7)
        series.append(bars)
        self._chart.addSeries(series)

        # Bars sit at x = 0..n-1. Label today and every seventh day back; a category
        # axis would squeeze every label into one bar's width and elide it.
        x_axis = QCategoryAxis()
        x_axis.setRange(-0.5, len(daily) - 0.5)
        x_axis.setLabelsPosition(QCategoryAxis.AxisLabelsPositionOnValue)
        for i in reversed(range(len(daily) - 1, -1, -7)):  # QCategoryAxis wants ascending
            d = date.fromisoformat(daily[i][0])
            x_axis.append(f"{d:%b} {d.day}", i)
        x_axis.setLabelsColor(dim)
        x_axis.setGridLineVisible(False)
        x_axis.setLinePen(QPen(rule))

        y_axis = QValueAxis()
        y_axis.setLabelFormat("%d")
        y_axis.setRange(0, max(max(words for _d, words in daily) * 1.15, 10))
        y_axis.setTickCount(4)
        y_axis.setLabelsColor(dim)
        y_axis.setGridLinePen(QPen(rule, 1, Qt.DotLine))
        y_axis.setLineVisible(False)

        self._chart.addAxis(x_axis, Qt.AlignBottom)
        self._chart.addAxis(y_axis, Qt.AlignLeft)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)
