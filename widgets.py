"""
Widgets for the WTP Degradation Preview GUI.

- FloatParam / IntParam / ChoiceParam / BoolParam: individual parameter editors
- StepCard: one pipeline step (handle, toggle, title, summary, status dot, actions, params)
- PipelinePanel: toolbar + scrollable card list with drag-and-drop reorder
"""

import os
from html import escape

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,
    QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QLabel, QPushButton, QFrame,
    QScrollArea, QSizePolicy, QApplication, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QMimeData, QEvent, QTimer
from PySide6.QtGui import QDrag, QPainter, QPixmap

from schema import SCHEMAS, CATEGORY_OF, CATEGORY_COLORS, build_config, summarize
from add_menu import AddStepMenu
from presets import save_preset, load_preset, to_hcl

_ACCENT = "#5B9DF5"
_PRESET_FILTER = "Pipeline preset (*.json);;All files (*)"


# ──────────────────────────────────────────────
# Individual parameter widgets
#   value / default: current and schema-default value
#   controls: sub-widgets that react to the mouse wheel (the panel guards them)
# ──────────────────────────────────────────────

class FloatParam(QWidget):
    value_changed = Signal()

    def __init__(self, pdef, parent=None):
        super().__init__(parent)
        self._block = False

        lo = pdef["min"]
        hi = pdef["max"]
        step = pdef.get("step", 0.01)
        decimals = pdef.get("decimals", 2)
        self.default = pdef.get("default", lo)

        self._lo = lo
        self._step = step
        ticks = int(round((hi - lo) / step))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, ticks)
        self.slider.setValue(int(round((self.default - lo) / step)))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(self.default)
        self.spin.setFixedWidth(70)
        self.spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin, 0)
        self.controls = (self.slider, self.spin)

        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)

    def _slider_moved(self, v):
        if self._block:
            return
        self._block = True
        val = self._lo + v * self._step
        self.spin.setValue(val)
        self._block = False
        self.value_changed.emit()

    def _spin_changed(self, v):
        if self._block:
            return
        self._block = True
        tick = int(round((v - self._lo) / self._step))
        self.slider.setValue(tick)
        self._block = False
        self.value_changed.emit()

    @property
    def value(self):
        return self.spin.value()

    @value.setter
    def value(self, v):
        self.spin.setValue(float(v))


class IntParam(QWidget):
    value_changed = Signal()

    def __init__(self, pdef, parent=None):
        super().__init__(parent)
        self._block = False

        lo = pdef["min"]
        hi = pdef["max"]
        self.default = pdef.get("default", lo)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(self.default)

        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setValue(self.default)
        self.spin.setFixedWidth(70)
        self.spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin, 0)
        self.controls = (self.slider, self.spin)

        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)

    def _slider_moved(self, v):
        if self._block:
            return
        self._block = True
        self.spin.setValue(v)
        self._block = False
        self.value_changed.emit()

    def _spin_changed(self, v):
        if self._block:
            return
        self._block = True
        self.slider.setValue(v)
        self._block = False
        self.value_changed.emit()

    def set_range(self, lo, hi, default=None):
        """Dynamically update slider/spin range and optionally reset value."""
        self._block = True
        self.slider.setRange(lo, hi)
        self.spin.setRange(lo, hi)
        if default is not None:
            self.slider.setValue(default)
            self.spin.setValue(default)
        else:
            # Clamp current value into new range
            clamped = max(lo, min(hi, self.spin.value()))
            self.slider.setValue(clamped)
            self.spin.setValue(clamped)
        self._block = False
        self.value_changed.emit()

    @property
    def value(self):
        return self.spin.value()

    @value.setter
    def value(self, v):
        self.spin.setValue(int(v))


class ChoiceParam(QComboBox):
    value_changed = Signal()

    def __init__(self, pdef, parent=None):
        super().__init__(parent)
        options = pdef["options"]
        self.addItems([str(o) for o in options])
        self.default = pdef.get("default", options[0])
        idx = options.index(self.default) if self.default in options else 0
        self.setCurrentIndex(idx)
        self.controls = (self,)
        self.currentIndexChanged.connect(lambda _: self.value_changed.emit())

    @property
    def value(self):
        return self.currentText()

    @value.setter
    def value(self, v):
        idx = self.findText(str(v))
        if idx >= 0:
            self.setCurrentIndex(idx)


class BoolParam(QCheckBox):
    value_changed = Signal()

    def __init__(self, pdef, parent=None):
        super().__init__(pdef["label"], parent)
        self.default = pdef.get("default", False)
        self.setChecked(self.default)
        self.controls = ()
        self.toggled.connect(lambda _: self.value_changed.emit())

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, v):
        self.setChecked(bool(v))


_WIDGET_MAP = {
    "float": FloatParam,
    "int": IntParam,
    "choice": ChoiceParam,
    "bool": BoolParam,
}


def _repolish(widget):
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ──────────────────────────────────────────────
# StepCard — one pipeline step
# ──────────────────────────────────────────────

class StepCard(QFrame):
    """Card with a category accent bar, a header row and the step's parameters."""

    changed = Signal()                  # a parameter value changed
    enabled_toggled = Signal()
    request_move = Signal(object, int)  # card, -1 (up) / +1 (down)
    request_duplicate = Signal(object)
    request_remove = Signal(object)

    MIME_TYPE = "application/x-wtp-step-card"

    def __init__(self, schema_key, parent=None):
        super().__init__(parent)
        self.schema_key = schema_key
        self._schema = SCHEMAS[schema_key]
        self._collapsed = False
        self._result_state = ""
        self._summary_text = ""
        self._press_pos = None
        self._press_target = None  # "handle" | "header" | None

        self.setObjectName("stepCard")
        self.setProperty("state", "")
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        color_bar = QFrame()
        color_bar.setObjectName("categoryBar")
        color_bar.setFixedWidth(3)
        color_bar.setStyleSheet(f"background-color: {CATEGORY_COLORS[CATEGORY_OF[schema_key]]};")
        outer.addWidget(color_bar)

        column = QVBoxLayout()
        column.setContentsMargins(8, 6, 8, 8)
        column.setSpacing(8)
        outer.addLayout(column, 1)

        column.addWidget(self._build_header())
        self.content = self._build_params()
        column.addWidget(self.content)

        self.set_result(None)
        self._sync_visibility()
        self._refresh_summary()

    # ── Construction ──

    def _build_header(self):
        self.header = QWidget()
        self.header.setObjectName("cardHeader")
        grid = QGridLayout(self.header)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)

        self.handle = QLabel("\u22EE\u22EE")
        self.handle.setObjectName("dragHandle")
        self.handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self.handle.setToolTip("Drag to reorder")

        self.enable_cb = QCheckBox()
        self.enable_cb.setChecked(True)
        self.enable_cb.setToolTip("Enable / disable this step")
        self.enable_cb.toggled.connect(self._on_enabled_toggled)

        self.title = QLabel()
        self.title.setObjectName("cardTitle")
        self.title.setCursor(Qt.CursorShape.PointingHandCursor)
        self.title.setToolTip("Click to collapse / expand")

        self.status_dot = QLabel()
        self.status_dot.setObjectName("statusDot")
        self.status_dot.setFixedSize(8, 8)

        actions = QWidget()
        actions.setObjectName("cardActions")
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        for glyph, tip, name, slot in (
            ("\u25B2", "Move up", "iconBtn", lambda: self.request_move.emit(self, -1)),
            ("\u25BC", "Move down", "iconBtn", lambda: self.request_move.emit(self, 1)),
            ("\u2750", "Duplicate", "iconBtn", lambda: self.request_duplicate.emit(self)),
            ("\u21BA", "Reset to defaults", "iconBtn", self.reset_values),
            ("\u2715", "Remove (Del)", "deleteBtn", lambda: self.request_remove.emit(self)),
        ):
            btn = QPushButton(glyph)
            btn.setObjectName(name)
            btn.setFixedSize(22, 22)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)

        self.summary = QLabel()
        self.summary.setObjectName("cardSummary")
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        grid.addWidget(self.handle, 0, 0)
        grid.addWidget(self.enable_cb, 0, 1)
        grid.addWidget(self.title, 0, 2)
        grid.addWidget(self.status_dot, 0, 3)
        grid.addWidget(actions, 0, 4)
        grid.addWidget(self.summary, 1, 2, 1, 3)
        grid.setColumnStretch(2, 1)
        return self.header

    def _build_params(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.param_widgets = {}
        self.param_labels = {}
        for pdef in self._schema["params"]:
            widget = _WIDGET_MAP[pdef["type"]](pdef)
            help_text = pdef.get("help", "")
            widget.setToolTip(help_text)
            widget.value_changed.connect(self._on_value_changed)
            self.param_widgets[pdef["key"]] = widget
            if isinstance(widget, BoolParam):  # the checkbox carries its own label
                layout.addWidget(widget)
                continue
            label = QLabel(pdef["label"])
            label.setObjectName("paramLabel")
            label.setWordWrap(True)
            label.setToolTip(help_text)
            self.param_labels[pdef["key"]] = label
            field = QVBoxLayout()
            field.setSpacing(3)
            field.addWidget(label)
            field.addWidget(widget)
            layout.addLayout(field)

        # Dynamic profile dependencies (e.g. codec → quality label and range)
        for pdef in self._schema["params"]:
            profiles = pdef.get("profiles")
            if profiles:
                self.param_widgets[profiles["source"]].currentTextChanged.connect(
                    lambda val, tk=pdef["key"], pm=profiles["map"]: self._apply_profile(tk, pm, val)
                )
        return content

    def _apply_profile(self, target_key, profile_map, source_value):
        """Update a parameter widget when its profile source changes."""
        profile = profile_map.get(source_value)
        if profile is None:
            return
        self.param_widgets[target_key].set_range(profile["min"], profile["max"], profile["default"])
        label = self.param_labels.get(target_key)
        if label is not None and "label" in profile:
            label.setText(profile["label"])

    # ── State ──

    @property
    def enabled(self):
        return self.enable_cb.isChecked()

    def set_collapsed(self, collapsed):
        self._collapsed = collapsed
        self._sync_visibility()

    def get_values(self):
        return {key: w.value for key, w in self.param_widgets.items()}

    def get_config(self):
        return build_config(self.schema_key, self.get_values())

    def get_state(self):
        return {"type": self.schema_key, "enabled": self.enabled,
                "collapsed": self._collapsed, "values": self.get_values()}

    def apply_state(self, step):
        """Restore enabled/collapsed/values; unknown value keys are ignored, missing ones keep defaults."""
        values = step.get("values", {})
        for key, widget in self.param_widgets.items():  # schema order: profile sources first
            if key in values:
                widget.value = values[key]
        self.enable_cb.setChecked(bool(step.get("enabled", True)))
        self.set_collapsed(bool(step.get("collapsed", False)))

    def reset_values(self):
        for widget in self.param_widgets.values():
            widget.value = widget.default

    def set_result(self, step):
        """Show a run result (an object with elapsed_ms / error / error_summary / cached) or idle for None."""
        if step is None:
            self._result_state = ""
            tip = "Not run yet"
        elif step.error:
            self._result_state = "error"
            lines = step.error.strip().splitlines() or ["Error"]
            summary = step.error_summary or lines[-1]
            tip = f"<b>{escape(summary)}</b><pre>{escape(step.error.strip())}</pre>"
        elif getattr(step, "cached", False):
            self._result_state = "ok"
            tip = "cached (unchanged since the last run)"
        else:
            self._result_state = "ok"
            ms = step.elapsed_ms
            tip = f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"
        self.status_dot.setToolTip(tip)
        level = self._result_state or "idle"
        if self.status_dot.property("level") != level:
            self.status_dot.setProperty("level", level)
            _repolish(self.status_dot)
        self._refresh_state()

    def _refresh_state(self):
        state = self._result_state if self.enabled else "disabled"
        if self.property("state") == state:
            return
        self.setProperty("state", state)
        # Descendant selectors such as #stepCard[state="disabled"] #cardTitle need the header re-polished too
        for widget in (self, self.header, *self.header.findChildren(QWidget)):
            _repolish(widget)

    def _sync_visibility(self):
        self.content.setVisible(self.enabled and not self._collapsed)
        chevron = "\u25B8" if self._collapsed else "\u25BE"
        self.title.setText(f"{chevron}  {self._schema['label']}")
        self._refresh_state()

    def _on_enabled_toggled(self, _checked):
        self._sync_visibility()
        self.enabled_toggled.emit()

    def _on_value_changed(self):
        self._refresh_summary()
        self.changed.emit()

    def _refresh_summary(self):
        self._summary_text = summarize(self.schema_key, self.get_values())
        self.summary.setToolTip(self._summary_text)
        self._elide_summary()

    def _elide_summary(self):
        self.summary.setText(self.summary.fontMetrics().elidedText(
            self._summary_text, Qt.TextElideMode.ElideRight, max(0, self.summary.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide_summary()

    # ── Mouse: drag from the handle, click elsewhere on the header to collapse ──

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        pos = event.position().toPoint()
        child = self.childAt(pos)
        self._press_pos = pos
        if child is self.handle:
            self._press_target = "handle"
        elif child is not None and (child is self.header or self.header.isAncestorOf(child)):
            self._press_target = "header"
        else:
            self._press_target = None
        event.accept()

    def mouseMoveEvent(self, event):
        if self._press_target != "handle":
            return super().mouseMoveEvent(event)
        if (event.position().toPoint() - self._press_pos).manhattanLength() >= QApplication.startDragDistance():
            self._press_target = None
            self._start_drag()

    def mouseReleaseEvent(self, event):
        if self._press_target == "header" and self.header.geometry().contains(event.position().toPoint()):
            self.set_collapsed(not self._collapsed)
        self._press_target = None
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(self.MIME_TYPE, b"")
        drag.setMimeData(mime)

        # Semi-transparent snapshot of the card
        pixmap = self.grab()
        faded = QPixmap(pixmap.size())
        faded.setDevicePixelRatio(pixmap.devicePixelRatio())
        faded.fill(Qt.GlobalColor.transparent)
        painter = QPainter(faded)
        painter.setOpacity(0.7)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        drag.setPixmap(faded)
        drag.setHotSpot(self._press_pos)

        self.handle.setCursor(Qt.CursorShape.ClosedHandCursor)
        drag.exec(Qt.DropAction.MoveAction)
        self.handle.setCursor(Qt.CursorShape.OpenHandCursor)

    def keyPressEvent(self, event):
        # Only when the card itself has focus, not a slider or spin box inside it
        if event.key() == Qt.Key.Key_Delete and self.hasFocus():
            self.request_remove.emit(self)
        else:
            super().keyPressEvent(event)


# ──────────────────────────────────────────────
# Card list: drop target with an insertion indicator
# ──────────────────────────────────────────────

class _CardList(QWidget):
    drop_requested = Signal(object, int)  # card, insertion index among the cards

    def __init__(self, scroll_area):
        super().__init__()
        self._scroll_area = scroll_area
        self.setAcceptDrops(True)
        self._indicator = QFrame(self)
        self._indicator.setObjectName("dropIndicator")
        self._indicator.setStyleSheet(f"background-color: {_ACCENT}; border: none;")
        self._indicator.hide()

    def cards(self):
        """Cards in layout order (the empty hint and the stretch are not counted)."""
        layout = self.layout()
        widgets = (layout.itemAt(i).widget() for i in range(layout.count()))
        return [w for w in widgets if isinstance(w, StepCard)]

    def drop_index(self, y):
        cards = self.cards()
        for i, card in enumerate(cards):
            if y < card.y() + card.height() / 2:
                return i
        return len(cards)

    def _accepts(self, event):
        return event.mimeData().hasFormat(StepCard.MIME_TYPE) and event.source() in self.cards()

    def dragEnterEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if not self._accepts(event):
            return
        event.acceptProposedAction()
        pos = event.position().toPoint()
        self._scroll_area.ensureVisible(pos.x(), pos.y(), 0, 48)  # auto-scroll near the edges
        cards = self.cards()
        idx = self.drop_index(pos.y())
        gap = self.layout().spacing()
        if idx < len(cards):
            y = cards[idx].y() - (gap + 2) // 2
        else:
            y = cards[-1].geometry().bottom() + 1 + (gap - 2) // 2
        margins = self.layout().contentsMargins()
        self._indicator.setGeometry(margins.left(), max(0, y), self.width() - margins.left() - margins.right(), 2)
        self._indicator.raise_()
        self._indicator.show()

    def dragLeaveEvent(self, event):
        self._indicator.hide()

    def dropEvent(self, event):
        self._indicator.hide()
        if self._accepts(event):
            event.acceptProposedAction()
            self.drop_requested.emit(event.source(), self.drop_index(event.position().y()))


# ──────────────────────────────────────────────
# PipelinePanel
# ──────────────────────────────────────────────

class PipelinePanel(QWidget):
    """Toolbar (add, load, save, copy HCL, clear) over a reorderable list of step cards."""

    changed = Signal()     # any value / order / enable change (main debounces)
    message = Signal(str)  # one-line feedback for the status bar

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cards = []
        self._preset_dir = ""
        # Lets a stylesheet background (main names this panel #pipelinePanel) paint on a QWidget subclass
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ──
        toolbar = QWidget()
        toolbar.setObjectName("panelToolbar")
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(4)

        self.add_btn = QPushButton("+  Add step")
        self.add_btn.setObjectName("addStepBtn")
        self.add_btn.setToolTip("Add a degradation step")
        self.add_btn.clicked.connect(self._show_add_menu)
        row.addWidget(self.add_btn)
        row.addStretch(1)
        for text, tip, slot in (
            ("Load", "Load a pipeline preset (.json)", self.request_load),
            ("Save", "Save this pipeline as a preset (.json)", self.request_save),
            ("Copy HCL", "Copy the enabled steps as wtp_dataset_destroyer HCL", self.copy_hcl),
            ("Clear", "Remove all steps", self._confirm_clear),
        ):
            btn = QPushButton(text)
            btn.setObjectName("toolBtn")
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        root.addWidget(toolbar)

        self.add_menu = AddStepMenu(self)
        self.add_menu.step_chosen.connect(self.add_step)

        # ── Card list ──
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.card_list = _CardList(self.scroll_area)
        self.card_list.drop_requested.connect(self._on_drop)
        self.list_layout = QVBoxLayout(self.card_list)
        self.list_layout.setContentsMargins(8, 4, 8, 8)
        self.list_layout.setSpacing(6)

        # Layout order: cards…, empty hint, stretch (card i sits at layout index i)
        self.empty_hint = QLabel("No steps yet.\nUse \u201C+ Add step\u201D to build a degradation pipeline.")
        self.empty_hint.setObjectName("emptyHint")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setWordWrap(True)
        self.list_layout.addWidget(self.empty_hint)
        self.list_layout.addStretch(1)

        self.scroll_area.setWidget(self.card_list)
        # QScrollArea fills its viewport and widget with palette colours; let the panel background show
        self.scroll_area.viewport().setAutoFillBackground(False)
        self.card_list.setAutoFillBackground(False)
        root.addWidget(self.scroll_area, 1)

    # ── Public API ──

    def get_configs(self):
        return [card.get_config() for card in self.cards if card.enabled]

    def get_state(self):
        return [card.get_state() for card in self.cards]

    def set_state(self, state):
        skipped = self._replace_cards(state)
        if skipped:
            self.message.emit(f"Skipped unknown step type(s): {', '.join(skipped)}")

    def set_step_results(self, steps):
        """Steps index into get_configs() order; a result whose type_key no longer matches is dropped."""
        enabled = [card for card in self.cards if card.enabled]
        results = {}
        for step in steps:
            if 0 <= step.index < len(enabled) \
                    and getattr(step, "type_key", enabled[step.index].schema_key) == enabled[step.index].schema_key:
                results[step.index] = step
        for i, card in enumerate(enabled):
            card.set_result(results.get(i))

    def clear_step_results(self):
        for card in self.cards:
            card.set_result(None)

    def add_step(self, schema_key):
        card = self._wire(StepCard(schema_key))
        self._insert(len(self.cards), card)
        self._structure_changed()
        card.setFocus()
        QTimer.singleShot(0, lambda: self.scroll_area.ensureWidgetVisible(card))

    def clear_all(self):
        """Remove every step immediately (the toolbar's Clear button asks first)."""
        if not self.cards:
            return
        for card in self.cards:
            self._discard(card)
        self.cards.clear()
        self._structure_changed()

    def request_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save pipeline preset", os.path.join(self._preset_dir, "pipeline.json"), _PRESET_FILTER)
        if not path:
            return
        self._preset_dir = os.path.dirname(path)
        try:
            save_preset(path, self.get_state())
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", f"Could not save {path}:\n{exc}")
            return
        self.message.emit(f"Saved {len(self.cards)} step(s) to {os.path.basename(path)}")

    def request_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load pipeline preset", self._preset_dir, _PRESET_FILTER)
        if not path:
            return
        self._preset_dir = os.path.dirname(path)
        try:
            skipped = self._replace_cards(load_preset(path))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Load failed", f"Could not load {path}:\n{exc}")
            return
        text = f"Loaded {len(self.cards)} step(s) from {os.path.basename(path)}"
        if skipped:
            text += f"; skipped unknown type(s): {', '.join(skipped)}"
        self.message.emit(text)

    def copy_hcl(self):
        configs = self.get_configs()
        if not configs:
            self.message.emit("Nothing to copy: no enabled steps")
            return
        QApplication.clipboard().setText(to_hcl(configs))
        self.message.emit(f"Copied {len(configs)} step(s) as HCL to the clipboard")

    # ── Internals ──

    def _show_add_menu(self):
        self.add_menu.popup_below(self.add_btn)

    def _confirm_clear(self):
        if not self.cards:
            return
        answer = QMessageBox.question(self, "Clear pipeline", f"Remove all {len(self.cards)} step(s)?")
        if answer == QMessageBox.StandardButton.Yes:
            self.clear_all()

    def _wire(self, card):
        card.changed.connect(self.changed.emit)
        card.enabled_toggled.connect(self._structure_changed)
        card.request_move.connect(self._move)
        card.request_duplicate.connect(self._duplicate)
        card.request_remove.connect(self._remove)
        for widget in card.param_widgets.values():
            for control in widget.controls:
                control.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # no focus-by-wheel
                control.installEventFilter(self)
        return card

    def eventFilter(self, obj, event):
        # The wheel scrolls the list unless the control under the cursor has focus
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            QApplication.sendEvent(self.scroll_area.verticalScrollBar(), event)
            return True
        return super().eventFilter(obj, event)

    def _replace_cards(self, state):
        """Build cards for `state` (all or nothing), swap them in; return the skipped type names."""
        new_cards, skipped = [], []
        for step in state:
            if not isinstance(step, dict) or step.get("type") not in SCHEMAS:
                skipped.append(str(step.get("type") if isinstance(step, dict) else step))
                continue
            card = StepCard(step["type"])
            card.apply_state(step)
            new_cards.append(card)
        for card in self.cards:
            self._discard(card)
        self.cards.clear()
        for card in new_cards:
            self._insert(len(self.cards), self._wire(card))
        self._structure_changed()
        return skipped

    def _insert(self, index, card):
        self.cards.insert(index, card)
        self.list_layout.insertWidget(index, card)

    def _discard(self, card):
        self.list_layout.removeWidget(card)
        card.hide()
        card.deleteLater()

    def _structure_changed(self):
        self.empty_hint.setVisible(not self.cards)
        self.clear_step_results()  # result indexes are stale after add/remove/reorder/toggle
        self.changed.emit()

    def _move_to(self, card, index):
        self.cards.remove(card)
        self.list_layout.removeWidget(card)
        self._insert(index, card)
        self._structure_changed()

    def _move(self, card, delta):
        index = self.cards.index(card) + delta
        if 0 <= index < len(self.cards):
            self._move_to(card, index)

    def _on_drop(self, card, target):
        index = self.cards.index(card)
        if target > index:
            target -= 1
        if target != index:
            self._move_to(card, target)

    def _duplicate(self, card):
        copy = StepCard(card.schema_key)
        copy.apply_state(card.get_state())
        self._insert(self.cards.index(card) + 1, self._wire(copy))
        self._structure_changed()

    def _remove(self, card):
        index = self.cards.index(card)
        self.cards.remove(card)
        self._discard(card)
        self._structure_changed()
        if self.cards:  # keep keyboard flow: focus the neighbour so Delete can repeat
            self.cards[min(index, len(self.cards) - 1)].setFocus()
