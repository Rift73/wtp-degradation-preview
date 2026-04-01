"""
Widgets for the WTP Degradation Preview GUI.

- FloatParam / IntParam / ChoiceParam / BoolParam: individual parameter editors
- DegradationBlock: styled card with category color bar and collapsible params
- PipelinePanel: scrollable list with drag-and-drop reorder
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QCheckBox, QLabel, QPushButton, QFrame,
    QScrollArea, QSizePolicy, QApplication,
)
from PySide6.QtCore import Qt, Signal, QMimeData, QPoint
from PySide6.QtGui import QDrag, QDragEnterEvent, QDropEvent, QPainter, QPixmap, QColor

from schema import SCHEMAS, SCHEMA_ORDER, CATEGORY_COLORS, build_config


# ──────────────────────────────────────────────
# Individual parameter widgets
# ──────────────────────────────────────────────

class FloatParam(QWidget):
    value_changed = Signal()

    def __init__(self, pdef, parent=None):
        super().__init__(parent)
        self._pdef = pdef
        self._block = False

        lo = pdef["min"]
        hi = pdef["max"]
        step = pdef.get("step", 0.01)
        decimals = pdef.get("decimals", 2)
        default = pdef.get("default", lo)

        self._lo = lo
        self._step = step
        ticks = int(round((hi - lo) / step))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, ticks)
        self.slider.setValue(int(round((default - lo) / step)))

        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(default)
        self.spin.setFixedWidth(70)
        self.spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin, 0)

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
        self.spin.setValue(v)


class IntParam(QWidget):
    value_changed = Signal()

    def __init__(self, pdef, parent=None):
        super().__init__(parent)
        self._block = False

        lo = pdef["min"]
        hi = pdef["max"]
        default = pdef.get("default", lo)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(default)

        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setValue(default)
        self.spin.setFixedWidth(70)
        self.spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin, 0)

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
        self.spin.setValue(v)


class ChoiceParam(QComboBox):
    value_changed = Signal()

    def __init__(self, pdef, parent=None):
        super().__init__(parent)
        options = pdef["options"]
        self.addItems([str(o) for o in options])
        default = pdef.get("default", options[0])
        idx = options.index(default) if default in options else 0
        self.setCurrentIndex(idx)
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
        super().__init__(parent)
        self.setChecked(pdef.get("default", False))
        self.toggled.connect(lambda _: self.value_changed.emit())

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, v):
        self.setChecked(v)


_WIDGET_MAP = {
    "float": FloatParam,
    "int": IntParam,
    "choice": ChoiceParam,
    "bool": BoolParam,
}


def make_param_widget(pdef):
    cls = _WIDGET_MAP.get(pdef["type"])
    if cls is None:
        return QLabel(f"[unsupported: {pdef['type']}]")
    return cls(pdef)


# ──────────────────────────────────────────────
# DegradationBlock — styled card with category color bar
# ──────────────────────────────────────────────

class DegradationBlock(QFrame):
    """Styled degradation card with a category color accent bar."""

    changed = Signal()
    request_move_up = Signal(object)
    request_move_down = Signal(object)
    request_delete = Signal(object)

    MIME_TYPE = "application/x-wtp-degradation-block"

    def __init__(self, schema_key, parent=None):
        super().__init__(parent)
        self.schema_key = schema_key
        schema = SCHEMAS[schema_key]
        self._drag_start = QPoint()
        category_color = CATEGORY_COLORS.get(schema_key, "#5B9DF5")

        self.setObjectName("degradationBlock")
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Outer layout: color bar | content ──
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Category color bar
        color_bar = QFrame()
        color_bar.setObjectName("categoryBar")
        color_bar.setStyleSheet(f"background-color: {category_color};")
        color_bar.setFixedWidth(3)
        outer.addWidget(color_bar)

        # Content column
        content_col = QVBoxLayout()
        content_col.setContentsMargins(10, 6, 8, 6)
        content_col.setSpacing(4)
        outer.addLayout(content_col, 1)

        # ── Header row ──
        self.header_widget = QWidget()
        self.header_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        self.enable_cb = QCheckBox()
        self.enable_cb.setChecked(True)
        self.enable_cb.setToolTip("Enable/disable this degradation")
        self.enable_cb.toggled.connect(self._on_toggle)

        self.title_btn = QPushButton(schema["label"])
        self.title_btn.setObjectName("blockTitle")
        self.title_btn.clicked.connect(self._toggle_content)

        btn_up = QPushButton("\u25B2")
        btn_up.setObjectName("iconBtn")
        btn_up.setFixedSize(22, 22)
        btn_up.setToolTip("Move up")
        btn_up.clicked.connect(lambda: self.request_move_up.emit(self))

        btn_down = QPushButton("\u25BC")
        btn_down.setObjectName("iconBtn")
        btn_down.setFixedSize(22, 22)
        btn_down.setToolTip("Move down")
        btn_down.clicked.connect(lambda: self.request_move_down.emit(self))

        btn_del = QPushButton("\u2715")
        btn_del.setObjectName("deleteBtn")
        btn_del.setFixedSize(22, 22)
        btn_del.setToolTip("Remove")
        btn_del.clicked.connect(lambda: self.request_delete.emit(self))

        header.addWidget(self.enable_cb)
        header.addWidget(self.title_btn, 1)
        header.addWidget(btn_up)
        header.addWidget(btn_down)
        header.addWidget(btn_del)
        content_col.addWidget(self.header_widget)

        # ── Parameters ──
        self.content = QWidget()
        form = QFormLayout(self.content)
        form.setContentsMargins(4, 2, 0, 2)
        form.setSpacing(5)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.param_widgets = {}
        self.param_labels = {}
        for pdef in schema["params"]:
            w = make_param_widget(pdef)
            self.param_widgets[pdef["key"]] = w
            label = QLabel(pdef["label"])
            label.setObjectName("sectionLabel")
            label.setFixedWidth(110)
            self.param_labels[pdef["key"]] = label
            form.addRow(label, w)
            if hasattr(w, "value_changed"):
                w.value_changed.connect(self.changed.emit)

        # Wire dynamic profile dependencies (e.g. codec → quality range)
        for pdef in schema["params"]:
            profiles = pdef.get("profiles")
            if not profiles:
                continue
            source_key = profiles["source"]
            profile_map = profiles["map"]
            target_key = pdef["key"]
            source_widget = self.param_widgets.get(source_key)
            if source_widget is not None and hasattr(source_widget, "currentTextChanged"):
                source_widget.currentTextChanged.connect(
                    lambda val, tk=target_key, pm=profile_map:
                        self._apply_profile(tk, pm, val)
                )

        content_col.addWidget(self.content)

    def _apply_profile(self, target_key, profile_map, source_value):
        """Update a parameter widget when its profile source changes."""
        profile = profile_map.get(source_value)
        if profile is None:
            return
        widget = self.param_widgets.get(target_key)
        if widget is None:
            return
        if hasattr(widget, "set_range"):
            widget.set_range(profile["min"], profile["max"], profile["default"])
        label = self.param_labels.get(target_key)
        if label is not None and "label" in profile:
            label.setText(profile["label"])

    def _on_toggle(self, checked):
        self.content.setEnabled(checked)
        self.changed.emit()

    def _toggle_content(self):
        vis = not self.content.isVisible()
        self.content.setVisible(vis)

    # ── Drag support ──

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.header_widget.geometry().contains(event.pos()):
                self._drag_start = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return super().mouseMoveEvent(event)
        dist = (event.pos() - self._drag_start).manhattanLength()
        if dist < QApplication.startDragDistance():
            return super().mouseMoveEvent(event)

        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(self.MIME_TYPE, b"")
        drag.setMimeData(mime)

        # Grab a semi-transparent snapshot
        pixmap = self.grab()
        faded = QPixmap(pixmap.size())
        faded.fill(QColor(0, 0, 0, 0))
        painter = QPainter(faded)
        painter.setOpacity(0.7)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        drag.setPixmap(faded)
        drag.setHotSpot(event.pos())

        self.header_widget.setCursor(Qt.CursorShape.ClosedHandCursor)
        drag.exec(Qt.DropAction.MoveAction)
        self.header_widget.setCursor(Qt.CursorShape.OpenHandCursor)

    @property
    def enabled(self):
        return self.enable_cb.isChecked()

    def get_values(self):
        return {key: w.value for key, w in self.param_widgets.items()}

    def get_config(self):
        if not self.enabled:
            return None
        return build_config(self.schema_key, self.get_values())


# ──────────────────────────────────────────────
# Drop area for pipeline reordering
# ──────────────────────────────────────────────

class _DropArea(QWidget):
    drop_reorder = Signal(object, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasFormat(DegradationBlock.MIME_TYPE):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(DegradationBlock.MIME_TYPE):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasFormat(DegradationBlock.MIME_TYPE):
            return
        source = event.source()
        if not isinstance(source, DegradationBlock):
            return
        layout = self.layout()
        drop_y = event.position().y()
        target_idx = layout.count() - 1
        for i in range(layout.count() - 1):
            item = layout.itemAt(i)
            w = item.widget() if item else None
            if w is not None:
                mid = w.y() + w.height() / 2
                if drop_y < mid:
                    target_idx = i
                    break
        self.drop_reorder.emit(source, target_idx)
        event.acceptProposedAction()


# ──────────────────────────────────────────────
# PipelinePanel
# ──────────────────────────────────────────────

class PipelinePanel(QWidget):
    """Scrollable pipeline with styled add button and drag-and-drop."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blocks = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Pipeline header ──
        header = QWidget()
        header.setStyleSheet("background-color: #1A1A1A;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        lbl = QLabel("PIPELINE")
        lbl.setObjectName("headerLabel")
        lbl.setStyleSheet("color: #666666; font-size: 10px; font-weight: 700; letter-spacing: 2px;")
        header_layout.addWidget(lbl)
        header_layout.addStretch()
        root.addWidget(header)

        # ── Flow label: INPUT ──
        input_lbl = QLabel("  INPUT")
        input_lbl.setObjectName("dimLabel")
        input_lbl.setFixedHeight(18)
        input_lbl.setStyleSheet("color: #555555; font-size: 9px; padding-left: 12px; background: #1A1A1A;")
        root.addWidget(input_lbl)

        # ── Scroll area ──
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.scroll_content = _DropArea()
        self.scroll_content.drop_reorder.connect(self._on_drop_reorder)
        self.scroll_content.setStyleSheet("background-color: #1A1A1A;")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(6, 2, 6, 2)
        self.scroll_layout.setSpacing(2)

        # Empty state label
        self.empty_label = QLabel("Add a degradation to begin")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #444444; font-size: 11px; padding: 24px;")
        self.scroll_layout.addWidget(self.empty_label)

        self.scroll_layout.addStretch(1)
        scroll_area.setWidget(self.scroll_content)
        root.addWidget(scroll_area)

        # ── Flow label: OUTPUT ──
        output_lbl = QLabel("  OUTPUT")
        output_lbl.setObjectName("dimLabel")
        output_lbl.setFixedHeight(18)
        output_lbl.setStyleSheet("color: #555555; font-size: 9px; padding-left: 12px; background: #1A1A1A;")
        root.addWidget(output_lbl)

        # ── Add bar ──
        add_area = QWidget()
        add_area.setStyleSheet("background-color: #1A1A1A;")
        add_layout = QHBoxLayout(add_area)
        add_layout.setContentsMargins(8, 6, 8, 8)
        add_layout.setSpacing(6)

        self.type_combo = QComboBox()
        for key in SCHEMA_ORDER:
            self.type_combo.addItem(SCHEMAS[key]["label"], key)

        self.add_btn = QPushButton("+ Add")
        self.add_btn.setObjectName("addDegradation")
        self.add_btn.clicked.connect(self._add_clicked)

        add_layout.addWidget(self.type_combo, 1)
        add_layout.addWidget(self.add_btn, 0)
        root.addWidget(add_area)

    def _add_clicked(self):
        key = self.type_combo.currentData()
        self.add_block(key)

    def add_block(self, schema_key):
        block = DegradationBlock(schema_key)
        block.changed.connect(self.changed.emit)
        block.request_move_up.connect(self._move_up)
        block.request_move_down.connect(self._move_down)
        block.request_delete.connect(self._delete)

        self.blocks.append(block)
        # Insert before the stretch (last item)
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, block)
        self.empty_label.hide()
        self.changed.emit()

    def _move_up(self, block):
        idx = self.blocks.index(block)
        if idx <= 0:
            return
        self.blocks[idx], self.blocks[idx - 1] = self.blocks[idx - 1], self.blocks[idx]
        self._rebuild_layout()
        self.changed.emit()

    def _move_down(self, block):
        idx = self.blocks.index(block)
        if idx >= len(self.blocks) - 1:
            return
        self.blocks[idx], self.blocks[idx + 1] = self.blocks[idx + 1], self.blocks[idx]
        self._rebuild_layout()
        self.changed.emit()

    def _delete(self, block):
        self.blocks.remove(block)
        self.scroll_layout.removeWidget(block)
        block.deleteLater()
        if not self.blocks:
            self.empty_label.show()
        self.changed.emit()

    def _on_drop_reorder(self, source_block, target_idx):
        if source_block not in self.blocks:
            return
        old_idx = self.blocks.index(source_block)
        self.blocks.pop(old_idx)
        if target_idx > old_idx:
            target_idx -= 1
        target_idx = max(0, min(target_idx, len(self.blocks)))
        self.blocks.insert(target_idx, source_block)
        self._rebuild_layout()
        self.changed.emit()

    def _rebuild_layout(self):
        for b in self.blocks:
            self.scroll_layout.removeWidget(b)
        for i, b in enumerate(self.blocks):
            self.scroll_layout.insertWidget(i, b)

    def get_configs(self):
        configs = []
        for b in self.blocks:
            c = b.get_config()
            if c is not None:
                configs.append(c)
        return configs
