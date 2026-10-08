"""
"Add step" popup for the pipeline panel.

AddStepMenu: a search box over a list grouped by category. Typing filters, Up/Down moves the
highlight, Enter or a click adds the highlighted step, Escape or a click outside closes it.
"""

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLineEdit, QListWidget, QListWidgetItem
from PySide6.QtCore import Qt, Signal, QEvent, QPoint

from schema import SCHEMAS, SCHEMA_ORDER, CATEGORY_OF, CATEGORY_COLORS

_KEY_ROLE = Qt.ItemDataRole.UserRole


class AddStepMenu(QFrame):
    step_chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("addMenu")
        self.resize(280, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search steps\u2026")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        self.search.returnPressed.connect(self._choose_current)
        self.search.installEventFilter(self)

        self.list = QListWidget()
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # typing always goes to the search box
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemClicked.connect(self._choose)

        layout.addWidget(self.search)
        layout.addWidget(self.list, 1)

        self._groups = []  # (category name, header item, step items)
        self._populate()

    def _populate(self):
        by_category = {name: [] for name in CATEGORY_COLORS}  # schema keeps these in Add-menu order
        for key in SCHEMA_ORDER:
            by_category[CATEGORY_OF[key]].append(key)
        for name, keys in by_category.items():
            if not keys:
                continue
            header = QListWidgetItem(name.upper())
            header.setFlags(Qt.ItemFlag.NoItemFlags)  # styled as a muted group header
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            self.list.addItem(header)
            items = []
            for key in keys:
                item = QListWidgetItem(SCHEMAS[key]["label"])
                item.setData(_KEY_ROLE, key)
                self.list.addItem(item)
                items.append(item)
            self._groups.append((name, header, items))

    def popup_below(self, anchor):
        """Show under `anchor` with an empty filter and the first step highlighted."""
        self.search.clear()
        self._filter("")
        self.move(anchor.mapToGlobal(QPoint(0, anchor.height() + 2)))
        self.show()
        self.search.setFocus()

    def _filter(self, text):
        needle = text.strip().lower()
        for name, header, items in self._groups:
            group_hit = needle in name.lower()
            any_visible = False
            for item in items:
                visible = group_hit or needle in item.text().lower() or needle in item.data(_KEY_ROLE)
                item.setHidden(not visible)
                any_visible = any_visible or visible
            header.setHidden(not any_visible)
        rows = self._choosable_rows()
        self.list.setCurrentRow(rows[0] if rows else -1)

    def _choosable_rows(self):
        return [row for row in range(self.list.count())
                if self.list.item(row).data(_KEY_ROLE) and not self.list.item(row).isHidden()]

    def _move_highlight(self, delta):
        rows = self._choosable_rows()
        if not rows:
            return
        current = self.list.currentRow()
        pos = rows.index(current) + delta if current in rows else 0
        self.list.setCurrentRow(rows[max(0, min(len(rows) - 1, pos))])

    def eventFilter(self, obj, event):
        if obj is self.search and event.type() == QEvent.Type.KeyPress \
                and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self._move_highlight(-1 if event.key() == Qt.Key.Key_Up else 1)
            return True
        return super().eventFilter(obj, event)

    def _choose_current(self):
        item = self.list.currentItem()
        if item is not None and not item.isHidden():
            self._choose(item)

    def _choose(self, item):
        key = item.data(_KEY_ROLE)
        if key:
            self.hide()
            self.step_chosen.emit(key)
