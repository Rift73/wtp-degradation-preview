"""
HQ/LQ comparison preview: a slim toolbar above a zoomable canvas.

View modes: wipe (draggable divider, HQ left / LQ right), side-by-side (two
panes with linked zoom and pan) and A/B (one image at a time, flipped with
toggle_ab). Zoom is absolute: 100% shows one HQ pixel per physical screen
pixel. HQ is drawn with Catmull-Rom; an LQ smaller than HQ is drawn
nearest-neighbour over HQ's rect so its real pixels show as blocks. Only the
visible crop is ever scaled, so memory stays bounded at any zoom.
"""

import math

import numpy as np
from PIL import Image as PILImage

from PySide6.QtWidgets import (
    QWidget, QFrame, QLabel, QToolButton, QButtonGroup, QHBoxLayout, QVBoxLayout,
)
from PySide6.QtCore import Qt, QEvent, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import (
    QPainter, QPixmap, QImage, QColor, QPen, QFont, QFontMetricsF,
    QMouseEvent, QPaintEvent, QWheelEvent,
)

# Design tokens (docs/DESIGN.md)
_BG = "#0F1115"
_PANEL = "#161A20"
_BORDER = "#2A3039"
_TEXT = "#E6E8EB"
_MUTED = "#8B93A1"
_ACCENT = "#5B9DF5"
_WARN = "#D4A843"


def _color(hex_rgb: str, alpha: int = 255) -> QColor:
    color = QColor(hex_rgb)
    color.setAlpha(alpha)
    return color


def _catmull_rom_scale(pixmap: QPixmap, target_w: int, target_h: int) -> QPixmap:
    """Scale a QPixmap using Catmull-Rom (Pillow BICUBIC)."""
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    if ptr is None or target_w <= 0 or target_h <= 0:
        return pixmap
    arr = np.frombuffer(ptr, dtype=np.uint8, count=w * h * 4).reshape(h, w, 4).copy()
    pil = PILImage.fromarray(arr, "RGBA")
    scaled = pil.resize((target_w, target_h), PILImage.Resampling.BICUBIC)
    data = scaled.tobytes("raw", "RGBA")
    result = QImage(data, target_w, target_h, target_w * 4,
                    QImage.Format.Format_RGBA8888).copy()  # .copy() detaches from buffer
    return QPixmap.fromImage(result)


class _Canvas(QWidget):
    """Paints HQ/LQ in the current view mode; handles zoom, pan and the wipe divider."""

    zoom_changed = Signal(float)

    _ZOOM_MIN = 0.1             # 10% of native
    _ZOOM_MAX = 32.0            # 3200% of native
    _WHEEL_STEP = 1.15          # zoom factor per wheel notch
    _ZOOM_STOPS = (0.1, 0.15, 0.25, 0.33, 0.5, 0.67, 1.0, 1.5, 2.0, 3.0,
                   4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
    _OVERSCROLL = 64            # px a zoomed image may be dragged past its edge
    _SPLIT_GRAB = 12            # px either side of the divider that grab it
    _GUTTER = 1                 # px between side-by-side panes
    _BUSY_DELAY_MS = 200        # runs faster than this never show the indicator

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hq = None             # QPixmap
        self._lq = None             # QPixmap, None = LQ slot shows HQ
        self._hq_detail = ""
        self._lq_detail = ""
        self._lq_nearest = False

        self._mode = "wipe"
        self._show_b = False
        self._split = 0.5           # divider position, fraction of the canvas width

        # Zoom / pan state
        self._fit = True
        self._scale = 1.0           # device pixels per HQ pixel (1.0 = 100%)
        self._pan = QPointF(0.0, 0.0)   # image centre offset from the pane centre
        self._last_pct = None

        # Mouse state
        self._drag = None           # "split" | "pan" | None
        self._hover_split = False
        self._pan_anchor = QPointF()
        self._pan_start = QPointF()

        # Busy indicator: shown only after a short delay, then spins
        self._spin = 0
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(40)
        self._spin_timer.timeout.connect(self._advance_spin)
        self._busy_delay = QTimer(self)
        self._busy_delay.setSingleShot(True)
        self._busy_delay.setInterval(self._BUSY_DELAY_MS)
        self._busy_delay.timeout.connect(self._spin_timer.start)

        # Scaled-crop cache: slot name -> (key_tuple, QPixmap)
        self._cache: dict = {}

        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

    # ── Public API (driven by ComparisonView) ──

    def set_images(self, hq: QPixmap, lq: QPixmap | None, hq_dims: str, lq_dims: str):
        if lq is not None and lq.isNull():
            lq = None
        if self._hq is None or self._hq.size() != hq.size():
            self._fit = True
        self._hq, self._lq = hq, lq
        self._hq_detail = hq_dims
        self._lq_detail = lq_dims if lq is not None else ""
        self._lq_nearest = lq is not None and (lq.width() < hq.width()
                                               or lq.height() < hq.height())
        if lq is not None and lq.size() != hq.size():
            factor = hq.width() / lq.width()
            self._lq_detail = f"{lq_dims} (shown \u00d7{round(factor, 2):g})".strip()
        self._refresh_view()

    def clear(self):
        self._hq = self._lq = None
        self._hq_detail = self._lq_detail = ""
        self._fit = True
        self._last_pct = None
        self._drag = None
        self._cache.clear()
        self.update()

    def set_busy(self, busy: bool):
        if busy:
            if not self._spin_timer.isActive() and not self._busy_delay.isActive():
                self._busy_delay.start()
        else:
            self._busy_delay.stop()
            self._spin_timer.stop()
            self.update()

    def set_view_mode(self, mode: str):
        self._mode = mode
        self._hover_split = False
        self._refresh_view()

    def flip_ab(self):
        self._show_b = not self._show_b
        self.update()

    def zoom_fit(self):
        self._fit = True
        self._refresh_view()

    def zoom_to(self, scale: float, anchor: QPointF | None = None):
        """Set the absolute zoom; the image point under *anchor* (default: pane centre) stays."""
        if self._hq is None:
            return
        fit = self._fit_scale()
        scale = max(min(self._ZOOM_MIN, fit), min(scale, max(self._ZOOM_MAX, fit)))
        panes = self._panes()
        pane = next((p for p in panes if anchor is not None and p.contains(anchor)), panes[0])
        if anchor is None:
            anchor = pane.center()

        # Fraction of the image under the anchor, before the zoom
        old = self._image_rect(pane)
        rel_x = (anchor.x() - old.x()) / old.width()
        rel_y = (anchor.y() - old.y()) / old.height()

        # New geometry: solve for pan so that point stays under the anchor
        self._fit = False
        self._scale = scale
        sw, sh = self._image_size()
        self._pan = self._clamped_pan(QPointF(
            anchor.x() - pane.x() - (pane.width() - sw) / 2.0 - rel_x * sw,
            anchor.y() - pane.y() - (pane.height() - sh) / 2.0 - rel_y * sh,
        ))
        self._emit_zoom()
        self.update()

    def zoom_step(self, direction: int):
        """Zoom to the next preset stop in *direction* (+1 in, -1 out) about the pane centre."""
        if direction > 0:
            stops = [s for s in self._ZOOM_STOPS if s > self._scale * 1.001]
            target = stops[0] if stops else None
        else:
            stops = [s for s in self._ZOOM_STOPS if s < self._scale / 1.001]
            target = stops[-1] if stops else None
        if target is not None:
            self.zoom_to(target)

    # ── Geometry helpers ──

    def _panes(self) -> list[QRectF]:
        """Viewport rects: the whole canvas, or two panes split by the gutter."""
        w, h = self.width(), self.height()
        if self._mode != "side":
            return [QRectF(0, 0, w, h)]
        left_w = (w - self._GUTTER) // 2
        return [QRectF(0, 0, left_w, h),
                QRectF(left_w + self._GUTTER, 0, w - left_w - self._GUTTER, h)]

    def _fit_scale(self) -> float:
        pane = self._panes()[0]
        fit = min(pane.width() / self._hq.width(), pane.height() / self._hq.height())
        return fit * self.devicePixelRatioF()

    def _image_size(self) -> tuple[float, float]:
        """Displayed HQ size in logical pixels."""
        k = self._scale / self.devicePixelRatioF()
        return self._hq.width() * k, self._hq.height() * k

    def _image_rect(self, pane: QRectF) -> QRectF:
        sw, sh = self._image_size()
        return QRectF(pane.x() + (pane.width() - sw) / 2.0 + self._pan.x(),
                      pane.y() + (pane.height() - sh) / 2.0 + self._pan.y(), sw, sh)

    def _overflows(self) -> bool:
        """True when the image is larger than its pane (so it can be panned)."""
        pane = self._panes()[0]
        sw, sh = self._image_size()
        return sw > pane.width() + 0.5 or sh > pane.height() + 0.5

    def _clamped_pan(self, pan: QPointF) -> QPointF:
        """Centre axes where the image fits; elsewhere stop a little past the image edge."""
        pane = self._panes()[0]
        sw, sh = self._image_size()
        over_x = max(0.0, sw - pane.width())
        over_y = max(0.0, sh - pane.height())
        lim_x = over_x / 2.0 + min(self._OVERSCROLL, over_x / 2.0)
        lim_y = over_y / 2.0 + min(self._OVERSCROLL, over_y / 2.0)
        return QPointF(max(-lim_x, min(pan.x(), lim_x)), max(-lim_y, min(pan.y(), lim_y)))

    def _split_x(self) -> float:
        """Divider x in canvas pixels, kept inside the visible part of the image."""
        visible = self._image_rect(self._panes()[0]).intersected(QRectF(self.rect()))
        return max(visible.left(), min(self._split * self.width(), visible.right()))

    def _near_split(self, pos: QPointF) -> bool:
        return (self._mode == "wipe" and self._hq is not None
                and abs(pos.x() - self._split_x()) < self._SPLIT_GRAB)

    def _lq_source(self) -> tuple[QPixmap, bool]:
        """Pixmap for the LQ slot and whether it is drawn with Catmull-Rom."""
        if self._lq is None:
            return self._hq, True
        return self._lq, not self._lq_nearest

    def _refresh_view(self):
        """Re-derive the fit scale and pan limits after the images, mode or size changed."""
        if self._hq is not None:
            if self._fit:
                self._scale = self._fit_scale()
                self._pan = QPointF(0.0, 0.0)
            self._pan = self._clamped_pan(self._pan)
            self._emit_zoom()
        self.update()

    def _emit_zoom(self):
        pct = self._scale * 100.0
        if self._last_pct is None or abs(pct - self._last_pct) > 1e-6:
            self._last_pct = pct
            self.zoom_changed.emit(pct)

    def _font(self, scale: float = 1.0, bold: bool = False) -> QFont:
        """The widget's (application) font, resized by *scale*."""
        font = QFont(self.font())
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * scale)
        else:
            font.setPixelSize(max(1, round(font.pixelSize() * scale)))
        font.setBold(bold)
        return font

    # ── Image drawing ──

    def _draw_image(self, painter: QPainter, pixmap: QPixmap, rect: QRectF,
                    pane: QRectF, clip: QRectF, smooth: bool, slot: str):
        """Draw *pixmap* stretched over *rect*, inside *pane*, clipped to *clip*.

        Only the part visible in the pane is scaled (Catmull-Rom when *smooth*,
        nearest-neighbour otherwise), at device resolution, and cached per *slot*;
        the clip is applied at paint time so moving the divider never rescales.
        """
        visible = rect.intersected(pane)
        if visible.isEmpty() or visible.intersected(clip).isEmpty():
            return
        dpr = self.devicePixelRatioF()
        kx = rect.width() / pixmap.width()      # logical px per source px
        ky = rect.height() / pixmap.height()

        # Map the visible rect to source pixels (+ pad for the bicubic kernel)
        pad = 2 if smooth else 0
        sx = max(0, math.floor((visible.left() - rect.left()) / kx) - pad)
        sy = max(0, math.floor((visible.top() - rect.top()) / ky) - pad)
        sr = min(pixmap.width(), math.ceil((visible.right() - rect.left()) / kx) + pad)
        sb = min(pixmap.height(), math.ceil((visible.bottom() - rect.top()) / ky) + pad)
        cw, ch = sr - sx, sb - sy
        if cw <= 0 or ch <= 0:
            return

        tw = max(1, round(cw * kx * dpr))
        th = max(1, round(ch * ky * dpr))
        key = (pixmap.cacheKey(), sx, sy, cw, ch, tw, th, smooth, dpr)
        cached = self._cache.get(slot)
        if cached is not None and cached[0] == key:
            scaled = cached[1]
        else:
            crop = pixmap.copy(sx, sy, cw, ch)
            if tw == cw and th == ch:
                scaled = crop       # native resolution: no scaling needed
            elif smooth:
                scaled = _catmull_rom_scale(crop, tw, th)
            else:
                scaled = crop.scaled(tw, th, Qt.AspectRatioMode.IgnoreAspectRatio,
                                     Qt.TransformationMode.FastTransformation)
            scaled.setDevicePixelRatio(dpr)
            self._cache[slot] = (key, scaled)

        # Snap the origin to a device pixel so native and nearest drawing stay exact
        x = round((rect.left() + sx * kx) * dpr) / dpr
        y = round((rect.top() + sy * ky) * dpr) / dpr
        painter.save()
        painter.setClipRect(clip)
        painter.drawPixmap(QPointF(x, y), scaled)
        painter.restore()

    # ── Painting ──

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        # No SmoothPixmapTransform: scaling is done per crop in _draw_image
        painter.fillRect(self.rect(), QColor(_BG))
        if self._hq is None:
            self._paint_empty(painter)
        elif self._mode == "side":
            self._paint_side(painter)
        elif self._mode == "ab":
            self._paint_ab(painter)
        else:
            self._paint_wipe(painter)
        if self._spin_timer.isActive():
            self._paint_busy(painter)
        painter.end()

    def _paint_empty(self, painter: QPainter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin = 24.5
        painter.setPen(QPen(QColor(_BORDER), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(margin, margin, w - 2 * margin, h - 2 * margin), 10, 10)
        painter.setFont(self._font(1.3))
        painter.setPen(QColor(_MUTED))
        painter.drawText(QRectF(0, 0, w, h / 2.0 - 4), Qt.AlignmentFlag.AlignHCenter
                         | Qt.AlignmentFlag.AlignBottom, "Drop an image here or press Ctrl+O")
        painter.setFont(self._font(0.9))
        painter.setPen(_color(_MUTED, 150))
        painter.drawText(QRectF(0, h / 2.0 + 4, w, h / 2.0), Qt.AlignmentFlag.AlignHCenter
                         | Qt.AlignmentFlag.AlignTop,
                         "PNG \u2022 JPEG \u2022 TIFF \u2022 BMP \u2022 WebP")

    def _paint_wipe(self, painter: QPainter):
        pane = self._panes()[0]
        rect = self._image_rect(pane)
        visible = rect.intersected(pane)
        split_x = self._split_x()
        left = QRectF(visible.left(), visible.top(), split_x - visible.left(), visible.height())
        right = QRectF(split_x, visible.top(), visible.right() - split_x, visible.height())
        lq, lq_smooth = self._lq_source()
        self._draw_image(painter, self._hq, rect, pane, left, True, "a")
        self._draw_image(painter, lq, rect, pane, right, lq_smooth, "b")

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Each label is clipped to its own side, so the wipe uncovers it with its image
        painter.save()
        painter.setClipRect(left)
        self._paint_pill(painter, visible.topLeft(), "HQ", self._hq_detail, _ACCENT)
        painter.setClipRect(right)
        self._paint_pill(painter, visible.topRight(), "LQ", self._lq_detail, _WARN, right=True)
        painter.restore()
        self._paint_divider(painter, split_x, visible)

    def _paint_side(self, painter: QPainter):
        left, right = self._panes()
        painter.fillRect(QRectF(left.right(), 0, right.left() - left.right(), self.height()),
                         QColor(_BORDER))
        lq, lq_smooth = self._lq_source()
        sides = ((left, self._hq, True, "a", "HQ", self._hq_detail, _ACCENT),
                 (right, lq, lq_smooth, "b", "LQ", self._lq_detail, _WARN))
        for pane, pixmap, smooth, slot, *_ in sides:
            self._draw_image(painter, pixmap, self._image_rect(pane), pane, pane, smooth, slot)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for pane, _, _, _, tag, detail, color in sides:
            painter.save()
            painter.setClipRect(pane)
            visible = self._image_rect(pane).intersected(pane)
            self._paint_pill(painter, visible.topLeft(), tag, detail, color)
            painter.restore()

    def _paint_ab(self, painter: QPainter):
        pane = self._panes()[0]
        if self._show_b:
            pixmap, smooth = self._lq_source()
            tag, detail, color, slot = "B \u00b7 LQ", self._lq_detail, _WARN, "b"
        else:
            pixmap, smooth = self._hq, True
            tag, detail, color, slot = "A \u00b7 HQ", self._hq_detail, _ACCENT, "a"
        self._draw_image(painter, pixmap, self._image_rect(pane), pane, pane, smooth, slot)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_pill(painter, pane.topLeft(), tag, detail, color, big=True)

    def _paint_pill(self, painter: QPainter, corner: QPointF, tag: str, detail: str,
                    color: str, right: bool = False, big: bool = False):
        """Draw a 'TAG detail' label inset 8 px from *corner* (top-left, top-right if *right*)."""
        tag_font = self._font(1.6 if big else 0.9, bold=True)
        detail_font = self._font(1.0 if big else 0.9)
        tag_w = QFontMetricsF(tag_font).horizontalAdvance(tag)
        detail_w = QFontMetricsF(detail_font).horizontalAdvance(detail) if detail else 0.0
        pad = 10.0 if big else 7.0
        gap = 7.0 if detail else 0.0
        height = QFontMetricsF(tag_font).height() + (10.0 if big else 6.0)
        width = pad * 2 + tag_w + gap + detail_w
        x = corner.x() - 8 - width if right else corner.x() + 8
        box = QRectF(x, corner.y() + 8, width, height)

        painter.setPen(QPen(_color(_BORDER), 1))
        painter.setBrush(_color(_BG, 200))
        painter.drawRoundedRect(box, 6, 6)
        align = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        painter.setFont(tag_font)
        painter.setPen(QColor(color))
        painter.drawText(QRectF(box.left() + pad, box.top(), tag_w + 1, height), align, tag)
        if detail:
            painter.setFont(detail_font)
            painter.setPen(QColor(_TEXT))
            detail_x = box.left() + pad + tag_w + gap
            painter.drawText(QRectF(detail_x, box.top(), detail_w + 1, height), align, detail)

    def _paint_divider(self, painter: QPainter, x: float, visible: QRectF):
        active = self._hover_split or self._drag == "split"
        line = _color(_TEXT, 230 if active else 150)
        painter.setPen(QPen(line, 2 if active else 1.5))
        painter.drawLine(QPointF(x, visible.top()), QPointF(x, visible.bottom()))

        # Round grip with left/right arrows at the vertical centre
        cy = visible.center().y()
        painter.setPen(QPen(QColor(_ACCENT) if active else line, 1.5))
        painter.setBrush(_color(_PANEL, 230))
        painter.drawEllipse(QPointF(x, cy), 12, 12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_ACCENT) if active else QColor(_TEXT))
        painter.drawPolygon([QPointF(x - 3, cy - 4), QPointF(x - 3, cy + 4), QPointF(x - 7, cy)])
        painter.drawPolygon([QPointF(x + 3, cy - 4), QPointF(x + 3, cy + 4), QPointF(x + 7, cy)])

    def _paint_busy(self, painter: QPainter):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = "Processing"
        font = self._font(0.9)
        height = QFontMetricsF(font).height() + 10
        width = QFontMetricsF(font).horizontalAdvance(text) + 40
        box = QRectF(self.width() - width - 12, self.height() - height - 12, width, height)
        painter.setPen(QPen(_color(_BORDER), 1))
        painter.setBrush(_color(_PANEL, 220))
        painter.drawRoundedRect(box, 6, 6)

        arc = QRectF(box.left() + 10, box.center().y() - 6, 12, 12)
        pen = QPen(QColor(_ACCENT), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(arc, -self._spin * 16, 270 * 16)
        painter.setFont(font)
        painter.setPen(QColor(_MUTED))
        painter.drawText(QRectF(arc.right() + 7, box.top(), width, height),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def _advance_spin(self):
        self._spin = (self._spin + 30) % 360
        self.update()

    # ── Mouse interaction ──

    def _update_cursor(self, pos: QPointF):
        if self._drag == "pan":
            shape = Qt.CursorShape.ClosedHandCursor
        elif self._drag == "split" or self._near_split(pos):
            shape = Qt.CursorShape.SplitHCursor
        elif self._hq is not None and self._overflows():
            shape = Qt.CursorShape.OpenHandCursor
        else:
            shape = Qt.CursorShape.ArrowCursor
        if self.cursor().shape() != shape:
            self.setCursor(shape)

    def mousePressEvent(self, event: QMouseEvent):
        if self._hq is None or self._drag is not None:
            return
        pos = event.position()
        left = event.button() == Qt.MouseButton.LeftButton
        if left and self._near_split(pos):
            self._drag = "split"
        elif event.button() == Qt.MouseButton.MiddleButton or (left and self._overflows()):
            self._drag = "pan"
            self._pan_anchor = pos
            self._pan_start = QPointF(self._pan)
        self._update_cursor(pos)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._drag == "split":
            self._split = max(0.0, min(1.0, pos.x() / max(self.width(), 1)))
            self.update()
        elif self._drag == "pan":
            self._pan = self._clamped_pan(self._pan_start + (pos - self._pan_anchor))
            self.update()
        else:
            near = self._near_split(pos)
            if near != self._hover_split:
                self._hover_split = near
                self.update()
        self._update_cursor(pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._drag is not None and event.buttons() == Qt.MouseButton.NoButton:
            self._drag = None
            self._update_cursor(event.position())
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click fits the image to the view."""
        if event.button() == Qt.MouseButton.LeftButton and self._hq is not None:
            self._drag = None
            self.zoom_fit()

    def leaveEvent(self, event: QEvent):
        if self._hover_split:
            self._hover_split = False
            self.update()
        super().leaveEvent(event)

    # ── Zoom (wheel, with or without Ctrl) and view changes ──

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if self._hq is None or delta == 0:
            event.ignore()
            return
        self.zoom_to(self._scale * self._WHEEL_STEP ** (delta / 120.0), event.position())
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_view()

    def event(self, event: QEvent) -> bool:
        # Moving to a screen with another scale factor changes the fit and pan limits
        if event.type() == QEvent.Type.DevicePixelRatioChange:
            self._refresh_view()
        return super().event(event)


class ComparisonView(QWidget):
    """Preview area: slim toolbar (view mode, zoom, readout) above the comparison canvas."""

    zoom_changed = Signal(float)    # effective percent, 100 = native HQ pixels

    _MODES = (
        ("wipe", "Wipe", "Drag the divider to compare HQ (left) with LQ (right)"),
        ("side", "Side-by-side", "HQ and LQ next to each other with linked zoom and pan"),
        ("ab", "A/B", "One image at a time; Space flips between HQ and LQ"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvas = _Canvas()
        self._canvas.zoom_changed.connect(self._on_zoom_changed)

        toolbar = QFrame()
        toolbar.setObjectName("previewToolbar")
        bar = QHBoxLayout(toolbar)
        bar.setContentsMargins(8, 4, 8, 4)
        bar.setSpacing(2)

        self._mode_group = QButtonGroup(self)   # exclusive
        self._mode_buttons = {}
        for mode, text, tip in self._MODES:
            btn = self._tool_button(text, tip)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, m=mode: self.set_view_mode(m))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode] = btn
            bar.addWidget(btn)

        separator = QFrame()
        separator.setFixedSize(1, 18)
        separator.setStyleSheet(f"background-color: {_BORDER};")
        bar.addSpacing(6)
        bar.addWidget(separator)
        bar.addSpacing(6)

        self._zoom_buttons = []
        for text, tip, slot in (
            ("Fit", "Fit the image to the view (or double-click the image)", self.zoom_fit),
            ("100%", "Actual pixels: one HQ pixel per screen pixel", self.zoom_100),
            ("\u2212", "Zoom out (the mouse wheel zooms at the cursor)", self.zoom_out),
            ("+", "Zoom in (the mouse wheel zooms at the cursor)", self.zoom_in),
        ):
            btn = self._tool_button(text, tip)
            btn.clicked.connect(slot)
            self._zoom_buttons.append(btn)
            bar.addWidget(btn)

        self._readout = QLabel("\u2014")
        self._readout.setObjectName("zoomReadout")
        self._readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._readout.setToolTip("Zoom; 100% = native HQ pixels")
        bar.addWidget(self._readout)
        bar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(self._canvas, 1)

        self._mode_buttons["wipe"].setChecked(True)
        self._set_zoom_enabled(False)

    @staticmethod
    def _tool_button(text: str, tip: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("toolBtn")
        btn.setText(text)
        btn.setToolTip(tip)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep Space for the A/B shortcut
        return btn

    def _set_zoom_enabled(self, enabled: bool):
        for btn in self._zoom_buttons:
            btn.setEnabled(enabled)
        self._readout.setEnabled(enabled)

    def _on_zoom_changed(self, pct: float):
        self._readout.setText(f"{pct:.0f}%")
        self.zoom_changed.emit(pct)

    # ── Public API ──

    def set_images(self, hq: QPixmap | None, lq: QPixmap | None,
                   hq_dims: str, lq_dims: str) -> None:
        """Show *hq* against *lq*. No HQ clears the view; no LQ shows HQ in the LQ slot.

        Zoom and pan are kept while the HQ size stays the same; a new size fits the view.
        """
        if hq is None or hq.isNull():
            self.clear()
            return
        self._canvas.set_images(hq, lq, hq_dims, lq_dims)
        self._set_zoom_enabled(True)

    def set_busy(self, busy: bool) -> None:
        """Show a small spinner while a run is in flight (after 200 ms); images stay visible."""
        self._canvas.set_busy(busy)

    def clear(self) -> None:
        """Back to the empty state (drop-zone hint)."""
        self._canvas.clear()
        self._readout.setText("\u2014")
        self._set_zoom_enabled(False)

    def set_view_mode(self, mode: str) -> None:
        """Switch to "wipe", "side" or "ab"."""
        self._mode_buttons[mode].setChecked(True)
        self._canvas.set_view_mode(mode)

    def toggle_ab(self) -> None:
        """Flip A/B; from another mode, switch to A/B and flip."""
        if not self._mode_buttons["ab"].isChecked():
            self.set_view_mode("ab")
        self._canvas.flip_ab()

    def zoom_fit(self) -> None:
        self._canvas.zoom_fit()

    def zoom_100(self) -> None:
        self._canvas.zoom_to(1.0)

    def zoom_in(self) -> None:
        self._canvas.zoom_step(1)

    def zoom_out(self) -> None:
        self._canvas.zoom_step(-1)
