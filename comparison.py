"""
Before/after comparison widget with slider wipe.

Draws original and degraded images side by side with a draggable
vertical divider line. Left = original, right = degraded.
Supports Ctrl+Scroll zoom toward cursor and middle-click pan.
Uses Catmull-Rom interpolation for non-native zoom levels.
"""

import math

import numpy as np
from PIL import Image as PILImage

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect, QRectF, QPointF, QTimer
from PySide6.QtGui import (
    QPainter, QPixmap, QImage, QColor, QPen, QFont,
    QMouseEvent, QPaintEvent, QWheelEvent,
)


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


class ComparisonSlider(QWidget):
    """Before/after image comparison with a draggable split line."""

    _ZOOM_MIN = 1.0
    _ZOOM_MAX = 32.0
    _ZOOM_STEP = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._original = None   # QPixmap
        self._degraded = None   # QPixmap
        self._split = 0.5       # 0.0–1.0 fraction from left
        self._dragging = False
        self._hover_near_split = False

        self._orig_dims = ""
        self._degr_dims = ""

        # Zoom / pan state
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_anchor = QPointF()
        self._pan_start = QPointF()

        # Fade-out timer for zoom badge
        self._zoom_badge_opacity = 0.0
        self._zoom_badge_timer = QTimer(self)
        self._zoom_badge_timer.setInterval(50)
        self._zoom_badge_timer.timeout.connect(self._fade_zoom_badge)

        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #0D0D0D;")

        # Catmull-Rom crop cache: name -> (key_tuple, QPixmap)
        self._cr_cache: dict = {}

        # Empty state
        self._has_original = False
        self._has_degraded = False

    # ── Public API ──

    def set_original(self, pixmap, dims=""):
        self._original = pixmap
        self._orig_dims = dims
        self._has_original = pixmap is not None and not pixmap.isNull()
        self._cr_cache.clear()
        self.update()

    def set_degraded(self, pixmap, dims=""):
        self._degraded = pixmap
        self._degr_dims = dims
        self._has_degraded = pixmap is not None and not pixmap.isNull()
        self._cr_cache.pop("degr_r", None)
        self.update()

    # ── Geometry helpers ──

    def _base_scale(self):
        """Scale factor that fits the image into the widget (no zoom)."""
        if not self._has_original:
            return 1.0
        pm = self._original
        return min(self.width() / pm.width(), self.height() / pm.height())

    def _image_rect(self):
        """Compute the rect where the image is drawn (fit + zoom + pan)."""
        if not self._has_original:
            return QRect(0, 0, self.width(), self.height())
        pm = self._original
        w, h = self.width(), self.height()
        scale = self._base_scale() * self._zoom
        sw, sh = int(pm.width() * scale), int(pm.height() * scale)
        x = (w - sw) / 2.0 + self._pan.x()
        y = (h - sh) / 2.0 + self._pan.y()
        return QRect(int(x), int(y), sw, sh)

    def _split_x(self):
        """Pixel x-coordinate of the split line."""
        rect = self._image_rect()
        return rect.x() + int(rect.width() * self._split)

    # ── Catmull-Rom drawing ──

    def _draw_with_catrom(self, painter: QPainter, pixmap: QPixmap,
                          image_rect: QRect, clip_rect: QRect, cache_name: str):
        """Draw *pixmap* into *image_rect*, clipped to *clip_rect*, using Catmull-Rom.

        Only the visible crop is scaled, so memory stays bounded at any zoom.
        """
        widget_bounds = QRect(0, 0, self.width(), self.height())
        visible = image_rect.intersected(clip_rect).intersected(widget_bounds)
        if visible.isEmpty():
            return

        eff_scale = self._base_scale() * self._zoom
        if eff_scale <= 0:
            return

        # Map visible rect → source pixmap coordinates (+ 2 px pad for kernel)
        pad = 2
        sx = max(0, int((visible.x() - image_rect.x()) / eff_scale) - pad)
        sy = max(0, int((visible.y() - image_rect.y()) / eff_scale) - pad)
        sr = min(pixmap.width(),
                 math.ceil((visible.right() + 1 - image_rect.x()) / eff_scale) + pad)
        sb = min(pixmap.height(),
                 math.ceil((visible.bottom() + 1 - image_rect.y()) / eff_scale) + pad)
        cw, ch = sr - sx, sb - sy
        if cw <= 0 or ch <= 0:
            return

        tw = max(1, round(cw * eff_scale))
        th = max(1, round(ch * eff_scale))
        key = (sx, sy, cw, ch, tw, th)

        cached = self._cr_cache.get(cache_name)
        if cached is not None and cached[0] == key:
            scaled_pm = cached[1]
        else:
            crop = pixmap.copy(sx, sy, cw, ch)
            if abs(eff_scale - 1.0) < 0.005:
                scaled_pm = crop  # native res — no scaling needed
            else:
                scaled_pm = _catmull_rom_scale(crop, tw, th)
            self._cr_cache[cache_name] = (key, scaled_pm)

        draw_x = image_rect.x() + round(sx * eff_scale)
        draw_y = image_rect.y() + round(sy * eff_scale)

        painter.setClipRect(visible)
        painter.drawPixmap(draw_x, draw_y, scaled_pm)

    # ── Painting ──

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        # No SmoothPixmapTransform — we do our own Catmull-Rom scaling

        w, h = self.width(), self.height()

        # Empty state
        if not self._has_original:
            painter.fillRect(0, 0, w, h, QColor("#0D0D0D"))
            painter.setPen(QPen(QColor("#3A3A3A"), 1, Qt.PenStyle.DashLine))
            margin = 40
            painter.drawRoundedRect(margin, margin, w - 2 * margin, h - 2 * margin, 12, 12)
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(QRectF(0, 0, w, h - 20), Qt.AlignmentFlag.AlignCenter,
                             "Drop an image here\nor click Load Image")
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor("#444444"))
            painter.drawText(QRectF(0, h / 2 + 20, w, 30), Qt.AlignmentFlag.AlignCenter,
                             "PNG \u2022 JPEG \u2022 TIFF \u2022 BMP \u2022 WebP")
            painter.end()
            return

        rect = self._image_rect()
        split_x = self._split_x()

        # Draw original (left of split) — Catmull-Rom
        if self._has_original:
            left_clip = QRect(rect.x(), rect.y(), split_x - rect.x(), rect.height())
            self._draw_with_catrom(painter, self._original, rect, left_clip, "orig_l")

        # Draw degraded (right of split) — Catmull-Rom
        if self._has_degraded:
            right_clip = QRect(split_x, rect.y(), rect.right() - split_x + 1, rect.height())
            self._draw_with_catrom(painter, self._degraded, rect, right_clip, "degr_r")
        elif self._has_original:
            right_clip = QRect(split_x, rect.y(), rect.right() - split_x + 1, rect.height())
            self._draw_with_catrom(painter, self._original, rect, right_clip, "orig_r")

        # Reset clip for overlays
        painter.setClipping(False)

        # ── Split line ──
        line_color = QColor("#FFFFFF") if self._hover_near_split or self._dragging else QColor("#BBBBBB")
        line_color.setAlpha(180 if self._dragging else 120)
        painter.setPen(QPen(line_color, 2))
        painter.drawLine(split_x, rect.y(), split_x, rect.bottom())

        # Handle triangles
        handle_color = QColor("#FFFFFF")
        handle_color.setAlpha(200 if self._hover_near_split else 140)
        painter.setBrush(handle_color)
        painter.setPen(Qt.PenStyle.NoPen)

        # Top triangle
        ty = rect.y() + 8
        painter.drawPolygon([
            QPointF(split_x - 6, ty),
            QPointF(split_x + 6, ty),
            QPointF(split_x, ty + 8),
        ])

        # Bottom triangle
        by = rect.bottom() - 8
        painter.drawPolygon([
            QPointF(split_x - 6, by),
            QPointF(split_x + 6, by),
            QPointF(split_x, by - 8),
        ])

        # ── Labels ──
        painter.setPen(QColor("#AAAAAA"))
        label_font = QFont("Segoe UI", 9)
        label_font.setBold(True)
        painter.setFont(label_font)

        # "Original" label — top-left of image
        if self._has_original:
            lx = rect.x() + 8
            ly = rect.y() + 4
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.drawRoundedRect(lx - 2, ly, 62, 18, 3, 3)
            painter.setPen(QColor("#CCCCCC"))
            painter.drawText(lx + 2, ly + 13, "Original")

        # "Degraded" label — top-right of image
        if self._has_degraded:
            rx = rect.right() - 70
            ry = rect.y() + 4
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 140))
            painter.drawRoundedRect(rx - 2, ry, 68, 18, 3, 3)
            painter.setPen(QColor("#CCCCCC"))
            painter.drawText(rx + 2, ry + 13, "Degraded")

        # ── Dimension overlay (bottom-left) ──
        if self._orig_dims or self._degr_dims:
            info_lines = []
            if self._orig_dims:
                info_lines.append(f"Source: {self._orig_dims}")
            if self._degr_dims:
                info_lines.append(f"Output: {self._degr_dims}")
            info_text = "  |  ".join(info_lines)

            info_font = QFont("Cascadia Mono", 9)
            painter.setFont(info_font)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(info_text) + 12
            th = 20
            ix = rect.x() + 6
            iy = rect.bottom() - th - 6

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(ix, iy, tw, th, 4, 4)
            painter.setPen(QColor("#AAAAAA"))
            painter.drawText(ix + 6, iy + 14, info_text)

        # ── Zoom badge (fades out) — percentage where 100% = native resolution ──
        eff_pct = self._base_scale() * self._zoom * 100.0
        if self._zoom_badge_opacity > 0:
            badge = f"{eff_pct:.0f}%"
            badge_font = QFont("Cascadia Mono", 11)
            badge_font.setBold(True)
            painter.setFont(badge_font)
            fm = painter.fontMetrics()
            bw = fm.horizontalAdvance(badge) + 16
            bh = 26
            bx = (w - bw) // 2
            by = 12

            alpha = int(self._zoom_badge_opacity * 180)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, alpha))
            painter.drawRoundedRect(bx, by, bw, bh, 6, 6)
            text_color = QColor("#FFFFFF")
            text_color.setAlpha(int(self._zoom_badge_opacity * 255))
            painter.setPen(text_color)
            painter.drawText(QRectF(bx, by, bw, bh), Qt.AlignmentFlag.AlignCenter, badge)

        painter.end()

    # ── Zoom badge fade ──

    def _fade_zoom_badge(self):
        self._zoom_badge_opacity -= 0.06
        if self._zoom_badge_opacity <= 0:
            self._zoom_badge_opacity = 0.0
            self._zoom_badge_timer.stop()
        self.update()

    # ── Mouse interaction ──

    def _start_pan(self, event: QMouseEvent):
        self._panning = True
        self._pan_anchor = event.position()
        self._pan_start = QPointF(self._pan)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            near_split = abs(event.pos().x() - self._split_x()) < 12
            if near_split:
                self._dragging = True
                self.setCursor(Qt.CursorShape.SplitHCursor)
                self.update()
            elif self._zoom > 1.0:
                # Left-click away from split → pan
                self._start_pan(event)
        elif event.button() == Qt.MouseButton.MiddleButton and self._zoom > 1.0:
            self._start_pan(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position() - self._pan_anchor
            self._pan = self._pan_start + delta
            self.update()
            return

        rect = self._image_rect()
        if self._dragging:
            x = event.pos().x()
            self._split = max(0.0, min(1.0, (x - rect.x()) / max(rect.width(), 1)))
            self.update()
        else:
            near = abs(event.pos().x() - self._split_x()) < 12
            if near != self._hover_near_split:
                self._hover_near_split = near
                self.setCursor(
                    Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.ArrowCursor
                )
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging:
                self._dragging = False
                self.update()
            elif self._panning:
                self._panning = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click resets zoom to 1:1 fit."""
        if event.button() == Qt.MouseButton.LeftButton and self._zoom != 1.0:
            self._zoom = 1.0
            self._pan = QPointF(0.0, 0.0)
            self._zoom_badge_opacity = 1.0
            self._zoom_badge_timer.start()
            self.update()

    # ── Zoom ──

    def wheelEvent(self, event: QWheelEvent):
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            super().wheelEvent(event)
            return
        if not self._has_original:
            event.accept()
            return

        mouse = event.position()
        pm = self._original
        w, h = self.width(), self.height()
        bs = self._base_scale()

        # Current image geometry
        old_scale = bs * self._zoom
        old_sw, old_sh = pm.width() * old_scale, pm.height() * old_scale
        old_x = (w - old_sw) / 2.0 + self._pan.x()
        old_y = (h - old_sh) / 2.0 + self._pan.y()

        # Image-space coordinate under mouse
        img_x = (mouse.x() - old_x) / old_scale
        img_y = (mouse.y() - old_y) / old_scale

        # Compute new zoom
        delta = event.angleDelta().y()
        factor = self._ZOOM_STEP if delta > 0 else 1.0 / self._ZOOM_STEP
        new_zoom = max(self._ZOOM_MIN, min(self._zoom * factor, self._ZOOM_MAX))

        # New image geometry — solve for pan so img point stays under cursor
        new_scale = bs * new_zoom
        new_sw, new_sh = pm.width() * new_scale, pm.height() * new_scale
        new_pan_x = mouse.x() - (w - new_sw) / 2.0 - img_x * new_scale
        new_pan_y = mouse.y() - (h - new_sh) / 2.0 - img_y * new_scale

        if new_zoom == 1.0:
            new_pan_x, new_pan_y = 0.0, 0.0

        self._zoom = new_zoom
        self._pan = QPointF(new_pan_x, new_pan_y)

        # Show zoom badge
        self._zoom_badge_opacity = 1.0
        self._zoom_badge_timer.start()

        self.update()
        event.accept()

    def reset_zoom(self):
        """Programmatic zoom reset."""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()
