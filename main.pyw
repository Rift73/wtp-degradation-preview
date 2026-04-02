"""
WTP Degradation Preview GUI

Standalone PySide6 application for real-time preview of the
wtp_dataset_destroyer degradation pipeline.

Usage:
    python main.py
"""

import sys
import os
import json
import shutil
import time
import traceback
import logging

import numpy as np
import cv2

# ── Bootstrap imports ──
_OWN_DIR = os.path.dirname(os.path.abspath(__file__))
if _OWN_DIR not in sys.path:
    sys.path.insert(0, _OWN_DIR)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QPushButton, QLabel, QFileDialog, QStatusBar,
    QProgressBar, QFrame,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QDragEnterEvent, QDropEvent

from widgets import PipelinePanel
from comparison import ComparisonSlider

# ──────────────────────────────────────────────
# FFmpeg discovery
# ──────────────────────────────────────────────
_CONFIG_PATH = os.path.join(_OWN_DIR, "config.json")


def _load_config():
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(cfg):
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _ffmpeg_available():
    """Return True if ffmpeg is reachable from the current PATH."""
    return shutil.which("ffmpeg") is not None


def _try_restore_ffmpeg():
    """Try to restore ffmpeg path from saved config. Returns True if successful."""
    cfg = _load_config()
    saved = cfg.get("ffmpeg_path", "")
    if saved and os.path.isfile(saved):
        ffdir = os.path.dirname(saved)
        if ffdir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffdir + os.pathsep + os.environ.get("PATH", "")
        return _ffmpeg_available()
    return False


def _register_ffmpeg(path):
    """Save ffmpeg path to config and add its directory to PATH."""
    cfg = _load_config()
    cfg["ffmpeg_path"] = path
    _save_config(cfg)
    ffdir = os.path.dirname(path)
    if ffdir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffdir + os.pathsep + os.environ.get("PATH", "")


_pipeline_loaded = False


def _ensure_pipeline():
    global _pipeline_loaded
    if _pipeline_loaded:
        return
    import pipeline.process  # noqa: F401
    _pipeline_loaded = True


# ──────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────

def load_image(path):
    """Load image as float32 [0,1] RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.clip(img, 0, 1)


def numpy_to_qpixmap(img):
    """Convert float32 [0,1] numpy image to QPixmap."""
    img = np.ascontiguousarray(np.clip(img, 0, 1))
    img_u8 = (img * 255).astype(np.uint8)
    if img_u8.ndim == 2:
        h, w = img_u8.shape
        qimg = QImage(img_u8.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, c = img_u8.shape
        if c == 1:
            img_u8 = img_u8.squeeze()
            qimg = QImage(img_u8.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            qimg = QImage(img_u8.data, w, h, w * c, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ──────────────────────────────────────────────
# Processing thread
# ──────────────────────────────────────────────

class ProcessWorker(QThread):
    result_ready = Signal(object, object, float)  # lq, hq, elapsed
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.configs = []

    def run(self):
        try:
            _ensure_pipeline()
            from pipeline.utils.registry import get_class

            t0 = time.perf_counter()
            lq = self.source.copy()
            hq = self.source.copy()

            errors = []
            for config in self.configs:
                type_key = config["type"]
                cls = get_class(type_key)
                if cls is None:
                    errors.append(f"[{type_key}] Unknown degradation type")
                    continue
                instance = cls(config)
                try:
                    result = instance.run(lq, hq)
                except Exception:
                    tb = traceback.format_exc()
                    errors.append(f"[{type_key}] {tb}")
                    logging.error("Degradation %s failed:\n%s", type_key, tb)
                    continue
                if result is not None:
                    lq, hq = result

            elapsed = time.perf_counter() - t0
            if errors:
                self.error_occurred.emit(
                    f"{len(errors)} degradation(s) failed:\n\n"
                    + "\n".join(errors)
                )
            self.result_ready.emit(lq, hq, elapsed)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


# ──────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────

class MainWindow(QMainWindow):
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WTP Degradation Preview")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self.source_image = None
        self.source_path = ""
        self._pending_rerun = False
        self._has_ffmpeg = _ffmpeg_available() or _try_restore_ffmpeg()

        self.worker = ProcessWorker()
        self.worker.result_ready.connect(self._on_result)
        self.worker.error_occurred.connect(self._on_error)

        self.debounce = QTimer()
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(150)
        self.debounce.timeout.connect(self._run_pipeline)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Header bar ──
        header = QFrame()
        header.setFixedHeight(40)
        header.setStyleSheet(
            "background-color: #1A1A1A; border-bottom: 1px solid #2A2A2A;"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)
        header_layout.setSpacing(8)

        title = QLabel("WTP Degradation Preview")
        title.setStyleSheet("color: #E0E0E0; font-size: 13px; font-weight: 600;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        # ── FFmpeg locator (hidden once found) ──
        self.ffmpeg_btn = QPushButton("Locate FFmpeg")
        self.ffmpeg_btn.setToolTip(
            "Video codecs (H264, HEVC, VP9, MPEG) need FFmpeg.\n"
            "Click to select your ffmpeg.exe."
        )
        self.ffmpeg_btn.setStyleSheet(
            "QPushButton { background-color: #3A2A10; color: #F0C050; "
            "border: 1px solid #6B5020; padding: 3px 10px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #4A3A18; }"
        )
        self.ffmpeg_btn.clicked.connect(self._locate_ffmpeg)
        header_layout.addWidget(self.ffmpeg_btn)
        self.ffmpeg_btn.setVisible(not self._has_ffmpeg)

        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self._load_image)
        header_layout.addWidget(self.load_btn)

        self.reroll_btn = QPushButton("Re-roll")
        self.reroll_btn.setToolTip("Re-run with same settings (different random seed)")
        self.reroll_btn.clicked.connect(self._run_pipeline)
        header_layout.addWidget(self.reroll_btn)

        main_layout.addWidget(header)

        # ── Processing indicator ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setFixedHeight(2)
        self.progress.setTextVisible(False)
        self.progress.hide()
        main_layout.addWidget(self.progress)

        # ── Splitter: pipeline | preview ──
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: pipeline panel
        pipeline_frame = QFrame()
        pipeline_frame.setObjectName("pipelinePanel")
        pipeline_layout = QVBoxLayout(pipeline_frame)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.setSpacing(0)

        self.pipeline = PipelinePanel()
        self.pipeline.changed.connect(self._on_pipeline_changed)
        pipeline_layout.addWidget(self.pipeline)

        # Right: comparison slider preview
        preview_frame = QFrame()
        preview_frame.setObjectName("previewPanel")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)

        self.comparison = ComparisonSlider()
        preview_layout.addWidget(self.comparison)

        splitter.addWidget(pipeline_frame)
        splitter.addWidget(preview_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 1060])

        main_layout.addWidget(splitter, 1)

        # ── Status bar ──
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready \u2014 load an image or drag one onto the window")

    # ── FFmpeg locator ──

    def _locate_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate FFmpeg", "",
            "ffmpeg (ffmpeg.exe ffmpeg);;All (*)",
        )
        if not path:
            return
        _register_ffmpeg(path)
        if _ffmpeg_available():
            self._has_ffmpeg = True
            self.ffmpeg_btn.setVisible(False)
            self.status.showMessage(f"FFmpeg found: {path}")
        else:
            self.status.showMessage("Selected file is not a valid ffmpeg executable")

    # ── Drag-and-drop image loading ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    ext = os.path.splitext(url.toLocalFile())[1].lower()
                    if ext in self._IMAGE_EXTS:
                        event.acceptProposedAction()
                        return

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                ext = os.path.splitext(path)[1].lower()
                if ext in self._IMAGE_EXTS:
                    self._apply_image(path)
                    event.acceptProposedAction()
                    return

    # ── Image loading ──

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.source_path or "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;All (*)",
        )
        if path:
            self._apply_image(path)

    def _apply_image(self, path):
        img = load_image(path)
        if img is None:
            self.status.showMessage(f"Failed to load: {path}")
            return

        self.source_image = img
        self.source_path = os.path.dirname(path)
        dims = self._dims_str(img)
        pm = numpy_to_qpixmap(img)
        self.comparison.set_original(pm, dims)
        self.comparison.set_degraded(pm, dims)
        self.status.showMessage(f"Loaded: {os.path.basename(path)}  {dims}")
        self._on_pipeline_changed()

    @staticmethod
    def _dims_str(img):
        if img.ndim == 2:
            h, w = img.shape
            return f"{w}\u00d7{h} gray"
        h, w, c = img.shape
        return f"{w}\u00d7{h}\u00d7{c}"

    # ── Pipeline execution ──

    def _on_pipeline_changed(self):
        if self.source_image is not None:
            self.debounce.start()

    def _run_pipeline(self):
        if self.source_image is None:
            return
        if self.worker.isRunning():
            self._pending_rerun = True
            return
        self._pending_rerun = False

        configs = self.pipeline.get_configs()
        if not configs:
            dims = self._dims_str(self.source_image)
            self.comparison.set_degraded(numpy_to_qpixmap(self.source_image), dims)
            self.status.showMessage("No active degradations")
            return

        self.progress.show()
        self.status.showMessage("Processing\u2026")
        self.worker.source = self.source_image
        self.worker.configs = configs
        self.worker.start()

    def _on_result(self, lq, hq, elapsed):
        self.progress.hide()
        dims = self._dims_str(lq)
        self.comparison.set_degraded(numpy_to_qpixmap(lq), dims)
        # Update original side if HQ was modified by any degradation
        if hq is not self.source_image and not np.array_equal(hq, self.source_image):
            orig_dims = self._dims_str(hq)
            self.comparison.set_original(numpy_to_qpixmap(hq), orig_dims)
        else:
            orig_dims = self._dims_str(self.source_image)
            self.comparison.set_original(numpy_to_qpixmap(self.source_image), orig_dims)
        self.status.showMessage(f"Done in {elapsed:.3f}s \u2014 {dims}")
        if self._pending_rerun:
            self._pending_rerun = False
            self._run_pipeline()

    def _on_error(self, msg):
        self.progress.hide()
        # Show first line in status bar, full traceback in a dialog
        first_line = msg.strip().split("\n")[0][:120]
        self.status.showMessage(f"Error: {first_line}")
        logging.error("Pipeline error:\n%s", msg)

        from PySide6.QtWidgets import QMessageBox
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setWindowTitle("Degradation Error")
        dlg.setText("One or more degradation steps failed.")
        dlg.setDetailedText(msg)
        dlg.show()

        if self._pending_rerun:
            self._pending_rerun = False
            self._run_pipeline()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    # Log to file since pythonw.exe has no console
    log_path = os.path.join(_OWN_DIR, "wtp_preview.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),  # also stderr if running from terminal
        ],
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set default font
    font = QFont("Segoe UI Variable", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    # Load stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.isfile(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
