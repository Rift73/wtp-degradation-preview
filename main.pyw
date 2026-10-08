"""
WTP Degradation Preview GUI

Standalone PySide6 application for real-time preview of the
wtp_dataset_destroyer degradation pipeline.

Usage:
    run.bat            (pythonw main.pyw)
"""

import sys
import os
import json
import random
import logging

import numpy as np

# ── Bootstrap imports ──
_OWN_DIR = os.path.dirname(os.path.abspath(__file__))
if _OWN_DIR not in sys.path:
    sys.path.insert(0, _OWN_DIR)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QPushButton, QLabel, QFileDialog, QStatusBar,
    QFrame, QSpinBox, QMessageBox, QToolButton, QMenu,
)
from PySide6.QtCore import Qt, QTimer, QByteArray
from PySide6.QtGui import (
    QFont, QDragEnterEvent, QDropEvent, QShortcut, QKeySequence, QActionGroup,
)

import video_backend
from engine import PipelineEngine, load_image, numpy_to_qpixmap
from widgets import PipelinePanel
from comparison import ComparisonView

_CONFIG_PATH = os.path.join(_OWN_DIR, "config.json")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_SEED_MAX = 2**31 - 1


def _load_config():
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                return cfg
        except (json.JSONDecodeError, OSError):
            logging.warning("config.json unreadable, starting fresh")
    return {}


def _save_config(cfg):
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        logging.exception("Could not write config.json")


def _dims_str(img):
    h, w = img.shape[:2]
    return f"{w}×{h}"


# ──────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WTP Degradation Preview")
        self.resize(1500, 920)
        self.setAcceptDrops(True)

        self.cfg = _load_config()
        self.source_image = None
        self.source_name = ""
        self._source_pixmap = None
        self._last_hq_u8 = None
        self._last_hq_pixmap = None
        self.last_dir = self.cfg.get("last_dir", "")
        self._last_error = ""
        video_backend.restore_ffmpeg(self.cfg)

        self.engine = PipelineEngine(self)
        self.engine.result_ready.connect(self._on_result)
        self.engine.failed.connect(self._on_engine_failed)
        self.engine.busy_changed.connect(self._on_busy)

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(120)
        self.debounce.timeout.connect(self._run_pipeline)

        self._build_ui()
        self._build_shortcuts()
        self._restore_state()

    # ── UI construction ──

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(44)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 0, 14, 0)
        hl.setSpacing(8)

        title = QLabel("WTP Degradation Preview")
        title.setObjectName("appTitle")
        hl.addWidget(title)

        self.image_label = QLabel("")
        self.image_label.setObjectName("imageName")
        hl.addWidget(self.image_label)
        hl.addStretch()

        self.video_chip = QToolButton()
        self.video_chip.setObjectName("videoBackendChip")
        self.video_chip.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        video_menu = QMenu(self.video_chip)
        video_menu.addAction("Locate ffmpeg…", self._locate_ffmpeg)
        video_menu.addSeparator()
        backend_group = QActionGroup(video_menu)
        self.use_system_action = video_menu.addAction(
            "Use system ffmpeg", lambda: self._use_video_backend("ffmpeg"))
        self.use_pyav_action = video_menu.addAction(
            "Use built-in (PyAV)", lambda: self._use_video_backend("pyav"))
        for action in (self.use_system_action, self.use_pyav_action):
            action.setCheckable(True)
            backend_group.addAction(action)
        video_menu.aboutToShow.connect(self._sync_video_menu)
        self.video_chip.setMenu(video_menu)
        self._refresh_video_chip()
        hl.addWidget(self.video_chip)

        seed_lbl = QLabel("Seed")
        seed_lbl.setObjectName("dimLabel")
        hl.addWidget(seed_lbl)
        self.seed_spin = QSpinBox()
        self.seed_spin.setObjectName("seedSpin")
        self.seed_spin.setRange(0, _SEED_MAX)
        self.seed_spin.setValue(int(self.cfg.get("seed", 1234)))
        self.seed_spin.setToolTip("Random seed for every stochastic step. Edit it or press R to re-roll.")
        self.seed_spin.setFixedWidth(110)
        self.seed_spin.valueChanged.connect(lambda _v: self._schedule_run())
        hl.addWidget(self.seed_spin)

        self.reroll_btn = QPushButton("Re-roll")
        self.reroll_btn.setToolTip("New random seed, same settings  (R)")
        self.reroll_btn.clicked.connect(self._reroll)
        hl.addWidget(self.reroll_btn)

        self.open_btn = QPushButton("Open image")
        self.open_btn.setObjectName("accentButton")
        self.open_btn.setToolTip("Open an image  (Ctrl+O). Drag-and-drop works too.")
        self.open_btn.clicked.connect(self._open_image)
        hl.addWidget(self.open_btn)

        root.addWidget(header)

        # Splitter: pipeline | preview
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.pipeline = PipelinePanel()
        self.pipeline.setObjectName("pipelinePanel")
        self.pipeline.changed.connect(self._schedule_run)
        self.pipeline.message.connect(self._status)

        self.preview = ComparisonView()

        self.splitter.addWidget(self.pipeline)
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([440, 1060])
        root.addWidget(self.splitter, 1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.error_btn = QToolButton()
        self.error_btn.setObjectName("errorButton")
        self.error_btn.setText("⚠ details")
        self.error_btn.setToolTip("Show the full error of the last run")
        self.error_btn.clicked.connect(self._show_last_error)
        self.error_btn.hide()
        self.status.addPermanentWidget(self.error_btn)
        self._status("Ready — open an image (Ctrl+O) or drop one onto the window")

    def _build_shortcuts(self):
        def bind(keys, fn):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(fn)

        bind("Ctrl+O", self._open_image)
        bind("Ctrl+S", self.pipeline.request_save)
        bind("Ctrl+L", self.pipeline.request_load)
        bind("Ctrl+Shift+C", self.pipeline.copy_hcl)
        bind("R", self._reroll)
        bind("Space", self.preview.toggle_ab)
        bind("F", self.preview.zoom_fit)
        bind("1", self.preview.zoom_100)
        bind("Ctrl+=", self.preview.zoom_in)
        bind("Ctrl++", self.preview.zoom_in)
        bind("Ctrl+-", self.preview.zoom_out)

    # ── Persistence ──

    def _restore_state(self):
        geo = self.cfg.get("geometry")
        if geo:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode("ascii")))
        sizes = self.cfg.get("splitter")
        if isinstance(sizes, list) and len(sizes) == 2:
            self.splitter.setSizes([int(s) for s in sizes])
        mode = self.cfg.get("view_mode")
        if mode in ("wipe", "side", "ab"):
            self.preview.set_view_mode(mode)
        state = self.cfg.get("pipeline")
        if isinstance(state, list) and state:
            try:
                self.pipeline.set_state(state)
            except Exception:
                logging.exception("Could not restore the saved pipeline")

    def closeEvent(self, event):
        self.cfg["geometry"] = bytes(self.saveGeometry().toBase64()).decode("ascii")
        self.cfg["splitter"] = self.splitter.sizes()
        self.cfg["last_dir"] = self.last_dir
        self.cfg["seed"] = self.seed_spin.value()
        self.cfg["pipeline"] = self.pipeline.get_state()
        _save_config(self.cfg)
        super().closeEvent(event)

    # ── Video backend (header chip) ──

    def _refresh_video_chip(self):
        backend = video_backend.detect()
        if backend.kind == "ffmpeg":
            text = f"Video: ffmpeg {backend.version} (system)"
            tip = f"Video codecs run through {backend.path}"
        elif backend.kind == "pyav":
            text = "Video: built-in (PyAV)"
            tip = f"Video codecs run in-process through PyAV's FFmpeg {backend.version}"
        else:
            text = "Video: none"
            tip = ("No ffmpeg found and PyAV is not installed: the H.264, HEVC, MPEG-2,\n"
                   "MPEG-4 and VP9 codecs fail until you locate an ffmpeg executable")
        self.video_chip.setText(text)
        self.video_chip.setToolTip(f"{tip}\nClick to locate ffmpeg or switch the video backend.")
        self.video_chip.setProperty("state", backend.kind)
        self.video_chip.style().unpolish(self.video_chip)
        self.video_chip.style().polish(self.video_chip)

    def _sync_video_menu(self):
        """Check the backend in use; disable a backend that is not available."""
        kind = video_backend.detect().kind
        self.use_system_action.setEnabled(video_backend.ffmpeg_available())
        self.use_pyav_action.setEnabled(video_backend.pyav_available())
        self.use_system_action.setChecked(kind == "ffmpeg")
        self.use_pyav_action.setChecked(kind == "pyav")

    def _video_backend_changed(self):
        self._refresh_video_chip()
        _save_config(self.cfg)
        self._status(self.video_chip.text())
        self._schedule_run()

    def _use_video_backend(self, kind):
        video_backend.set_preference(kind)
        self.cfg["video_backend"] = kind
        self._video_backend_changed()

    def _locate_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Locate ffmpeg", "", "ffmpeg (ffmpeg.exe ffmpeg);;All (*)",
        )
        if not path:
            return
        if video_backend.register_ffmpeg(path, self.cfg):
            self._video_backend_changed()
        else:
            self._status(f"Not a working ffmpeg executable: {path}")

    # ── Image loading ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._dropped_image_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        path = self._dropped_image_path(event)
        if path is not None:
            self._apply_image(path)
            event.acceptProposedAction()

    @staticmethod
    def _dropped_image_path(event):
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if os.path.splitext(path)[1].lower() in _IMAGE_EXTS:
                    return path
        return None

    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.last_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;All (*)",
        )
        if path:
            self._apply_image(path)

    def _apply_image(self, path):
        img = load_image(path)
        if img is None:
            self._status(f"Failed to load: {path}")
            return
        self.source_image = img
        self.source_name = os.path.basename(path)
        self.last_dir = os.path.dirname(path)
        dims = _dims_str(img)
        self.image_label.setText(f"{self.source_name}  ·  {dims}")
        pm = numpy_to_qpixmap(img)
        self._source_pixmap = pm
        self._last_hq_u8 = None
        self._last_hq_pixmap = None
        self.preview.set_images(pm, pm, dims, dims)
        self.preview.zoom_fit()
        self._status(f"Loaded {self.source_name}  {dims}")
        self._run_pipeline()

    # ── Pipeline execution ──

    def _schedule_run(self):
        if self.source_image is not None:
            self.debounce.start()

    def _reroll(self):
        self.seed_spin.setValue(random.randrange(_SEED_MAX))

    def _run_pipeline(self):
        if self.source_image is None:
            return
        configs = self.pipeline.get_configs()
        if not configs:
            dims = _dims_str(self.source_image)
            pm = self._source_pixmap
            self.preview.set_images(pm, pm, dims, dims)
            self.pipeline.clear_step_results()
            self._set_error("")
            self._status("No active steps — add one with the + button")
            return
        self.engine.request_run(self.source_image, configs, self.seed_spin.value())

    def _on_busy(self, busy):
        self.preview.set_busy(busy)

    def _on_result(self, result):
        hq_dims = _dims_str(result.hq)
        lq_dims = _dims_str(result.lq)
        # Reuse a pixmap whenever HQ's bytes did not change, so the view keeps its
        # scaled cache: the source pixmap when no step touched HQ, else the last one.
        # The engine says when HQ is the previous result's; a result still in
        # flight when a new image was opened finds no pixmap and compares instead.
        if not result.hq_changed:
            hq_pm = self._source_pixmap
        elif result.hq_same_as_previous and self._last_hq_pixmap is not None:
            hq_pm = self._last_hq_pixmap
        elif self._last_hq_u8 is not None and self._last_hq_u8.shape == result.hq_u8.shape \
                and np.array_equal(self._last_hq_u8, result.hq_u8):
            hq_pm = self._last_hq_pixmap
        else:
            hq_pm = numpy_to_qpixmap(result.hq_u8)
            self._last_hq_u8, self._last_hq_pixmap = result.hq_u8, hq_pm
        self.preview.set_images(hq_pm, numpy_to_qpixmap(result.lq_u8), hq_dims, lq_dims)
        self.pipeline.set_step_results(result.steps)

        failed = [s for s in result.steps if s.error]
        n = len(result.steps)
        plural = "s" if n != 1 else ""
        cached = f" · {result.cached_steps} cached" if result.cached_steps else ""
        summary = f"{n} step{plural} · {result.total_ms:.0f} ms{cached} · seed {result.seed}"
        if failed:
            first = failed[0]
            self._set_error("\n\n".join(f"[{s.type_key}] {s.error}" for s in failed))
            self._status(
                f"{len(failed)} step(s) failed — {first.type_key}: "
                f"{first.error_summary}  ({summary})"
            )
        else:
            self._set_error("")
            self._status(f"Done · {summary} · LQ {lq_dims}")

    def _on_engine_failed(self, traceback_text):
        logging.error("Engine failure:\n%s", traceback_text)
        self._set_error(traceback_text)
        lines = traceback_text.strip().splitlines()
        last = lines[-1][:160] if lines else "unknown"
        self._status(f"Engine error: {last}")

    # ── Status helpers ──

    def _status(self, text):
        self.status.showMessage(text)

    def _set_error(self, text):
        self._last_error = text
        self.error_btn.setVisible(bool(text))

    def _show_last_error(self):
        if not self._last_error:
            return
        dlg = QMessageBox(self)
        dlg.setIcon(QMessageBox.Icon.Warning)
        dlg.setWindowTitle("Last run errors")
        dlg.setText("One or more steps failed. The failing steps were skipped.")
        dlg.setDetailedText(self._last_error)
        dlg.exec()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    log_path = os.path.join(_OWN_DIR, "wtp_preview.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Segoe UI Variable", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    qss_path = os.path.join(_OWN_DIR, "style.qss")
    if os.path.isfile(qss_path):
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
