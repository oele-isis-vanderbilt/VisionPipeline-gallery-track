from __future__ import annotations

"""
gallery_track.tools.build_gallery

Interactive gallery builder GUI.

Usage
-----
CLI:
    python -m gallery_track.tools.build_gallery
    python -m gallery_track.tools.build_gallery --gallery-root /path/to/galleries --video /path/to/video.mp4

This opens a small GUI that lets you:
  - Select a gallery root directory (once).
  - Open a video file.
  - Create identity folders under the gallery root:
        GALLERY_ROOT / <identity_name> / *.jpg
  - Create identities by name.
  - Scrub through the video.
  - Draw boxes on frames (left-click & drag) to save crops into the
    selected identity's folder.

The resulting directory structure is compatible with gallery loaders used by:
- gallery_track.trackers.gallery_only
- gallery_track.trackers.gallery_hybrid
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore


# ---------------------------------------------------------------------
# Video canvas widget: displays frames and lets the user draw rectangles
# ---------------------------------------------------------------------


class VideoCanvas(QtWidgets.QLabel):
    """
    QLabel-based widget that displays a video frame and allows the user
    to draw a rectangle (left-click & drag).

    Emits:
        boxDrawn(QRect): when the user completes a drag operation.
        drawAttemptedWithoutIdentity(): when drawing is attempted but not enabled.
        drawingStarted(): when the user begins drawing a rectangle.
    """

    boxDrawn = QtCore.pyqtSignal(QtCore.QRect)
    wheelStepped = QtCore.pyqtSignal(int)
    drawAttemptedWithoutIdentity = QtCore.pyqtSignal()
    drawingStarted = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(size_policy)
        self.setMouseTracking(True)

        self._frame: Optional[np.ndarray] = None  # BGR frame
        self._pixmap: Optional[QtGui.QPixmap] = None

        self._drawing: bool = False
        self._start_pos: Optional[QtCore.QPoint] = None
        self._current_pos: Optional[QtCore.QPoint] = None

        self._video_width: int = 0
        self._video_height: int = 0

        self._box_color: QtGui.QColor = QtGui.QColor(QtCore.Qt.green)
        self._drawing_enabled: bool = True

    # ------------------------------
    # Frame / pixmap handling
    # ------------------------------

    def set_frame(self, frame_bgr: Optional[np.ndarray]) -> None:
        """Set the current frame (BGR np.ndarray) to display."""
        self._frame = frame_bgr
        if frame_bgr is None:
            self._pixmap = None
            self.clear()
            return

        h, w, _ = frame_bgr.shape
        self._video_width = w
        self._video_height = h

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            rgb.data,
            w,
            h,
            rgb.strides[0],
            QtGui.QImage.Format_RGB888,
        )
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draw the pixmap (scaled with aspect ratio) and the current rectangle."""
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)

        if self._pixmap is not None:
            target_rect = self._scaled_target_rect()
            painter.drawPixmap(target_rect, self._pixmap)

            if self._drawing and self._start_pos and self._current_pos:
                pen = QtGui.QPen(self._box_color, 2, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                rect = QtCore.QRect(self._start_pos, self._current_pos).normalized()
                painter.drawRect(rect)

        painter.end()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """Use the mouse wheel to step frames proportionally to scroll speed, only while actively scrolling."""
        if self._pixmap is not None:
            phase = getattr(event, "phase", None)
            if callable(phase):
                try:
                    scroll_phase = phase()
                except TypeError:
                    scroll_phase = QtCore.Qt.NoScrollPhase
            else:
                scroll_phase = QtCore.Qt.NoScrollPhase

            if scroll_phase == QtCore.Qt.ScrollMomentum:
                event.ignore()
                return

            delta = event.angleDelta().y()
            if delta != 0:
                steps = delta // 120
                if steps == 0:
                    steps = 1 if delta > 0 else -1
                frames_per_step = 5
                total_steps = steps * frames_per_step
                step = -int(total_steps)
                self.wheelStepped.emit(step)

        super().wheelEvent(event)

    def _scaled_target_rect(self) -> QtCore.QRect:
        """Compute where the frame is drawn inside the widget, preserving aspect ratio."""
        if self._pixmap is None or self._video_width <= 0 or self._video_height <= 0:
            return self.rect()

        widget_w = self.width()
        widget_h = self.height()
        vid_w = self._video_width
        vid_h = self._video_height

        if vid_w <= 0 or vid_h <= 0:
            return self.rect()

        scale = min(widget_w / vid_w, widget_h / vid_h)
        scaled_w = int(vid_w * scale)
        scaled_h = int(vid_h * scale)
        offset_x = (widget_w - scaled_w) // 2
        offset_y = (widget_h - scaled_h) // 2
        return QtCore.QRect(offset_x, offset_y, scaled_w, scaled_h)

    def _widget_to_video_coords(self, p: QtCore.QPoint) -> Tuple[int, int]:
        """Map widget coordinates to video (frame) coordinates."""
        target = self._scaled_target_rect()
        if target.width() <= 0 or target.height() <= 0:
            return 0, 0

        x = p.x()
        y = p.y()

        if x < target.left():
            x = target.left()
        if x > target.right():
            x = target.right()
        if y < target.top():
            y = target.top()
        if y > target.bottom():
            y = target.bottom()

        rel_x = (x - target.left()) / max(1.0, target.width())
        rel_y = (y - target.top()) / max(1.0, target.height())

        vid_x = int(rel_x * self._video_width)
        vid_y = int(rel_y * self._video_height)
        return vid_x, vid_y

    def set_drawing_enabled(self, enabled: bool) -> None:
        """Enable or disable drawing on the canvas."""
        self._drawing_enabled = enabled

    # ------------------------------
    # Mouse handling
    # ------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pixmap is not None:
            if not self._drawing_enabled:
                self.drawAttemptedWithoutIdentity.emit()
                return

            if event.button() == QtCore.Qt.LeftButton:
                self._drawing = True
                self._start_pos = event.pos()
                self._current_pos = event.pos()
                self.drawingStarted.emit()
                self.update()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drawing:
            self._current_pos = event.pos()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._drawing:
            self._drawing = False
            if self._start_pos and self._current_pos:
                rect = QtCore.QRect(self._start_pos, self._current_pos).normalized()
                self.boxDrawn.emit(rect)
            self._start_pos = None
            self._current_pos = None
            self.update()
        else:
            super().mouseReleaseEvent(event)

    # ------------------------------
    # Public helpers
    # ------------------------------

    def current_frame(self) -> Optional[np.ndarray]:
        """Return the current frame (BGR) if any."""
        return self._frame

    def set_box_color(self, color: QtGui.QColor) -> None:
        """Set the color used for drawing the selection box."""
        self._box_color = color
        self.update()


# ---------------------------------------------------------------------
# Main window: gallery builder logic
# ---------------------------------------------------------------------


class GalleryBuilderWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        gallery_root: Optional[Path] = None,
        video_path: Optional[Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gallery Track — Gallery Builder")

        self.gallery_root: Optional[Path] = gallery_root
        self.video_path: Optional[Path] = video_path

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.current_frame_index: int = 0

        self.canvas = VideoCanvas(self)

        self.gallery_label = QtWidgets.QLabel("Gallery root: (not set)")
        self.video_label = QtWidgets.QLabel("Video: (not set)")

        self.browse_gallery_btn = QtWidgets.QPushButton("Browse gallery root…")
        self.browse_video_btn = QtWidgets.QPushButton("Open video…")

        self.identity_input = QtWidgets.QLineEdit()
        self.identity_input.setPlaceholderText("New identity name")
        self.add_identity_btn = QtWidgets.QPushButton("Add identity")
        self.identities_list = QtWidgets.QListWidget()

        self.prev_btn = QtWidgets.QPushButton("⏮ Prev")
        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.next_btn = QtWidgets.QPushButton("⏭ Next")
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.setPageStep(10)

        self.frame_info_label = QtWidgets.QLabel("Frame: 0 / 0")
        self.status_label = QtWidgets.QLabel("Left-click & drag on the video to save a crop.")

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._on_playback_tick)

        self._playing: bool = False

        self.identity_colors: Dict[str, QtGui.QColor] = {}
        self._last_clicked_identity_item: Optional[QtWidgets.QListWidgetItem] = None

        self._build_layout()
        self._connect_signals()

        self.canvas.set_drawing_enabled(False)

        self._update_gallery_label()
        self._update_video_label()

    # ------------------------------
    # UI construction
    # ------------------------------

    def _build_layout(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        paths_layout = QtWidgets.QGridLayout()
        paths_layout.addWidget(self.gallery_label, 0, 0)
        paths_layout.addWidget(self.browse_gallery_btn, 0, 1)
        paths_layout.addWidget(self.video_label, 1, 0)
        paths_layout.addWidget(self.browse_video_btn, 1, 1)

        id_layout = QtWidgets.QVBoxLayout()
        id_layout.addWidget(QtWidgets.QLabel("Identities:"))
        id_layout.addWidget(self.identities_list)
        new_id_layout = QtWidgets.QHBoxLayout()
        new_id_layout.addWidget(self.identity_input)
        new_id_layout.addWidget(self.add_identity_btn)
        id_layout.addLayout(new_id_layout)

        video_ctrl_layout = QtWidgets.QHBoxLayout()
        video_ctrl_layout.addWidget(self.prev_btn)
        video_ctrl_layout.addWidget(self.play_btn)
        video_ctrl_layout.addWidget(self.next_btn)
        video_ctrl_layout.addWidget(self.frame_info_label)

        bottom_layout = QtWidgets.QVBoxLayout()
        bottom_layout.addWidget(self.frame_slider)
        bottom_layout.addLayout(video_ctrl_layout)
        bottom_layout.addWidget(self.status_label)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.canvas)

        side_panel = QtWidgets.QVBoxLayout()
        side_panel.addLayout(paths_layout)
        side_panel.addSpacing(10)
        side_panel.addLayout(id_layout)
        side_panel.addStretch(1)

        side_widget = QtWidgets.QWidget()
        side_widget.setLayout(side_panel)
        side_widget.setMaximumWidth(320)

        main_layout.addWidget(side_widget)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.addLayout(main_layout)
        root_layout.addLayout(bottom_layout)

        self.resize(1200, 700)

    def _connect_signals(self) -> None:
        self.browse_gallery_btn.clicked.connect(self._choose_gallery_root)
        self.browse_video_btn.clicked.connect(self._choose_video)
        self.add_identity_btn.clicked.connect(self._on_add_identity)

        self.prev_btn.clicked.connect(self._on_prev_frame)
        self.next_btn.clicked.connect(self._on_next_frame)
        self.play_btn.clicked.connect(self._on_toggle_play)

        self.frame_slider.valueChanged.connect(self._on_slider_changed)

        self.canvas.boxDrawn.connect(self._on_box_drawn)
        self.canvas.wheelStepped.connect(self._on_wheel_step)
        self.canvas.drawAttemptedWithoutIdentity.connect(self._on_draw_attempted_without_identity)
        self.canvas.drawingStarted.connect(self._on_drawing_started)
        self.identities_list.itemSelectionChanged.connect(self._on_identity_selected)
        self.identities_list.itemClicked.connect(self._on_identity_clicked)

    def _on_drawing_started(self) -> None:
        if self._playing:
            self._playing = False
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")

    def _create_color_icon(self, color: QtGui.QColor) -> QtGui.QIcon:
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(color)
        return QtGui.QIcon(pixmap)

    def _get_identity_color(self, name: str) -> QtGui.QColor:
        import hashlib

        h = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16)
        hue = h % 360
        color = QtGui.QColor()
        color.setHsv(hue, 200, 255)
        return color

    # ------------------------------
    # Path handling
    # ------------------------------

    def _choose_gallery_root(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Select gallery root")
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                self.gallery_root = Path(selected[0]).expanduser().resolve()
                self._update_gallery_label()
                self._load_existing_identities()

    def _choose_video(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Select video file")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilter("Video files (*.mp4 *.avi *.mkv *.mov *.webm);;All files (*.*)")
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                self.video_path = Path(selected[0]).expanduser().resolve()
                self._open_video(self.video_path)

    def _update_gallery_label(self) -> None:
        if self.gallery_root is None:
            self.gallery_label.setText("Gallery root: (not set)")
        else:
            self.gallery_label.setText(f"Gallery root: {self.gallery_root}")

    def _update_video_label(self) -> None:
        if self.video_path is None:
            self.video_label.setText("Video: (not set)")
        else:
            self.video_label.setText(f"Video: {self.video_path}")

    # ------------------------------
    # Video handling
    # ------------------------------

    def _open_video(self, video_path: Path) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.video_path = video_path
        self._update_video_label()

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            self.status_label.setText(f"Failed to open video: {video_path}")
            self.frame_count = 0
            self.frame_slider.setMaximum(0)
            self.canvas.set_frame(None)
            return

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.current_frame_index = 0
        self.frame_slider.setMaximum(max(0, self.frame_count - 1))
        self.frame_slider.setValue(0)

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-2:
            fps = 25.0
        interval_ms = int(1000.0 / fps)
        self.play_timer.setInterval(interval_ms)

        self._load_frame(0)
        self._update_frame_info()

    def _load_frame(self, index: int) -> None:
        if self.cap is None or self.frame_count == 0:
            self.canvas.set_frame(None)
            return

        index = max(0, min(index, self.frame_count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            self.status_label.setText(f"Failed to read frame {index}")
            self.canvas.set_frame(None)
            return

        self.current_frame_index = index
        self.canvas.set_frame(frame)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self._update_frame_info()

    def _update_frame_info(self) -> None:
        self.frame_info_label.setText(f"Frame: {self.current_frame_index} / {max(0, self.frame_count - 1)}")

    # ------------------------------
    # Identity handling
    # ------------------------------

    def _load_existing_identities(self) -> None:
        self.identities_list.clear()
        if not self.gallery_root or not self.gallery_root.exists():
            return
        for entry in sorted(self.gallery_root.iterdir()):
            if entry.is_dir():
                name = entry.name
                if name not in self.identity_colors:
                    self.identity_colors[name] = self._get_identity_color(name)
                item = QtWidgets.QListWidgetItem(name)
                color = self.identity_colors[name]
                item.setIcon(self._create_color_icon(color))
                self.identities_list.addItem(item)

    def _current_identity(self) -> Optional[str]:
        selected_items = self.identities_list.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].text().strip()

    def _on_add_identity(self) -> None:
        name = self.identity_input.text().strip()
        if not name:
            self.status_label.setText("Please enter an identity name.")
            return
        if self.gallery_root is None:
            self.status_label.setText("Set gallery root first.")
            return

        target_dir = self.gallery_root / name
        target_dir.mkdir(parents=True, exist_ok=True)
        if name not in self.identity_colors:
            self.identity_colors[name] = self._get_identity_color(name)

        existing_items = [self.identities_list.item(i).text() for i in range(self.identities_list.count())]
        if name not in existing_items:
            item = QtWidgets.QListWidgetItem(name)
            color = self.identity_colors[name]
            item.setIcon(self._create_color_icon(color))
            self.identities_list.addItem(item)

        matches = self.identities_list.findItems(name, QtCore.Qt.MatchExactly)
        if matches:
            self.identities_list.setCurrentItem(matches[0])

        self.identity_input.clear()
        self.status_label.setText(f"Identity '{name}' ready. Left-click & drag to add images.")

    def _on_identity_selected(self) -> None:
        ident = self._current_identity()
        if ident:
            color = self.identity_colors.get(ident)
            if color is not None:
                self.canvas.set_box_color(color)
            self.canvas.set_drawing_enabled(True)
            self.status_label.setText(f"Selected identity: '{ident}'. Left-click & drag to save crops.")
        else:
            self.canvas.set_drawing_enabled(False)
            self.status_label.setText("No identity selected.")

    def _on_identity_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if (
            self.identities_list.selectedItems()
            and self.identities_list.currentItem() is item
            and self._last_clicked_identity_item is item
        ):
            self.identities_list.clearSelection()
            self._last_clicked_identity_item = None
        else:
            self._last_clicked_identity_item = item

    def _on_draw_attempted_without_identity(self) -> None:
        if self._current_identity() is None:
            QtWidgets.QMessageBox.warning(self, "Select identity", "Please select an identity.")

    def _on_wheel_step(self, step: int) -> None:
        if self.frame_count == 0:
            return
        self._load_frame(self.current_frame_index + step)

    # ------------------------------
    # Video control handlers
    # ------------------------------

    def _on_prev_frame(self) -> None:
        if self.frame_count == 0:
            return
        self._load_frame(self.current_frame_index - 1)

    def _on_next_frame(self) -> None:
        if self.frame_count == 0:
            return
        self._load_frame(self.current_frame_index + 1)

    def _on_toggle_play(self) -> None:
        if self.cap is None or self.frame_count == 0:
            return
        if self._playing:
            self._playing = False
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self._playing = True
            self.play_timer.start()
            self.play_btn.setText("⏸ Pause")

    def _on_playback_tick(self) -> None:
        if self.frame_count == 0:
            return
        next_index = self.current_frame_index + 1
        if next_index >= self.frame_count:
            self._playing = False
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
            return
        self._load_frame(next_index)

    def _on_slider_changed(self, value: int) -> None:
        if self.frame_count == 0:
            return
        self._load_frame(value)

    # ------------------------------
    # Box drawing / saving
    # ------------------------------

    def _on_box_drawn(self, rect: QtCore.QRect) -> None:
        ident = self._current_identity()
        if ident is None:
            self.status_label.setText("Select or create an identity first.")
            return
        if self.gallery_root is None:
            self.status_label.setText("Set gallery root first.")
            return

        frame = self.canvas.current_frame()
        if frame is None:
            self.status_label.setText("No frame loaded.")
            return

        p1 = rect.topLeft()
        p2 = rect.bottomRight()
        x1, y1 = self.canvas._widget_to_video_coords(p1)
        x2, y2 = self.canvas._widget_to_video_coords(p2)

        if x2 <= x1 or y2 <= y1:
            self.status_label.setText("Invalid box; ignoring.")
            return

        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            self.status_label.setText("Invalid box after clamping; ignoring.")
            return

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            self.status_label.setText("Empty crop; ignoring.")
            return

        ident_dir = self.gallery_root / ident
        ident_dir.mkdir(parents=True, exist_ok=True)

        existing_indices: List[int] = []
        pattern = f"{ident}_*.jpg"
        for p in ident_dir.glob(pattern):
            if not p.is_file():
                continue
            stem = p.stem
            suffix = stem[len(ident) + 1 :]
            if suffix.isdigit():
                existing_indices.append(int(suffix))

        next_idx = (max(existing_indices) + 1) if existing_indices else 0
        filename = f"{ident}_{next_idx:05d}.jpg"
        out_path = ident_dir / filename

        ok = cv2.imwrite(str(out_path), crop)
        if ok:
            self.status_label.setText(f"Saved crop for '{ident}' → {out_path.name}")
        else:
            self.status_label.setText(f"Failed to save crop to {out_path}")

    # ------------------------------
    # Cleanup
    # ------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Gallery builder GUI (gallery_track)")
    ap.add_argument("--gallery-root", type=Path, default=None, help="Root directory for galleries (identities as subdirs).")
    ap.add_argument("--video", type=Path, default=None, help="Optional initial video to open.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    gallery_root = args.gallery_root
    if gallery_root is not None:
        gallery_root = gallery_root.expanduser().resolve()

    video_path = args.video
    if video_path is not None:
        video_path = video_path.expanduser().resolve()

    app = QtWidgets.QApplication([])
    win = GalleryBuilderWindow(gallery_root=gallery_root, video_path=video_path)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()