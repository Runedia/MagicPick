import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QScrollArea


class ImageViewer(QScrollArea):
    zoom_changed = pyqtSignal(float)  # 배율 변경 시그널

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_image = None
        self.zoom_factor = 1.0  # 확대/축소 배율

        # 패닝 관련 변수
        self.panning = False
        self.pan_start_pos = None

        self.init_ui()

    def init_ui(self):
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; }")
        self.image_label.setText("이미지를 열거나 화면을 캡처하세요")
        self.image_label.setStyleSheet(
            "QLabel { background-color: #2b2b2b; color: #888; font-size: 14pt; }"
        )

        self.setWidget(self.image_label)

    def set_image(self, image):
        if isinstance(image, Image.Image):
            self.current_image = image
            self.display_image()
        elif isinstance(image, np.ndarray):
            self.current_image = Image.fromarray(image)
            self.display_image()
        elif isinstance(image, str):
            try:
                self.current_image = Image.open(image)
                self.display_image()
            except Exception as e:
                print(f"이미지 열기 실패: {e}")

    def display_image(self):
        if self.current_image is None:
            return

        img = self.current_image.copy()

        viewer_width = self.viewport().width()
        viewer_height = self.viewport().height()

        img_width, img_height = img.size

        # 기본 맞춤 스케일 계산
        fit_scale = min(viewer_width / img_width, viewer_height / img_height, 1.0)

        # zoom_factor 적용
        final_scale = fit_scale * self.zoom_factor

        if final_scale != 1.0:
            new_width = int(img_width * final_scale * 0.95)
            new_height = int(img_height * final_scale * 0.95)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if img.mode == "RGB":
            data = img.tobytes("raw", "RGB")
            qimage = QImage(
                data, img.width, img.height, img.width * 3, QImage.Format_RGB888
            )
        elif img.mode == "RGBA":
            data = img.tobytes("raw", "RGBA")
            qimage = QImage(
                data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888
            )
        else:
            img = img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimage = QImage(
                data, img.width, img.height, img.width * 3, QImage.Format_RGB888
            )

        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.setStyleSheet("")

        # zoom_changed 시그널 발생
        if hasattr(self, "zoom_changed"):
            self.zoom_changed.emit(self.zoom_factor)

    def get_image(self):
        return self.current_image

    def clear_image(self):
        self.current_image = None
        self.image_label.clear()
        self.image_label.setText("이미지를 열거나 화면을 캡처하세요")
        self.image_label.setStyleSheet(
            "QLabel { background-color: #2b2b2b; color: #888; font-size: 14pt; }"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image:
            self.display_image()

    def wheelEvent(self, event):
        """CTRL+휠로 확대/축소"""
        if event.modifiers() == Qt.ControlModifier:
            # 휠 델타값 (보통 ±120)
            delta = event.angleDelta().y()

            if delta > 0:
                # 확대 (10% 증가)
                self.set_zoom(self.zoom_factor * 1.1)
            elif delta < 0:
                # 축소 (10% 감소)
                self.set_zoom(self.zoom_factor / 1.1)

            event.accept()
        else:
            super().wheelEvent(event)

    def set_zoom(self, factor):
        """배율 설정 (0.1 ~ 5.0 범위)"""
        self.zoom_factor = max(0.1, min(5.0, factor))
        if self.current_image:
            self.display_image()

    def reset_zoom(self):
        """배율 초기화 (100%)"""
        self.set_zoom(1.0)

    def get_zoom(self):
        """현재 배율 반환"""
        return self.zoom_factor

    def mousePressEvent(self, event):
        """마우스 누름 - 패닝 시작"""
        if event.button() == Qt.LeftButton:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """마우스 이동 - 패닝 중"""
        if self.panning and self.pan_start_pos:
            # 이동 거리 계산
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()

            # 스크롤바 이동
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()

            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """마우스 놓음 - 패닝 종료"""
        if event.button() == Qt.LeftButton:
            self.panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
