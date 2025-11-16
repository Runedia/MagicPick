from PyQt5.QtWidgets import QLabel, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np


class ImageViewer(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_image = None
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
        scale = min(viewer_width / img_width, viewer_height / img_height, 1.0)

        if scale < 1.0:
            new_width = int(img_width * scale * 0.95)
            new_height = int(img_height * scale * 0.95)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if img.mode == 'RGB':
            data = img.tobytes("raw", "RGB")
            qimage = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
        elif img.mode == 'RGBA':
            data = img.tobytes("raw", "RGBA")
            qimage = QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
        else:
            img = img.convert('RGB')
            data = img.tobytes("raw", "RGB")
            qimage = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        self.image_label.setStyleSheet("")

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
