"""
스케치 효과 조정 다이얼로그

사용자가 스케치 효과 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from filters.artistic import SketchFilter


class SketchDialog(QDialog):
    """스케치 효과 조정 다이얼로그"""

    filter_applied = pyqtSignal(np.ndarray)  # 필터 적용된 이미지

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.original_image = original_image.copy()
        self.filtered_image = None
        self.sketch_filter = SketchFilter()

        self.init_ui()
        self._initial_filter_applied = False

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("스케치 효과")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 스케치 타입 선택
        type_group = QGroupBox("스케치 타입")
        type_layout = QHBoxLayout()

        type_layout.addWidget(QLabel("타입:"))

        self.type_combo = QComboBox()
        self.type_combo.addItems(["연필 (Pencil)", "숯 (Charcoal)"])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        type_layout.addWidget(self.type_combo)

        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Blur Sigma 슬라이더
        blur_group = QGroupBox("부드러움")
        blur_layout = QVBoxLayout()

        self.blur_label = QLabel("50")
        self.blur_label.setAlignment(Qt.AlignCenter)
        blur_layout.addWidget(self.blur_label)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(10)
        self.blur_slider.setMaximum(100)
        self.blur_slider.setValue(50)
        self.blur_slider.setTickPosition(QSlider.TicksBelow)
        self.blur_slider.setTickInterval(10)
        self.blur_slider.valueChanged.connect(self.on_blur_changed)
        blur_layout.addWidget(self.blur_slider)

        blur_group.setLayout(blur_layout)
        layout.addWidget(blur_group)

        # 버튼
        button_layout = QHBoxLayout()

        ok_button = QPushButton("확인")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        cancel_button = QPushButton("취소")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def showEvent(self, event):
        """다이얼로그가 표시될 때 초기 필터 적용"""
        super().showEvent(event)
        if not self._initial_filter_applied:
            self._initial_filter_applied = True
            self.apply_filter()

    def on_type_changed(self, index):
        """스케치 타입 변경"""
        self.apply_filter()

    def on_blur_changed(self, value):
        """부드러움 슬라이더 변경"""
        self.blur_label.setText(str(value))
        self.apply_filter()

    def apply_filter(self):
        """필터 적용"""
        blur_sigma = self.blur_slider.value()
        sketch_type = "pencil" if self.type_combo.currentIndex() == 0 else "charcoal"

        self.filtered_image = self.sketch_filter.apply(
            self.original_image, blur_sigma=blur_sigma, sketch_type=sketch_type
        )

        self.filter_applied.emit(self.filtered_image)

    def get_filtered_image(self):
        """필터 적용된 이미지 반환"""
        return self.filtered_image

    def get_settings(self):
        """현재 설정 반환"""
        return {
            "blur_sigma": self.blur_slider.value(),
            "sketch_type": "pencil"
            if self.type_combo.currentIndex() == 0
            else "charcoal",
        }
