"""
유화 효과 조정 다이얼로그

사용자가 유화 효과 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from filters.artistic import OilPaintingFilter


class OilPaintingDialog(QDialog):
    """유화 효과 조정 다이얼로그"""

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
        self.oil_filter = OilPaintingFilter()

        # 마우스 드래그 상태 플래그
        self.is_dragging = False

        self.init_ui()
        self._initial_filter_applied = False

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("유화 효과")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Size 슬라이더
        size_group = QGroupBox("붓 크기")
        size_layout = QVBoxLayout()

        self.size_label = QLabel("7")
        self.size_label.setAlignment(Qt.AlignCenter)
        size_layout.addWidget(self.size_label)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(1)
        self.size_slider.setMaximum(15)
        self.size_slider.setValue(7)
        self.size_slider.setTickPosition(QSlider.TicksBelow)
        self.size_slider.setTickInterval(2)
        self.size_slider.valueChanged.connect(self.on_size_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.size_slider.sliderPressed.connect(self.on_slider_pressed)
        self.size_slider.sliderReleased.connect(self.on_slider_released)
        size_layout.addWidget(self.size_slider)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Dynamic Ratio 슬라이더
        ratio_group = QGroupBox("디테일")
        ratio_layout = QVBoxLayout()

        self.ratio_label = QLabel("1")
        self.ratio_label.setAlignment(Qt.AlignCenter)
        ratio_layout.addWidget(self.ratio_label)

        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setMinimum(1)
        self.ratio_slider.setMaximum(3)
        self.ratio_slider.setValue(1)
        self.ratio_slider.setTickPosition(QSlider.TicksBelow)
        self.ratio_slider.setTickInterval(1)
        self.ratio_slider.valueChanged.connect(self.on_ratio_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.ratio_slider.sliderPressed.connect(self.on_slider_pressed)
        self.ratio_slider.sliderReleased.connect(self.on_slider_released)
        ratio_layout.addWidget(self.ratio_slider)

        ratio_group.setLayout(ratio_layout)
        layout.addWidget(ratio_group)

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

    def on_size_changed(self, value):
        """붓 크기 슬라이더 변경"""
        self.size_label.setText(str(value))
        if not self.is_dragging:
            self.apply_filter()

    def on_ratio_changed(self, value):
        """디테일 슬라이더 변경"""
        self.ratio_label.setText(str(value))
        if not self.is_dragging:
            self.apply_filter()

    def on_slider_pressed(self):
        """슬라이더 마우스 드래그 시작"""
        self.is_dragging = True

    def on_slider_released(self):
        """슬라이더 마우스 드래그 종료 시 필터 적용"""
        self.is_dragging = False
        self.apply_filter()

    def apply_filter(self):
        """필터 적용"""
        size = self.size_slider.value()
        dynRatio = self.ratio_slider.value()

        self.filtered_image = self.oil_filter.apply(
            self.original_image, size=size, dynRatio=dynRatio
        )

        self.filter_applied.emit(self.filtered_image)

    def get_filtered_image(self):
        """필터 적용된 이미지 반환"""
        return self.filtered_image

    def get_settings(self):
        """현재 설정 반환"""
        return {"size": self.size_slider.value(), "dynRatio": self.ratio_slider.value()}
