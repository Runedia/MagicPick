"""
카툰 효과 조정 다이얼로그

사용자가 카툰 효과 설정을 조정할 수 있는 다이얼로그입니다.
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

from filters.artistic import CartoonFilter


class CartoonDialog(QDialog):
    """카툰 효과 조정 다이얼로그"""

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
        self.cartoon_filter = CartoonFilter()

        # 마우스 드래그 상태 플래그
        self.is_dragging = False

        self.init_ui()
        self._initial_filter_applied = False

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("카툰 효과")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Color Smoothness 슬라이더
        smoothness_group = QGroupBox("색상 평탄화")
        smoothness_layout = QVBoxLayout()

        self.smoothness_label = QLabel("75")
        self.smoothness_label.setAlignment(Qt.AlignCenter)
        smoothness_layout.addWidget(self.smoothness_label)

        self.smoothness_slider = QSlider(Qt.Horizontal)
        self.smoothness_slider.setMinimum(10)
        self.smoothness_slider.setMaximum(150)
        self.smoothness_slider.setValue(75)
        self.smoothness_slider.setTickPosition(QSlider.TicksBelow)
        self.smoothness_slider.setTickInterval(20)
        self.smoothness_slider.valueChanged.connect(self.on_smoothness_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.smoothness_slider.sliderPressed.connect(self.on_slider_pressed)
        self.smoothness_slider.sliderReleased.connect(self.on_slider_released)
        smoothness_layout.addWidget(self.smoothness_slider)

        smoothness_group.setLayout(smoothness_layout)
        layout.addWidget(smoothness_group)

        # Edge Threshold 슬라이더
        edge_group = QGroupBox("윤곽선 강도")
        edge_layout = QVBoxLayout()

        self.edge_label = QLabel("100")
        self.edge_label.setAlignment(Qt.AlignCenter)
        edge_layout.addWidget(self.edge_label)

        self.edge_slider = QSlider(Qt.Horizontal)
        self.edge_slider.setMinimum(50)
        self.edge_slider.setMaximum(200)
        self.edge_slider.setValue(100)
        self.edge_slider.setTickPosition(QSlider.TicksBelow)
        self.edge_slider.setTickInterval(25)
        self.edge_slider.valueChanged.connect(self.on_edge_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.edge_slider.sliderPressed.connect(self.on_slider_pressed)
        self.edge_slider.sliderReleased.connect(self.on_slider_released)
        edge_layout.addWidget(self.edge_slider)

        edge_group.setLayout(edge_layout)
        layout.addWidget(edge_group)

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

    def on_smoothness_changed(self, value):
        """색상 평탄화 슬라이더 변경"""
        self.smoothness_label.setText(str(value))
        if not self.is_dragging:
            self.apply_filter()

    def on_edge_changed(self, value):
        """윤곽선 강도 슬라이더 변경"""
        self.edge_label.setText(str(value))
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
        bilateral_sigma_color = self.smoothness_slider.value()
        bilateral_sigma_space = self.smoothness_slider.value()
        edge_threshold1 = self.edge_slider.value()
        edge_threshold2 = edge_threshold1 * 2

        self.filtered_image = self.cartoon_filter.apply(
            self.original_image,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            edge_threshold1=edge_threshold1,
            edge_threshold2=edge_threshold2,
        )

        self.filter_applied.emit(self.filtered_image)

    def get_filtered_image(self):
        """필터 적용된 이미지 반환"""
        return self.filtered_image

    def get_settings(self):
        """현재 설정 반환"""
        return {
            "bilateral_sigma_color": self.smoothness_slider.value(),
            "bilateral_sigma_space": self.smoothness_slider.value(),
            "edge_threshold1": self.edge_slider.value(),
            "edge_threshold2": self.edge_slider.value() * 2,
        }
