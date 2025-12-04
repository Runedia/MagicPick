"""
빈티지 효과 조정 다이얼로그

사용자가 빈티지 효과 설정을 조정할 수 있는 다이얼로그입니다.
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

from filters.artistic import VintageFilter


class VintageDialog(QDialog):
    """빈티지 효과 조정 다이얼로그"""

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
        self.vintage_filter = VintageFilter()

        # 마우스 드래그 상태 플래그
        self.is_dragging = False

        self.init_ui()
        self._initial_filter_applied = False

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("빈티지 효과")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Sepia Intensity 슬라이더
        sepia_group = QGroupBox("세피아 강도")
        sepia_layout = QVBoxLayout()

        self.sepia_label = QLabel("70%")
        self.sepia_label.setAlignment(Qt.AlignCenter)
        sepia_layout.addWidget(self.sepia_label)

        self.sepia_slider = QSlider(Qt.Horizontal)
        self.sepia_slider.setMinimum(0)
        self.sepia_slider.setMaximum(100)
        self.sepia_slider.setValue(70)
        self.sepia_slider.setTickPosition(QSlider.TicksBelow)
        self.sepia_slider.setTickInterval(10)
        self.sepia_slider.valueChanged.connect(self.on_sepia_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.sepia_slider.sliderPressed.connect(self.on_slider_pressed)
        self.sepia_slider.sliderReleased.connect(self.on_slider_released)
        sepia_layout.addWidget(self.sepia_slider)

        sepia_group.setLayout(sepia_layout)
        layout.addWidget(sepia_group)

        # Vignette Strength 슬라이더
        vignette_group = QGroupBox("비네팅 강도")
        vignette_layout = QVBoxLayout()

        self.vignette_label = QLabel("50%")
        self.vignette_label.setAlignment(Qt.AlignCenter)
        vignette_layout.addWidget(self.vignette_label)

        self.vignette_slider = QSlider(Qt.Horizontal)
        self.vignette_slider.setMinimum(0)
        self.vignette_slider.setMaximum(100)
        self.vignette_slider.setValue(50)
        self.vignette_slider.setTickPosition(QSlider.TicksBelow)
        self.vignette_slider.setTickInterval(10)
        self.vignette_slider.valueChanged.connect(self.on_vignette_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.vignette_slider.sliderPressed.connect(self.on_slider_pressed)
        self.vignette_slider.sliderReleased.connect(self.on_slider_released)
        vignette_layout.addWidget(self.vignette_slider)

        vignette_group.setLayout(vignette_layout)
        layout.addWidget(vignette_group)

        # Grain Intensity 슬라이더
        grain_group = QGroupBox("그레인 강도")
        grain_layout = QVBoxLayout()

        self.grain_label = QLabel("15")
        self.grain_label.setAlignment(Qt.AlignCenter)
        grain_layout.addWidget(self.grain_label)

        self.grain_slider = QSlider(Qt.Horizontal)
        self.grain_slider.setMinimum(0)
        self.grain_slider.setMaximum(50)
        self.grain_slider.setValue(15)
        self.grain_slider.setTickPosition(QSlider.TicksBelow)
        self.grain_slider.setTickInterval(10)
        self.grain_slider.valueChanged.connect(self.on_grain_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.grain_slider.sliderPressed.connect(self.on_slider_pressed)
        self.grain_slider.sliderReleased.connect(self.on_slider_released)
        grain_layout.addWidget(self.grain_slider)

        grain_group.setLayout(grain_layout)
        layout.addWidget(grain_group)

        # Fade Amount 슬라이더
        fade_group = QGroupBox("페이드")
        fade_layout = QVBoxLayout()

        self.fade_label = QLabel("20%")
        self.fade_label.setAlignment(Qt.AlignCenter)
        fade_layout.addWidget(self.fade_label)

        self.fade_slider = QSlider(Qt.Horizontal)
        self.fade_slider.setMinimum(0)
        self.fade_slider.setMaximum(100)
        self.fade_slider.setValue(20)
        self.fade_slider.setTickPosition(QSlider.TicksBelow)
        self.fade_slider.setTickInterval(10)
        self.fade_slider.valueChanged.connect(self.on_fade_changed)
        # 마우스 드래그 시작/종료 시그널 연결
        self.fade_slider.sliderPressed.connect(self.on_slider_pressed)
        self.fade_slider.sliderReleased.connect(self.on_slider_released)
        fade_layout.addWidget(self.fade_slider)

        fade_group.setLayout(fade_layout)
        layout.addWidget(fade_group)

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

    def on_sepia_changed(self, value):
        """세피아 강도 슬라이더 변경"""
        self.sepia_label.setText(f"{value}%")
        if not self.is_dragging:
            self.apply_filter()

    def on_vignette_changed(self, value):
        """비네팅 강도 슬라이더 변경"""
        self.vignette_label.setText(f"{value}%")
        if not self.is_dragging:
            self.apply_filter()

    def on_grain_changed(self, value):
        """그레인 강도 슬라이더 변경"""
        self.grain_label.setText(str(value))
        if not self.is_dragging:
            self.apply_filter()

    def on_fade_changed(self, value):
        """페이드 슬라이더 변경"""
        self.fade_label.setText(f"{value}%")
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
        sepia_intensity = self.sepia_slider.value() / 100.0
        vignette_strength = self.vignette_slider.value() / 100.0
        grain_intensity = self.grain_slider.value()
        fade_amount = self.fade_slider.value() / 100.0

        self.filtered_image = self.vintage_filter.apply(
            self.original_image,
            sepia_intensity=sepia_intensity,
            vignette_strength=vignette_strength,
            grain_intensity=grain_intensity,
            fade_amount=fade_amount,
        )

        self.filter_applied.emit(self.filtered_image)

    def get_filtered_image(self):
        """필터 적용된 이미지 반환"""
        return self.filtered_image

    def get_settings(self):
        """현재 설정 반환"""
        return {
            "sepia_intensity": self.sepia_slider.value() / 100.0,
            "vignette_strength": self.vignette_slider.value() / 100.0,
            "grain_intensity": self.grain_slider.value(),
            "fade_amount": self.fade_slider.value() / 100.0,
        }
