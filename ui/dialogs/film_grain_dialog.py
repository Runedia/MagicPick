"""
필름 그레인 효과 조정 다이얼로그

사용자가 필름 그레인 효과 설정을 조정할 수 있는 다이얼로그입니다.
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

from filters.artistic import FilmGrainFilter


class FilmGrainDialog(QDialog):
    """필름 그레인 효과 조정 다이얼로그"""

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
        self.grain_filter = FilmGrainFilter()

        self.init_ui()
        self._initial_filter_applied = False

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("필름 그레인")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 그레인 타입 선택
        type_group = QGroupBox("그레인 타입")
        type_layout = QHBoxLayout()

        type_layout.addWidget(QLabel("타입:"))

        self.type_combo = QComboBox()
        self.type_combo.addItems(["가우시안 (Gaussian)", "균일 (Uniform)"])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        type_layout.addWidget(self.type_combo)

        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Intensity 슬라이더
        intensity_group = QGroupBox("강도")
        intensity_layout = QVBoxLayout()

        self.intensity_label = QLabel("25")
        self.intensity_label.setAlignment(Qt.AlignCenter)
        intensity_layout.addWidget(self.intensity_label)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(5)
        self.intensity_slider.setMaximum(100)
        self.intensity_slider.setValue(25)
        self.intensity_slider.setTickPosition(QSlider.TicksBelow)
        self.intensity_slider.setTickInterval(10)
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        intensity_layout.addWidget(self.intensity_slider)

        intensity_group.setLayout(intensity_layout)
        layout.addWidget(intensity_group)

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
        """그레인 타입 변경"""
        self.apply_filter()

    def on_intensity_changed(self, value):
        """강도 슬라이더 변경"""
        self.intensity_label.setText(str(value))
        self.apply_filter()

    def apply_filter(self):
        """필터 적용"""
        intensity = self.intensity_slider.value()
        grain_type = "gaussian" if self.type_combo.currentIndex() == 0 else "uniform"

        self.filtered_image = self.grain_filter.apply(
            self.original_image, intensity=intensity, grain_type=grain_type
        )

        self.filter_applied.emit(self.filtered_image)

    def get_filtered_image(self):
        """필터 적용된 이미지 반환"""
        return self.filtered_image

    def get_settings(self):
        """현재 설정 반환"""
        return {
            "intensity": self.intensity_slider.value(),
            "grain_type": "gaussian"
            if self.type_combo.currentIndex() == 0
            else "uniform",
        }
