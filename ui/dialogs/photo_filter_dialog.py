"""
Photo Filter 조정 다이얼로그

사용자가 Photo Filter 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from filters.photo_filter import PhotoFilter


class PhotoFilterDialog(QDialog):
    """Photo Filter 조정 다이얼로그"""

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
        self.photo_filter = PhotoFilter()

        self.init_ui()

        # 초기 필터 적용
        self.apply_filter()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Photo Filter")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 필터 선택 그룹
        filter_group = QGroupBox("Filter")
        filter_layout = QVBoxLayout()

        # 필터 프리셋 선택
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Filter:"))

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(PhotoFilter.get_preset_names())
        self.filter_combo.currentTextChanged.connect(self.on_settings_changed)
        preset_layout.addWidget(self.filter_combo)

        filter_layout.addLayout(preset_layout)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Density 슬라이더
        density_group = QGroupBox("Density")
        density_layout = QVBoxLayout()

        self.density_label = QLabel("25%")
        self.density_label.setAlignment(Qt.AlignCenter)
        density_layout.addWidget(self.density_label)

        self.density_slider = QSlider(Qt.Horizontal)
        self.density_slider.setMinimum(0)
        self.density_slider.setMaximum(100)
        self.density_slider.setValue(25)
        self.density_slider.setTickPosition(QSlider.TicksBelow)
        self.density_slider.setTickInterval(10)
        self.density_slider.valueChanged.connect(self.on_density_changed)
        density_layout.addWidget(self.density_slider)

        density_group.setLayout(density_layout)
        layout.addWidget(density_group)

        # Preserve Luminosity 체크박스
        self.preserve_luminosity = QCheckBox("Preserve Luminosity")
        self.preserve_luminosity.setChecked(True)
        self.preserve_luminosity.stateChanged.connect(self.on_settings_changed)
        layout.addWidget(self.preserve_luminosity)

        # 미리보기 영역 (선택사항)
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        self.preview_label = QLabel("미리보기가 메인 창에 표시됩니다")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(60)
        self.preview_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }"
        )
        preview_layout.addWidget(self.preview_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # 버튼
        button_layout = QHBoxLayout()

        self.reset_button = QPushButton("초기화")
        self.reset_button.clicked.connect(self.reset_settings)
        button_layout.addWidget(self.reset_button)

        button_layout.addStretch()

        self.ok_button = QPushButton("확인")
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def on_density_changed(self, value):
        """Density 슬라이더 변경 이벤트"""
        self.density_label.setText(f"{value}%")
        self.on_settings_changed()

    def on_settings_changed(self):
        """설정 변경 시 필터 재적용"""
        self.apply_filter()

    def apply_filter(self):
        """현재 설정으로 필터 적용"""
        filter_name = self.filter_combo.currentText()
        density = self.density_slider.value() / 100.0
        preserve_luminosity = self.preserve_luminosity.isChecked()

        # 필터 적용
        self.filtered_image = self.photo_filter.apply(
            self.original_image,
            filter_name=filter_name,
            density=density,
            preserve_luminosity=preserve_luminosity,
        )

        # 메인 창에 미리보기 표시를 위해 시그널 발생
        self.filter_applied.emit(self.filtered_image)

    def reset_settings(self):
        """설정 초기화"""
        self.filter_combo.setCurrentIndex(0)
        self.density_slider.setValue(25)
        self.preserve_luminosity.setChecked(True)

    def get_filtered_image(self):
        """필터가 적용된 이미지 반환"""
        return self.filtered_image

    def get_settings(self):
        """현재 설정 반환"""
        return {
            "filter_name": self.filter_combo.currentText(),
            "density": self.density_slider.value() / 100.0,
            "preserve_luminosity": self.preserve_luminosity.isChecked(),
        }
