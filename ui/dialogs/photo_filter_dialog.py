"""
Photo Filter 조정 다이얼로그

사용자가 Photo Filter 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QGroupBox, QLabel, QVBoxLayout

from filters.photo_filter import PhotoFilter
from ui.dialogs.base_filter_dialog import BaseFilterDialog


class PhotoFilterDialog(BaseFilterDialog):
    """Photo Filter 조정 다이얼로그"""

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(original_image, parent)
        self.setWindowTitle("Photo Filter")

    def create_filter(self):
        """PhotoFilter 인스턴스 생성"""
        return PhotoFilter()

    def build_parameter_ui(self, layout):
        """파라미터 UI 구성"""
        # 필터 선택 콤보박스
        filter_group, self.filter_combo = self._create_combo_box(
            "Filter",
            "Filter:",
            PhotoFilter.get_preset_names(),
            lambda idx: self.apply_filter(),
        )
        layout.addWidget(filter_group)

        # Density 슬라이더
        density_group, self.density_slider, _ = self._create_slider_with_label(
            "Density", 0, 100, 25, lambda v: self.apply_filter(), suffix="%"
        )
        layout.addWidget(density_group)

        # Preserve Luminosity 체크박스
        self.preserve_luminosity = QCheckBox("Preserve Luminosity")
        self.preserve_luminosity.setChecked(True)
        self.preserve_luminosity.stateChanged.connect(lambda: self.apply_filter())
        layout.addWidget(self.preserve_luminosity)

        # 미리보기 안내
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        preview_label = QLabel("미리보기가 메인 창에 표시됩니다")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setMinimumHeight(60)
        preview_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }"
        )
        preview_layout.addWidget(preview_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

    def get_filter_params(self):
        """필터 파라미터 반환"""
        return {
            "filter_name": self.filter_combo.currentText(),
            "density": self.density_slider.value() / 100.0,
            "preserve_luminosity": self.preserve_luminosity.isChecked(),
        }
