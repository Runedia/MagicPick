"""
필름 그레인 효과 조정 다이얼로그

사용자가 필름 그레인 효과 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np

from filters.artistic import FilmGrainFilter
from ui.dialogs.base_filter_dialog import BaseFilterDialog


class FilmGrainDialog(BaseFilterDialog):
    """필름 그레인 효과 조정 다이얼로그"""

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(original_image, parent)
        self.setWindowTitle("필름 그레인")

    def create_filter(self):
        """FilmGrainFilter 인스턴스 생성"""
        return FilmGrainFilter()

    def build_parameter_ui(self, layout):
        """파라미터 UI 구성"""
        # 그레인 타입 선택
        type_group, self.type_combo = self._create_combo_box(
            "그레인 타입",
            "타입:",
            ["가우시안 (Gaussian)", "균일 (Uniform)"],
            lambda idx: self.apply_filter(),
        )
        layout.addWidget(type_group)

        # 강도 슬라이더
        intensity_group, self.intensity_slider, _ = self._create_slider_with_label(
            "강도", 5, 100, 25, lambda v: self.apply_filter(), tick_interval=10
        )
        layout.addWidget(intensity_group)

    def get_filter_params(self):
        """필터 파라미터 반환"""
        intensity = self.intensity_slider.value()
        grain_type = "gaussian" if self.type_combo.currentIndex() == 0 else "uniform"

        return {"intensity": intensity, "grain_type": grain_type}
