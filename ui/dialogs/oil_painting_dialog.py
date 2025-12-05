"""
유화 효과 조정 다이얼로그

사용자가 유화 효과 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np

from filters.artistic import OilPaintingFilter
from ui.dialogs.base_filter_dialog import BaseFilterDialog


class OilPaintingDialog(BaseFilterDialog):
    """유화 효과 조정 다이얼로그"""

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(original_image, parent)
        self.setWindowTitle("유화 효과")

    def create_filter(self):
        """OilPaintingFilter 인스턴스 생성"""
        return OilPaintingFilter()

    def build_parameter_ui(self, layout):
        """파라미터 UI 구성"""
        # 붓 크기 슬라이더
        size_group, self.size_slider, _ = self._create_slider_with_label(
            "붓 크기", 1, 15, 7, lambda v: self.apply_filter(), tick_interval=2
        )
        layout.addWidget(size_group)

        # 디테일 슬라이더
        ratio_group, self.ratio_slider, _ = self._create_slider_with_label(
            "디테일", 1, 3, 1, lambda v: self.apply_filter(), tick_interval=1
        )
        layout.addWidget(ratio_group)

    def get_filter_params(self):
        """필터 파라미터 반환"""
        return {
            "size": self.size_slider.value(),
            "dynRatio": self.ratio_slider.value(),
        }
