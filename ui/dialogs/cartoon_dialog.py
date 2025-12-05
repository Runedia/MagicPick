"""
카툰 효과 조정 다이얼로그

사용자가 카툰 효과 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np

from filters.artistic import CartoonFilter
from ui.dialogs.base_filter_dialog import BaseFilterDialog


class CartoonDialog(BaseFilterDialog):
    """카툰 효과 조정 다이얼로그"""

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(original_image, parent)
        self.setWindowTitle("카툰 효과")

    def create_filter(self):
        """CartoonFilter 인스턴스 생성"""
        return CartoonFilter()

    def build_parameter_ui(self, layout):
        """파라미터 UI 구성"""
        # 색상 평탄화 슬라이더
        smoothness_group, self.smoothness_slider, _ = self._create_slider_with_label(
            "색상 평탄화", 10, 150, 75, lambda v: self.apply_filter(), tick_interval=20
        )
        layout.addWidget(smoothness_group)

        # 윤곽선 강도 슬라이더
        edge_group, self.edge_slider, _ = self._create_slider_with_label(
            "윤곽선 강도", 50, 200, 100, lambda v: self.apply_filter(), tick_interval=25
        )
        layout.addWidget(edge_group)

    def get_filter_params(self):
        """필터 파라미터 반환"""
        smoothness = self.smoothness_slider.value()
        edge = self.edge_slider.value()

        return {
            "bilateral_sigma_color": smoothness,
            "bilateral_sigma_space": smoothness,
            "edge_threshold1": edge,
            "edge_threshold2": edge * 2,
        }
