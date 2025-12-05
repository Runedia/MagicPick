"""
스케치 효과 조정 다이얼로그

사용자가 스케치 효과 설정을 조정할 수 있는 다이얼로그입니다.
"""

import numpy as np

from filters.artistic import SketchFilter
from ui.dialogs.base_filter_dialog import BaseFilterDialog


class SketchDialog(BaseFilterDialog):
    """스케치 효과 조정 다이얼로그"""

    def __init__(self, original_image: np.ndarray, parent=None):
        """
        초기화

        Args:
            original_image: 원본 이미지 (NumPy array, RGB 형식)
            parent: 부모 위젯
        """
        super().__init__(original_image, parent)
        self.setWindowTitle("스케치 효과")

    def create_filter(self):
        """SketchFilter 인스턴스 생성"""
        return SketchFilter()

    def build_parameter_ui(self, layout):
        """파라미터 UI 구성"""
        # 스케치 타입 선택
        type_group, self.type_combo = self._create_combo_box(
            "스케치 타입",
            "타입:",
            ["연필 (Pencil)", "숯 (Charcoal)"],
            lambda idx: self.apply_filter(),
        )
        layout.addWidget(type_group)

        # 부드러움 슬라이더
        blur_group, self.blur_slider, _ = self._create_slider_with_label(
            "부드러움", 10, 100, 50, lambda v: self.apply_filter(), tick_interval=10
        )
        layout.addWidget(blur_group)

    def get_filter_params(self):
        """필터 파라미터 반환"""
        blur_sigma = self.blur_slider.value()
        sketch_type = "pencil" if self.type_combo.currentIndex() == 0 else "charcoal"

        return {"blur_sigma": blur_sigma, "sketch_type": sketch_type}
