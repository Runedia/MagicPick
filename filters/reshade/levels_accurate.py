"""
Levels - 정확한 구현

Original HLSL shader by Christian Cann Schuldt Jensen (CeeJay.dk)
Python/NumPy port for static image processing
"""

import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import saturate


class LevelsFilterAccurate(BaseFilter):
    """
    Levels 정확한 구현

    새로운 블랙 포인트와 화이트 포인트를 설정하여 대비를 증가시킵니다.
    TV 범위(16-235)를 PC 범위(0-255)로 확장하는 데 유용합니다.
    """

    def __init__(self):
        super().__init__("Levels", "레벨 조정 (정확)")

        self.black_point = 16
        self.white_point = 235
        self.highlight_clipping = False

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Levels 효과 적용"""
        self.black_point = params.get("black_point", self.black_point)
        self.white_point = params.get("white_point", self.white_point)
        self.highlight_clipping = params.get(
            "highlight_clipping", self.highlight_clipping
        )

        img_float = image.astype(np.float32) / 255.0

        black_point_float = self.black_point / 255.0

        if self.white_point == self.black_point:
            white_point_float = 255.0 / 0.00025
        else:
            white_point_float = 255.0 / (self.white_point - self.black_point)

        color = img_float * white_point_float - (black_point_float * white_point_float)

        if self.highlight_clipping:
            color = self._apply_clipping_visualization(color)

        return (saturate(color) * 255).astype(np.uint8)

    def _apply_clipping_visualization(self, color):
        """
        클리핑 영역 시각화

        Red: 일부 디테일이 하이라이트에서 손실
        Yellow: 모든 디테일이 하이라이트에서 손실
        Blue: 일부 디테일이 섀도우에서 손실
        Cyan: 모든 디테일이 섀도우에서 손실
        """
        h, w = color.shape[:2]
        clipped_colors = color.copy()

        saturated = saturate(color)

        any_whiter = np.any(color > saturated, axis=2, keepdims=True)
        all_whiter = np.all(color > saturated, axis=2, keepdims=True)
        any_blacker = np.any(color < saturated, axis=2, keepdims=True)
        all_blacker = np.all(color < saturated, axis=2, keepdims=True)

        clipped_colors = np.where(
            all_blacker, np.array([0.0, 1.0, 1.0]), clipped_colors
        )

        clipped_colors = np.where(
            any_blacker & ~all_blacker, np.array([0.0, 0.0, 1.0]), clipped_colors
        )

        clipped_colors = np.where(all_whiter, np.array([1.0, 1.0, 0.0]), clipped_colors)

        clipped_colors = np.where(
            any_whiter & ~all_whiter, np.array([1.0, 0.0, 0.0]), clipped_colors
        )

        return clipped_colors
