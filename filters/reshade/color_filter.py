"""
ColorFilter.fx 구현

색상 필터 오버레이
유사한 ColorMatrix 필터를 기반으로 구현
"""

import numpy as np

from filters.base_filter import BaseFilter


class ColorFilterFilter(BaseFilter):
    """
    ColorFilter - 색상 필터 오버레이

    Features:
    - RGB 색상 필터 적용
    - 강도 조절
    - Multiply 블렌드 모드
    """

    def __init__(self):
        super().__init__("ColorFilter", "색상 필터")

        # Parameters
        self.filter_color = np.array(
            [1.0, 0.5, 0.0], dtype=np.float32
        )  # RGB filter color
        self.strength = 0.5  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply ColorFilter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - filter_color: RGB filter color
                    ([0-1, 0-1, 0-1], default [1.0, 0.5, 0.0])
                - strength: Filter strength (0.0 ~ 1.0, default 0.5)

        Returns:
            Color filtered image (uint8, 0-255)
        """
        # Update parameters
        if "filter_color" in params:
            self.filter_color = np.array(params["filter_color"], dtype=np.float32)
        self.strength = params.get("strength", self.strength)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        # Apply color filter (Multiply blend)
        filtered = img_float * self.filter_color

        # Blend with original based on strength
        result = img_float + (filtered - img_float) * self.strength

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
