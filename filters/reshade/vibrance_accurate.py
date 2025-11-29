"""
Vibrance - 정확한 구현

Original HLSL shader by Christian Cann Schuldt Jensen (CeeJay.dk)
Python/NumPy port for static image processing
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp


class VibranceFilterAccurate(BaseFilter):
    """
    Vibrance 정확한 구현

    색상이 적은 픽셀은 더 큰 부스트를, 색상이 많은 픽셀은 작은 부스트를 주어
    채도를 지능적으로 향상시킵니다.
    이미 매우 채도가 높은 픽셀의 과포화를 방지합니다.
    """

    def __init__(self):
        super().__init__("Vibrance", "지능형 채도 부스트 (정확)")

        self.vibrance = 0.15
        self.vibrance_rgb_balance = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Vibrance 효과 적용"""
        self.vibrance = params.get("vibrance", self.vibrance)
        self.vibrance_rgb_balance = params.get(
            "vibrance_rgb_balance", self.vibrance_rgb_balance
        )

        img_float = image.astype(np.float32) / 255.0

        coef_luma = np.array([0.212656, 0.715158, 0.072186])

        luma = np.sum(img_float * coef_luma, axis=2, keepdims=True)

        max_color = np.maximum(
            img_float[:, :, 0:1], np.maximum(img_float[:, :, 1:2], img_float[:, :, 2:3])
        )
        min_color = np.minimum(
            img_float[:, :, 0:1], np.minimum(img_float[:, :, 1:2], img_float[:, :, 2:3])
        )

        color_saturation = max_color - min_color

        coeff_vibrance = self.vibrance_rgb_balance * self.vibrance

        result = lerp(
            luma,
            img_float,
            1.0
            + (coeff_vibrance * (1.0 - (np.sign(coeff_vibrance) * color_saturation))),
        )

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
