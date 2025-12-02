"""
Vibrance - 정확한 구현

Original HLSL shader by Christian Cann Schuldt Jensen (CeeJay.dk)
Python/NumPy port for static image processing
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import lerp


@njit(parallel=True, fastmath=True, cache=True)
def _vibrance_pass(img, coeff_vibrance):
    rows, cols, channels = img.shape
    output = np.empty_like(img)

    # Luma coefficients (Rec.709 - Accurate)
    coef_luma_r = 0.212656
    coef_luma_g = 0.715158
    coef_luma_b = 0.072186

    for y in prange(rows):
        for x in range(cols):
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]

            luma = r * coef_luma_r + g * coef_luma_g + b * coef_luma_b

            max_color = max(r, max(g, b))
            min_color = min(r, min(g, b))

            color_saturation = max_color - min_color

            for c in range(3):
                coeff = coeff_vibrance[c]
                sign_coeff = np.sign(coeff)

                factor = 1.0 + (coeff * (1.0 - (sign_coeff * color_saturation)))

                # lerp(luma, val, factor) -> luma + factor * (val - luma)
                val = img[y, x, c]
                output[y, x, c] = lerp(luma, val, factor)

    return output


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
        self.vibrance_rgb_balance = np.array(
            params.get("vibrance_rgb_balance", self.vibrance_rgb_balance),
            dtype=np.float32,
        )

        img_float = image.astype(np.float32) / 255.0

        coeff_vibrance = self.vibrance_rgb_balance * self.vibrance

        result = _vibrance_pass(img_float, coeff_vibrance)

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
