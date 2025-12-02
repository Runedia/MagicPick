"""
Curves - S-커브를 사용하여 대비를 증가시킵니다
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class CurvesFilterAccurate(BaseFilter):
    """Curves - S-커브를 사용하여 대비를 증가시킵니다"""

    def __init__(self):
        super().__init__("Curves", "커브")

        self.mode = 0
        self.formula = 4
        self.contrast = 0.65

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.mode = params.get("Mode", self.mode)
        self.formula = params.get("Formula", self.formula)
        self.contrast = params.get("Contrast", self.contrast)

        img_float = image.astype(np.float32) / 255.0

        lum_coeff = np.array([0.2126, 0.7152, 0.0722])
        contrast_blend = self.contrast
        PI = np.pi

        luma = np.sum(img_float * lum_coeff, axis=2, keepdims=True)
        chroma = img_float - luma

        if self.mode == 0:
            x = luma
        elif self.mode == 1:
            x = chroma * 0.5 + 0.5
        else:
            x = img_float

        if self.formula == 0:
            x = np.sin(PI * 0.5 * x)
            x *= x
        elif self.formula == 1:
            x = x - 0.5
            x = (x / (0.5 + np.abs(x))) + 0.5
        elif self.formula == 2:
            x = x * x * (3.0 - 2.0 * x)
        elif self.formula == 3:
            x = (1.0524 * np.exp(6.0 * x) - 1.05248) / (np.exp(6.0 * x) + 20.0855)
        elif self.formula == 4:
            x = x * (x * (1.5 - x) + 0.5)
            contrast_blend = self.contrast * 2.0
        elif self.formula == 5:
            x = x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

        if self.mode == 0:
            color = luma + chroma
            color = lerp(color, x + chroma, contrast_blend)
        elif self.mode == 1:
            x = x * 2.0 - 1.0
            color = lerp(luma + chroma, luma + x, contrast_blend)
        else:
            color = lerp(img_float, x, contrast_blend)

        return (saturate(color) * 255).astype(np.uint8)
