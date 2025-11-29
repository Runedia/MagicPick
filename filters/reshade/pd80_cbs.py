"""
PD80_04_Contrast_Brightness_Saturation.fx 구현
PD80 대비/밝기/채도 조정

원본 셰이더: https://github.com/prod80/prod80-ReShade-Repository
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class PD80CBSFilter(BaseFilter):
    def __init__(self):
        super().__init__("PD80CBS", "대비/밝기/채도 조정")
        self.contrast = 0.0  # -1.0 ~ 1.0
        self.brightness = 0.0  # -1.0 ~ 1.0
        self.saturation = 0.0  # -1.0 ~ 1.0  # -1.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        대비/밝기/채도 조정 적용

        Parameters:
        -----------
        contrast : float
            대비 조정 (-1.0 ~ 1.0)
        brightness : float
            밝기 조정 (-1.0 ~ 1.0)
        saturation : float
            채도 조정 (-1.0 ~ 1.0)
        """
        self.contrast = params.get("contrast", self.contrast)
        self.brightness = params.get("brightness", self.brightness)
        self.saturation = params.get("saturation", self.saturation)

        img_float = image.astype(np.float32) / 255.0

        # 1. 밝기 조정
        result = img_float + self.brightness

        # 2. 대비 조정
        # contrast_factor = (1 + contrast)
        contrast_factor = 1.0 + self.contrast
        result = (result - 0.5) * contrast_factor + 0.5

        # 3. 채도 조정
        luma = np.dot(result, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)

        saturation_factor = 1.0 + self.saturation
        result = lerp(luma, result, saturation_factor)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
