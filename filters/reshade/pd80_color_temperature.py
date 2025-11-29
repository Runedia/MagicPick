"""
PD80_04_Color_Temperature.fx 구현
PD80 색온도 조정

원본 셰이더: https://github.com/prod80/prod80-ReShade-Repository
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80ColorTemperatureFilter(BaseFilter):
    def __init__(self):
        super().__init__("PD80ColorTemperature", "색온도 조정")
        self.temperature_kelvin = 6500.0  # 1000.0 ~ 15000.0
        self.strength = 1.0  # 0.0 ~ 1.0  # -1.0 ~ 1.0

    def _kelvin_to_rgb(self, kelvin):
        """
        색온도(K)를 RGB로 변환
        """
        temp = kelvin / 100.0

        # Red 계산
        if temp <= 66.0:
            red = 1.0
        else:
            red = temp - 60.0
            red = 329.698727446 * np.power(red, -0.1332047592)
            red = np.clip(red / 255.0, 0.0, 1.0)

        # Green 계산
        if temp <= 66.0:
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            green = temp - 60.0
            green = 288.1221695283 * np.power(green, -0.0755148492)
        green = np.clip(green / 255.0, 0.0, 1.0)

        # Blue 계산
        if temp >= 66.0:
            blue = 1.0
        elif temp <= 19.0:
            blue = 0.0
        else:
            blue = temp - 10.0
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
            blue = np.clip(blue / 255.0, 0.0, 1.0)

        return np.array([blue, green, red], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        색온도 조정 적용

        Parameters:
        -----------
        temperature : float
            색온도 (1000.0 ~ 40000.0 K)
            낮은 값: 따뜻한 색감 (주황색)
            높은 값: 차가운 색감 (파란색)
        tint : float
            틴트 조정 (-1.0 ~ 1.0)
            음수: 녹색, 양수: 마젠타
        """
        self.temperature = params.get("temperature", self.temperature)
        self.tint = params.get("tint", self.tint)

        img_float = image.astype(np.float32) / 255.0

        # 색온도 RGB 계산
        temp_rgb = self._kelvin_to_rgb(self.temperature)

        # 색온도 적용
        result = img_float * temp_rgb

        # Tint 적용 (Green-Magenta axis)
        if self.tint != 0.0:
            # Green channel 조정
            tint_factor = self.tint
            result[:, :, 1] *= 1.0 - tint_factor * 0.5  # Green
            result[:, :, 0] *= 1.0 + tint_factor * 0.25  # Blue (Magenta)
            result[:, :, 2] *= 1.0 + tint_factor * 0.25  # Red (Magenta)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
