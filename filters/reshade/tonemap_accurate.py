"""
Tonemap - 노출, 감마, 채도 조정
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, pow_safe, saturate


class TonemapFilterAccurate(BaseFilter):
    """Tonemap - 노출, 감마, 채도 조정"""

    def __init__(self):
        super().__init__("Tonemap", "톤맵")

        self.gamma = 1.0
        self.exposure = 0.0
        self.saturation = 0.0
        self.bleach = 0.0
        self.defog = 0.0
        self.fog_color = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.gamma = params.get("Gamma", self.gamma)
        self.exposure = params.get("Exposure", self.exposure)
        self.saturation = params.get("Saturation", self.saturation)
        self.bleach = params.get("Bleach", self.bleach)
        self.defog = params.get("Defog", self.defog)
        fog_color = params.get("FogColor", tuple(self.fog_color))
        self.fog_color = np.array(fog_color, dtype=np.float32)

        img_float = image.astype(np.float32) / 255.0

        color = saturate(img_float - self.defog * self.fog_color * 2.55)

        color *= np.power(2.0, self.exposure)

        color = pow_safe(color, self.gamma)

        coef_luma = np.array([0.2126, 0.7152, 0.0722])
        lum = np.sum(coef_luma * color, axis=2, keepdims=True)

        L = saturate(10.0 * (lum - 0.45))
        A2 = self.bleach * color

        result1 = 2.0 * color * lum
        result2 = 1.0 - 2.0 * (1.0 - lum) * (1.0 - color)

        new_color = lerp(result1, result2, L)
        mix_rgb = A2 * new_color
        color += (1.0 - A2) * mix_rgb

        middlegray = np.sum(color, axis=2, keepdims=True) * (1.0 / 3.0)
        diffcolor = color - middlegray
        color = (color + diffcolor * self.saturation) / (
            1 + (diffcolor * self.saturation)
        )

        return (saturate(color) * 255).astype(np.uint8)
