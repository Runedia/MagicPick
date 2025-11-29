"""
Technicolor2.fx 구현
테크니컬러 v2 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import dot3, lerp, saturate


class Technicolor2Filter(BaseFilter):
    def __init__(self):
        super().__init__("Technicolor2", "테크니컬러 3-strip")
        self.strength = 1.0  # 0.0 ~ 1.0
        self.saturation = 1.0  # 0.0 ~ 2.0
        self.brightness = 0.0  # -0.5 ~ 0.5  # -1.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        테크니컬러 v2 효과 적용

        개선된 3-strip 프로세스 시뮬레이션

        Parameters:
        -----------
        strength : float
            효과 강도 (0.0 ~ 1.0)
        saturation : float
            채도 조정 (0.0 ~ 2.0)
        brightness : float
            밝기 조정 (-1.0 ~ 1.0)
        """
        self.strength = params.get("strength", self.strength)
        self.saturation = params.get("saturation", self.saturation)
        self.brightness = params.get("brightness", self.brightness)

        img_float = image.astype(np.float32) / 255.0

        # BGR → RGB 변환
        rgb = img_float[:, :, ::-1].copy()

        # Technicolor 3-strip 매트릭스
        # Red channel (Cyan filter)
        cyan = dot3(rgb, np.array([0.0, 0.5, 0.5], dtype=np.float32))

        # Green channel (Magenta filter)
        magenta = dot3(rgb, np.array([0.5, 0.0, 0.5], dtype=np.float32))

        # Blue channel (Yellow filter)
        yellow = dot3(rgb, np.array([0.5, 0.5, 0.0], dtype=np.float32))

        # 3-strip 결합
        result_rgb = np.stack(
            [
                cyan + magenta,  # Red
                cyan + yellow,  # Green
                magenta + yellow,  # Blue
            ],
            axis=2,
        )

        result_rgb = saturate(result_rgb)

        # 채도 조정
        luma = np.dot(result_rgb, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)
        result_rgb = lerp(luma, result_rgb, self.saturation)

        # 밝기 조정
        result_rgb = result_rgb + self.brightness

        result_rgb = saturate(result_rgb)

        # RGB → BGR 변환
        result = result_rgb[:, :, ::-1]

        # 원본과 블렌딩
        result = lerp(img_float, result, self.strength)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
