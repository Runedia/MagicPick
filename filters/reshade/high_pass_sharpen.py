"""
HighPassSharpen.fx 구현
하이패스 샤프닝

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class HighPassSharpenFilter(BaseFilter):
    def __init__(self):
        super().__init__("HighPassSharpen", "하이패스 샤프닝")
        self.sharp_radius = 2.0  # 0.5 ~ 8.0
        self.sharp_strength = 0.65  # 0.0 ~ 2.0
        self.sharp_clamp = 0.035  # 0.005 ~ 1.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        하이패스 샤프닝 적용

        Parameters:
        -----------
        sharp_radius : float
            샤프닝 반경 (0.5 ~ 5.0)
        sharp_strength : float
            샤프닝 강도 (0.0 ~ 2.0)
        sharp_clamp : float
            샤프닝 클램핑 (0.0 ~ 1.0)
        """
        self.sharp_radius = params.get("sharp_radius", self.sharp_radius)
        self.sharp_strength = params.get("sharp_strength", self.sharp_strength)
        self.sharp_clamp = params.get("sharp_clamp", self.sharp_clamp)

        img_float = image.astype(np.float32) / 255.0

        # Gaussian Blur 적용
        sigma = self.sharp_radius * 0.5
        kernel_size = int(self.sharp_radius * 4) | 1
        kernel_size = max(3, kernel_size)

        blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)

        # High-pass filter = 원본 - 블러
        high_pass = img_float - blurred

        # Clamp 적용 (과도한 샤프닝 방지)
        high_pass = np.clip(high_pass, -self.sharp_clamp, self.sharp_clamp)

        # 샤프닝 적용
        result = img_float + high_pass * self.sharp_strength

        result = saturate(result)

        return (result * 255).astype(np.uint8)
