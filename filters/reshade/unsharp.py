"""
Unsharp.fx 구현
언샤프 마스크 샤프닝

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class UnsharpFilter(BaseFilter):
    def __init__(self):
        super().__init__("Unsharp", "언샤프 마스크")
        self.unsharp_radius = 1.5  # 0.5 ~ 5.0
        self.unsharp_strength = 0.5  # 0.0 ~ 2.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        언샤프 마스크 적용

        Parameters:
        -----------
        unsharp_radius : float
            블러 반경 (0.5 ~ 5.0)
        unsharp_strength : float
            샤프닝 강도 (0.0 ~ 1.0)
        """
        self.unsharp_radius = params.get("unsharp_radius", self.unsharp_radius)
        self.unsharp_strength = params.get("unsharp_strength", self.unsharp_strength)

        img_float = image.astype(np.float32) / 255.0

        # Gaussian Blur 적용
        # Sigma는 radius에 비례
        sigma = self.unsharp_radius * 0.5
        kernel_size = int(self.unsharp_radius * 4) | 1  # 홀수로 만들기
        kernel_size = max(3, kernel_size)

        blurred = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)

        # 언샤프 마스크 = 원본 - 블러
        mask = img_float - blurred

        # 샤프닝 적용
        result = img_float + mask * self.unsharp_strength

        result = saturate(result)

        return (result * 255).astype(np.uint8)
