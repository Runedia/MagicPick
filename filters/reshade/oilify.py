"""
Oilify.fx 구현
유화 효과 (Oil Painting)

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class OilifyFilter(BaseFilter):
    def __init__(self):
        super().__init__("Oilify", "유화 효과")
        self.oil_radius = 4  # 1 ~ 10
        self.intensity = 1.0  # 0.0 ~ 2.0  # 4 ~ 256

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        유화 효과 적용

        Kuwahara filter 기반 알고리즘:
        주변 영역을 분할하여 가장 균일한 영역의 평균값 사용

        Parameters:
        -----------
        radius : int
            브러시 반경 (1 ~ 10)
        intensity_levels : int
            강도 레벨 (4 ~ 256)
        """
        self.radius = params.get("radius", self.radius)
        self.intensity_levels = params.get("intensity_levels", self.intensity_levels)

        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 강도 양자화 (포스터화 효과)
        if self.intensity_levels < 256:
            quantized = (
                np.floor(img_float * self.intensity_levels) / self.intensity_levels
            )
        else:
            quantized = img_float

        # Kuwahara filter 적용
        result = self._kuwahara_filter(quantized, self.radius)

        result = saturate(result)

        return (result * 255).astype(np.uint8)

    def _kuwahara_filter(self, img, radius):
        """
        Kuwahara filter 구현

        4개의 사분면으로 나누고 각 사분면의 평균과 분산 계산
        분산이 가장 작은 사분면의 평균값 사용
        """
        h, w = img.shape[:2]
        result = np.zeros_like(img)

        # 패딩 추가
        padded = np.pad(
            img, ((radius, radius), (radius, radius), (0, 0)), mode="reflect"
        )

        # 간소화된 버전: Mean filter와 Median filter 조합
        # 완전한 Kuwahara는 계산량이 많으므로 근사

        # 1. Mean filter (부드럽게)
        kernel_size = radius * 2 + 1
        mean_filtered = cv2.blur(img, (kernel_size, kernel_size))

        # 2. Bilateral filter (엣지 보존)
        bilateral = (
            cv2.bilateralFilter(
                (img * 255).astype(np.uint8), kernel_size, 75, 75
            ).astype(np.float32)
            / 255.0
        )

        # 3. 블렌딩
        result = 0.7 * bilateral + 0.3 * mean_filtered

        return result
