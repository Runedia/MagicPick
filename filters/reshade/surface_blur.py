"""
SurfaceBlur.fx 구현
표면 블러 (엣지 보존 블러)

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class SurfaceBlurFilter(BaseFilter):
    def __init__(self):
        super().__init__("SurfaceBlur", "표면 블러 (엣지 보존)")
        self.blur_radius = 3.0  # 1.0 ~ 20.0
        self.blur_threshold = 0.05  # 0.001 ~ 0.2  # 0.01 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        표면 블러 적용 (엣지 보존 블러)

        bilateral filter와 유사하게 엣지를 보존하면서 블러 적용

        Parameters:
        -----------
        blur_radius : float
            블러 반경 (1.0 ~ 10.0)
        blur_threshold : float
            엣지 보존 임계값 (0.01 ~ 1.0)
            낮을수록 엣지를 강하게 보존
        """
        self.blur_radius = float(params.get("blur_radius", self.blur_radius))
        self.blur_threshold = params.get("blur_threshold", self.blur_threshold)

        img_float = image.astype(np.float32) / 255.0

        # Bilateral Filter 사용 (엣지 보존 블러)
        # d: 필터 크기
        d = int(self.blur_radius * 2) + 1

        # sigmaColor: 색상 차이에 대한 sigma (threshold와 연관)
        sigma_color = self.blur_threshold * 255.0

        # sigmaSpace: 공간적 거리에 대한 sigma
        sigma_space = self.blur_radius

        # BGR 형식으로 변환하여 처리
        result = cv2.bilateralFilter(
            (img_float * 255).astype(np.uint8),
            d,
            sigma_color,
            sigma_space,
        )

        result = result.astype(np.float32) / 255.0
        result = saturate(result)

        return (result * 255).astype(np.uint8)
