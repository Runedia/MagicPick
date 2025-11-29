"""
Cartoon.fx 구현
카툰 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class CartoonFilter(BaseFilter):
    def __init__(self):
        super().__init__("Cartoon", "카툰 효과")
        self.color_levels = 8.0  # 2.0 ~ 16.0
        self.edge_threshold = 0.2  # 0.0 ~ 1.0  # 2 ~ 16

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        카툰 효과 적용

        1. 색상 포스터화 (양자화)
        2. 엣지 검출 및 강조

        Parameters:
        -----------
        edge_threshold : float
            엣지 검출 임계값 (0.0 ~ 1.0)
        color_levels : int
            색상 레벨 수 (2 ~ 16)
        """
        self.edge_threshold = params.get("edge_threshold", self.edge_threshold)
        self.color_levels = params.get("color_levels", self.color_levels)

        img_float = image.astype(np.float32) / 255.0

        # 1. 색상 포스터화 (양자화)
        posterized = np.floor(img_float * self.color_levels) / self.color_levels

        # 2. 엣지 검출
        # 그레이스케일 변환
        gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Sobel 엣지 검출
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 엣지 강도 계산
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_magnitude = edge_magnitude / 255.0

        # 엣지 임계값 적용
        edge_mask = (edge_magnitude > self.edge_threshold).astype(np.float32)
        edge_mask = np.expand_dims(edge_mask, axis=2)

        # 3. 엣지와 포스터화된 이미지 합성
        # 엣지 부분을 검은색으로
        result = posterized * (1.0 - edge_mask)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
