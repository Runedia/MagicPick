"""
Sketch.fx 구현
스케치 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class SketchFilter(BaseFilter):
    def __init__(self):
        super().__init__("Sketch", "스케치 효과")
        self.threshold = 0.5  # 0.0 ~ 1.0
        self.intensity = 1.0  # 0.0 ~ 2.0
        self.invert = False  # 반전 (검은 배경에 흰 선)  # 반전 (검은 배경에 흰 선)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        스케치 효과 적용

        Sobel 엣지 검출로 윤곽선 추출

        Parameters:
        -----------
        threshold : float
            엣지 검출 임계값 (0.0 ~ 1.0)
        intensity : float
            스케치 강도 (0.0 ~ 2.0)
        invert : bool
            반전 (True: 검은 배경에 흰 선)
        """
        self.threshold = params.get("threshold", self.threshold)
        self.intensity = params.get("intensity", self.intensity)
        self.invert = params.get("invert", self.invert)

        img_float = image.astype(np.float32) / 255.0

        # 그레이스케일 변환
        gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # Sobel 엣지 검출
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 엣지 강도 계산
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_magnitude = edge_magnitude / 255.0

        # 강도 적용
        edge_magnitude = edge_magnitude * self.intensity

        # 임계값 적용
        edges = np.where(edge_magnitude > self.threshold, 1.0, 0.0)

        # 반전
        if self.invert:
            sketch = edges
        else:
            sketch = 1.0 - edges

        # RGB로 변환
        result = np.stack([sketch, sketch, sketch], axis=2)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
