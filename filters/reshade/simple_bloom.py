"""
SimpleBloom.fx 구현
심플 블룸 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class SimpleBloomFilter(BaseFilter):
    def __init__(self):
        super().__init__("SimpleBloom", "심플 블룸")
        self.bloom_threshold = 0.8  # 0.0 ~ 1.0
        self.bloom_radius = 5.0  # 1.0 ~ 50.0
        self.bloom_intensity = 1.0  # 0.0 ~ 5.0  # 1.0 ~ 50.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        심플 블룸 적용

        Parameters:
        -----------
        bloom_threshold : float
            밝기 임계값 (0.0 ~ 1.0)
        bloom_intensity : float
            블룸 강도 (0.0 ~ 5.0)
        bloom_radius : float
            블룸 반경 (1.0 ~ 50.0)
        """
        self.bloom_threshold = params.get("bloom_threshold", self.bloom_threshold)
        self.bloom_intensity = params.get("bloom_intensity", self.bloom_intensity)
        self.bloom_radius = params.get("bloom_radius", self.bloom_radius)

        img_float = image.astype(np.float32) / 255.0

        # 밝은 영역 추출
        luma = np.dot(img_float, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)

        # 임계값보다 밝은 부분만 추출
        bright_mask = (luma > self.bloom_threshold).astype(np.float32)
        bright_areas = img_float * bright_mask

        # 블러 적용
        sigma = self.bloom_radius / 3.0
        kernel_size = int(self.bloom_radius * 2) | 1
        kernel_size = max(3, min(99, kernel_size))

        bloom = cv2.GaussianBlur(bright_areas, (kernel_size, kernel_size), sigma)

        # 블룸 강도 적용
        bloom = bloom * self.bloom_intensity

        # 원본과 블렌딩 (Screen blend)
        result = 1.0 - (1.0 - img_float) * (1.0 - bloom)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
