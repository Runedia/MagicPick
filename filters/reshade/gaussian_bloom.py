"""
GaussianBloom.fx 구현
가우시안 블룸 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class GaussianBloomFilter(BaseFilter):
    def __init__(self):
        super().__init__("GaussianBloom", "가우시안 블룸")
        self.bloom_threshold = 0.8  # 0.0 ~ 1.0
        self.bloom_radius = 5.0  # 1.0 ~ 50.0
        self.bloom_intensity = 1.0  # 0.0 ~ 5.0
        self.bloom_saturation = 1.0  # 0.0 ~ 3.0  # 0.0 ~ 2.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        가우시안 블룸 적용

        Parameters:
        -----------
        bloom_threshold : float
            밝기 임계값 (0.0 ~ 1.0)
        bloom_intensity : float
            블룸 강도 (0.0 ~ 5.0)
        bloom_radius : float
            블룸 반경 (1.0 ~ 100.0)
        bloom_saturation : float
            블룸 채도 (0.0 ~ 2.0)
        """
        self.bloom_threshold = params.get("bloom_threshold", self.bloom_threshold)
        self.bloom_intensity = params.get("bloom_intensity", self.bloom_intensity)
        self.bloom_radius = params.get("bloom_radius", self.bloom_radius)
        self.bloom_saturation = params.get("bloom_saturation", self.bloom_saturation)

        img_float = image.astype(np.float32) / 255.0

        # 1. 밝은 영역 추출
        luma = np.dot(img_float, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)

        # Soft threshold
        threshold_mask = np.maximum(luma - self.bloom_threshold, 0.0) / (
            1.0 - self.bloom_threshold + 1e-8
        )
        bright_areas = img_float * threshold_mask

        # 2. 채도 조정
        bloom_luma = np.dot(bright_areas, [0.2126, 0.7152, 0.0722])
        bloom_luma = np.expand_dims(bloom_luma, axis=2)

        # 채도 조정: lerp(luma, color, saturation)
        bright_areas = bloom_luma + (bright_areas - bloom_luma) * self.bloom_saturation
        bright_areas = saturate(bright_areas)

        # 3. Gaussian Blur
        sigma = self.bloom_radius / 3.0
        kernel_size = int(self.bloom_radius * 2) | 1
        kernel_size = max(3, min(99, kernel_size))

        bloom = cv2.GaussianBlur(bright_areas, (kernel_size, kernel_size), sigma)

        # 4. 블룸 강도 적용 및 원본과 합성
        bloom = bloom * self.bloom_intensity
        result = img_float + bloom

        result = saturate(result)

        return (result * 255).astype(np.uint8)
