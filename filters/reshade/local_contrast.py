"""
LocalContrastCS - 로컬 대비 향상

언샵 마스크 기반 적응형 대비 강화
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter


class LocalContrastCSFilter(BaseFilter):
    """Local Contrast 필터 (대비 향상)"""

    def __init__(self):
        super().__init__("LocalContrastCS", "로컬 대비")
        self.strength = 1.0
        self.weight_exponent = 5.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        로컬 대비 향상 적용
        언샵 마스크 기반 로컬 대비 강화
        """
        self.strength = params.get("Strength", self.strength)
        self.weight_exponent = params.get("WeightExponent", self.weight_exponent)

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_float = img_bgr.astype(np.float32)

        radius = int(5 * self.strength)
        if radius % 2 == 0:
            radius += 1
        radius = max(3, radius)

        blurred = cv2.GaussianBlur(img_float, (radius, radius), 0)

        detail = img_float - blurred

        detail_magnitude = np.abs(detail)
        weight = np.power(
            detail_magnitude / 255.0, 1.0 / max(0.1, self.weight_exponent)
        )

        enhanced_detail = detail * weight * self.strength

        result = img_float + enhanced_detail
        result = np.clip(result, 0, 255).astype(np.uint8)

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
