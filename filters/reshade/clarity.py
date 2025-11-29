"""
Clarity - 로컬 대비 기반 선명도 향상

언샵 마스크를 사용한 디테일 강화
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter


class ClarityFilter(BaseFilter):
    """Clarity 필터 (선명도 향상)"""

    def __init__(self):
        super().__init__("Clarity", "선명도 향상")
        self.strength = 0.4
        self.radius = 2

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.strength = params.get("ClarityStrength", self.strength)
        self.radius = int(params.get("ClarityRadius", self.radius))

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_float = img_bgr.astype(np.float32)

        ksize = int(self.radius * 2 + 1)
        blurred = cv2.GaussianBlur(img_float, (ksize, ksize), 0)

        detail = img_float - blurred
        result = img_float + detail * self.strength

        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
