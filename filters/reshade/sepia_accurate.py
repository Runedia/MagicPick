"""
Sepia - 세피아 톤 효과
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class SepiaFilterAccurate(BaseFilter):
    """Sepia - 세피아 톤 효과"""

    def __init__(self):
        super().__init__("Sepia", "세피아")

        self.strength = 0.58
        self.tint = np.array([1.40, 1.10, 0.90], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.strength = params.get("Strength", self.strength)
        tint = params.get("Tint", tuple(self.tint))
        self.tint = np.array(tint, dtype=np.float32)

        img_float = image.astype(np.float32) / 255.0

        luma = np.sum(
            img_float * np.array([0.2126, 0.7152, 0.0722]), axis=2, keepdims=True
        )

        sepia = luma * self.tint

        result = lerp(img_float, sepia, self.strength)

        return (saturate(result) * 255).astype(np.uint8)
