"""
Vignette - 비네팅 효과
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import pow_safe, saturate


class VignetteFilterAccurate(BaseFilter):
    """Vignette - 비네팅 효과"""

    def __init__(self):
        super().__init__("Vignette", "비네팅")

        self.vignette_type = 0
        self.ratio = 1.0
        self.radius = 2.0
        self.amount = -1.0
        self.slope = 2
        self.center = np.array([0.5, 0.5], dtype=np.float32)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        self.vignette_type = params.get("Type", self.vignette_type)
        self.ratio = params.get("Ratio", self.ratio)
        self.radius = params.get("Radius", self.radius)
        self.amount = params.get("Amount", self.amount)
        self.slope = params.get("Slope", self.slope)
        center = params.get("Center", tuple(self.center))
        self.center = np.array(center, dtype=np.float32)

        img_float = image.astype(np.float32) / 255.0

        h, w = img_float.shape[:2]

        y_coords = np.linspace(0, 1, h).reshape(h, 1)
        x_coords = np.linspace(0, 1, w).reshape(1, w)

        tc = np.stack(
            [np.repeat(x_coords, h, axis=0), np.repeat(y_coords, w, axis=1)], axis=2
        )

        tc -= self.center
        tc[:, :, 0] *= self.ratio

        v = np.sqrt(np.sum(tc * tc, axis=2, keepdims=True))

        if self.vignette_type == 0:
            v = 1.0 - saturate((v - self.radius) * self.slope)
        else:
            v = saturate(((self.radius - v) * self.slope) + 1.0)

        v = pow_safe(v, abs(self.amount))

        result = img_float * v

        return (saturate(result) * 255).astype(np.uint8)
