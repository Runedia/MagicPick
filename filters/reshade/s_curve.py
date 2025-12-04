"""
SCurve.fx 구현

S-Curve 대비 조정 필터
Original: FXShaders
"""

import numpy as np

from filters.base_filter import BaseFilter


class SCurveFilter(BaseFilter):
    """
    SCurve - S-커브 대비 조정

    Features:
    - Power curve 기반 대비 조정
    - 어두운 부분은 더 어둡게, 밝은 부분은 더 밝게
    - Low/High 색상 오프셋
    """

    def __init__(self):
        super().__init__("SCurve", "S-커브")

        # Parameters
        self.curve = 1.0  # 1.0 ~ 3.0, Curve strength
        self.offset_low = 0.0  # -1.0 ~ 1.0, Low color offset
        self.offset_high = 0.0  # -1.0 ~ 1.0, High color offset
        self.offset_both = 0.0  # -1.0 ~ 1.0, Both offset

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply SCurve filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - curve: Curve strength (1.0 ~ 3.0, default 1.0)
                - offset_low: Low color offset (-1.0 ~ 1.0, default 0.0)
                - offset_high: High color offset (-1.0 ~ 1.0, default 0.0)
                - offset_both: Both offset (-1.0 ~ 1.0, default 0.0)

        Returns:
            S-curve adjusted image (uint8, 0-255)
        """
        # Update parameters
        self.curve = params.get("curve", self.curve)
        self.offset_low = params.get("offset_low", self.offset_low)
        self.offset_high = params.get("offset_high", self.offset_high)
        self.offset_both = params.get("offset_both", self.offset_both)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0

        # Apply S-curve
        # Low curve: shadows darker (power > 1)
        low = np.power(img_float, self.curve) + self.offset_low

        # High curve: highlights brighter (power < 1)
        high = np.power(img_float, 1.0 / self.curve) + self.offset_high

        # Lerp based on pixel value
        result = np.zeros_like(img_float)
        blend = img_float + self.offset_both
        result[:, :, 0] = low[:, :, 0] + (high[:, :, 0] - low[:, :, 0]) * blend[:, :, 0]
        result[:, :, 1] = low[:, :, 1] + (high[:, :, 1] - low[:, :, 1]) * blend[:, :, 1]
        result[:, :, 2] = low[:, :, 2] + (high[:, :, 2] - low[:, :, 2]) * blend[:, :, 2]

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
