"""
HueFX.fx 구현
색조 조정 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class HueFXFilter(BaseFilter):
    def __init__(self):
        super().__init__("HueFX", "색조 조정")
        self.hue_shift = 0.0  # -180.0 ~ 180.0 (degrees)  # -180.0 ~ 180.0 (degrees)

    def rgb_to_hsv(self, rgb):
        """RGB를 HSV로 변환"""
        r, g, b = rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c

        # Hue 계산
        h = np.zeros_like(max_c)

        # Red is max
        mask = (max_c == r) & (delta > 0)
        h[mask] = 60.0 * (((g[mask] - b[mask]) / delta[mask]) % 6.0)

        # Green is max
        mask = (max_c == g) & (delta > 0)
        h[mask] = 60.0 * (((b[mask] - r[mask]) / delta[mask]) + 2.0)

        # Blue is max
        mask = (max_c == b) & (delta > 0)
        h[mask] = 60.0 * (((r[mask] - g[mask]) / delta[mask]) + 4.0)

        # Saturation
        s = np.where(max_c > 0, delta / max_c, 0.0)

        # Value
        v = max_c

        return np.stack([h, s, v], axis=2)

    def hsv_to_rgb(self, hsv):
        """HSV를 RGB로 변환"""
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        c = v * s
        x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
        m = v - c

        h_i = (h / 60.0).astype(np.int32) % 6

        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)

        mask = h_i == 0
        r[mask], g[mask], b[mask] = c[mask], x[mask], 0.0

        mask = h_i == 1
        r[mask], g[mask], b[mask] = x[mask], c[mask], 0.0

        mask = h_i == 2
        r[mask], g[mask], b[mask] = 0.0, c[mask], x[mask]

        mask = h_i == 3
        r[mask], g[mask], b[mask] = 0.0, x[mask], c[mask]

        mask = h_i == 4
        r[mask], g[mask], b[mask] = x[mask], 0.0, c[mask]

        mask = h_i == 5
        r[mask], g[mask], b[mask] = c[mask], 0.0, x[mask]

        r += m
        g += m
        b += m

        return np.stack([b, g, r], axis=2)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        색조 조정 적용

        Parameters:
        -----------
        hue_shift : float
            색조 시프트 각도 (-180.0 ~ 180.0)
        """
        self.hue_shift = params.get("hue_shift", self.hue_shift)

        img_float = image.astype(np.float32) / 255.0

        # RGB를 HSV로 변환
        hsv = self.rgb_to_hsv(img_float)

        # Hue 시프트 적용
        hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift) % 360.0

        # HSV를 RGB로 변환
        result = self.hsv_to_rgb(hsv)

        result = saturate(result)

        return (result * 255).astype(np.uint8)
