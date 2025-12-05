"""
lilium__filmgrain.fx 간소화 구현

Gaussian Film Grain
Original: haasn, ported by Lilium
"""

import numpy as np

from filters.base_filter import BaseFilter

# Approximation constants for inverse normal CDF
A0 = 0.151015505647689
A1 = -0.5303572634357367
A2 = 1.365020122861334
B0 = 0.132089632343748
B1 = -0.7607324991323768


class LiliumFilmGrainFilter(BaseFilter):
    """
    LiliumFilmGrain - Lilium 필름 그레인

    Features:
    - Gaussian film grain
    - Luma 채널에만 노이즈 적용
    - 고품질 통계적 노이즈
    """

    def __init__(self):
        super().__init__("LiliumFilmGrain", "Lilium 필름 그레인")

        # Parameters
        self.intensity = 0.025  # 0.0 ~ 0.1

    def _permute(self, x):
        """Permutation function for pseudo-random generation"""
        x = (34.0 * x + 1.0) * x
        return np.fmod(x / 289.0, 1.0) * 289.0

    def _rand(self, state):
        """Generate random number from state"""
        state = self._permute(state)
        return np.fmod(state / 41.0, 1.0)

    def _rgb_to_ycbcr_bt709(self, rgb):
        """Convert RGB to YCbCr (BT.709)"""
        y = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
        cb = (rgb[:, :, 2] - y) / 1.8556
        cr = (rgb[:, :, 0] - y) / 1.5748
        return np.stack([y, cb, cr], axis=2)

    def _ycbcr_to_rgb_bt709(self, ycbcr):
        """Convert YCbCr to RGB (BT.709)"""
        y = ycbcr[:, :, 0]
        cb = ycbcr[:, :, 1]
        cr = ycbcr[:, :, 2]

        r = y + 1.5748 * cr
        g = y - 0.1873 * cb - 0.4681 * cr
        b = y + 1.8556 * cb

        return np.stack([r, g, b], axis=2)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply LiliumFilmGrain filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - intensity: Grain intensity (0.0 ~ 0.1, default 0.025)

        Returns:
            Film grained image (uint8, 0-255)
        """
        # Update parameters
        self.intensity = params.get("intensity", self.intensity)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Convert to YCbCr
        ycbcr = self._rgb_to_ycbcr_bt709(img_float)

        # Generate grain per pixel
        # Using simplified random generation
        np.random.seed(None)

        # Generate pseudo-random grain using approximation of inverse normal CDF
        p = 0.95 * np.random.rand(h, w) + 0.025
        q = p - 0.5
        r = q * q

        # Rational approximation
        grain = q * (A2 + (A1 * r + A0) / (r * r + B1 * r + B0))
        grain *= 0.255121822830526  # Normalize to (-1, 1)

        # Apply grain to Y channel only
        ycbcr[:, :, 0] += self.intensity * grain

        # Convert back to RGB
        result = self._ycbcr_to_rgb_bt709(ycbcr)

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
