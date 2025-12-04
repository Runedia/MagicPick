"""
OrtonBloom.fx 구현

Orton Effect (soft glow) Bloom
Original: moriz1
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from filters.base_filter import BaseFilter


class OrtonBloomFilter(BaseFilter):
    """
    OrtonBloom - Orton 효과 블룸

    Features:
    - Gaussian blur 기반 soft glow
    - 블랙/화이트 포인트 조정
    - 중간톤 시프트
    - Screen 블렌딩
    """

    def __init__(self):
        super().__init__("OrtonBloom", "Orton 블룸")

        # Parameters
        self.blur_multi = 1.0  # 0.0 ~ 1.0
        self.black_point = 60  # 0 ~ 255
        self.white_point = 150  # 0 ~ 255
        self.mid_tones_shift = -0.84  # -1.0 ~ 1.0
        self.blend_strength = 0.07  # 0.0 ~ 1.0
        self.gamma_correction_enable = True

    def _calc_luma(self, color):
        """Calculate luma with optional gamma correction"""
        if self.gamma_correction_enable:
            # Weighted average with gamma correction
            luma = (color[:, :, 0] * 2 + color[:, :, 2] + color[:, :, 1] * 3) / 6
            return np.power(np.abs(luma), 1.0 / 2.2)
        else:
            return (color[:, :, 0] * 2 + color[:, :, 2] + color[:, :, 1] * 3) / 6

    def _gaussian_blur_adaptive(self, img):
        """
        Gaussian blur with luma-adaptive blur power
        
        Uses 18-tap Gaussian blur with specific weights
        """
        h, w = img.shape[:2]
        
        # Calculate blur power per pixel
        luma = self._calc_luma(img)
        
        # Since we can't do per-pixel variable blur easily,
        # we'll use a simpler approach: single gaussian blur
        # with strength controlled by average luma
        avg_luma = np.mean(luma)
        blur_sigma = avg_luma * self.blur_multi * 5.0  # Scale factor
        
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = gaussian_filter(
                img[:, :, c], sigma=blur_sigma, mode='reflect'
            )
        
        return result

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply OrtonBloom filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - blur_multi: Blur multiplier (0.0 ~ 1.0, default 1.0)
                - black_point: Black point (0 ~ 255, default 60)
                - white_point: White point (0 ~ 255, default 150)
                - mid_tones_shift: Midtones shift
                    (-1.0 ~ 1.0, default -0.84)
                - blend_strength: Blend strength (0.0 ~ 1.0, default 0.07)
                - gamma_correction_enable: Enable gamma correction
                    (True/False, default True)

        Returns:
            Orton bloomed image (uint8, 0-255)
        """
        # Update parameters
        self.blur_multi = params.get("blur_multi", self.blur_multi)
        self.black_point = params.get("black_point", self.black_point)
        self.white_point = params.get("white_point", self.white_point)
        self.mid_tones_shift = params.get("mid_tones_shift", self.mid_tones_shift)
        self.blend_strength = params.get("blend_strength", self.blend_strength)
        self.gamma_correction_enable = params.get(
            "gamma_correction_enable", self.gamma_correction_enable
        )

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        original = img_float.copy()

        # 1. Two-pass Gaussian blur (horizontal then vertical)
        # For simplicity, use scipy gaussian_filter
        blurred = self._gaussian_blur_adaptive(img_float)

        # 2. Apply levels adjustment to blurred image
        black_point_float = self.black_point / 255.0
        
        # Avoid division by zero
        if self.white_point == self.black_point:
            white_point_float = 255.0 / 0.00025
        else:
            white_point_float = 255.0 / (self.white_point - self.black_point)
        
        # Midpoint
        mid_point_float = (
            (white_point_float + black_point_float) / 2.0 + self.mid_tones_shift
        )
        mid_point_float = np.clip(
            mid_point_float, black_point_float, white_point_float
        )

        # Apply levels to blurred image
        adjusted = (
            blurred * white_point_float
            - (black_point_float * white_point_float)
        ) * mid_point_float

        # 3. Blend with original using Screen blend mode
        # Screen blend: 1 - (1 - a) * (1 - b)
        # Which is equivalent to: max(original, lerp(original, screen, strength))
        
        # Saturate adjusted
        adjusted = np.clip(adjusted, 0, 1)
        
        # Screen blend
        screen = 1.0 - (1.0 - adjusted) * (1.0 - adjusted)
        
        # Blend with original
        result = np.maximum(
            original, original + (screen - original) * self.blend_strength
        )

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
