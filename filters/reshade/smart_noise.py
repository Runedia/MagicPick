"""
SmartNoise.fx 구현

지능형 노이즈 - 밝기 적응형 노이즈
Original: Bapho - https://github.com/Bapho
"""

import numpy as np
from numba import njit

from filters.base_filter import BaseFilter

# Constants
PHI = 1.61803398874989 * 0.1  # Golden Ratio
PI = 3.14159265359 * 0.1  # PI
SQ2 = 1.41421356237 * 10000.0  # Square Root of Two


@njit(fastmath=True, cache=True)
def gold_noise(coord_x, coord_y, seed):
    """
    Golden noise function - unique noise based on coordinates and seed

    Args:
        coord_x: X coordinate (scaled)
        coord_y: Y coordinate (scaled)
        seed: Seed value

    Returns:
        Noise value (0.0 ~ 1.0)
    """
    # Distance calculation
    dx = coord_x * (seed + PHI) - PHI
    dy = coord_y * (seed + PHI) - PI
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(dist_sq)

    # Generate noise
    tan_val = np.tan(dist)
    noise = np.abs(tan_val * SQ2 - np.floor(tan_val * SQ2))

    return noise


class SmartNoiseFilter(BaseFilter):
    """
    SmartNoise - 지능형 노이즈

    Features:
    - Golden noise 기반 고품질 노이즈
    - 밝기 적응형 노이즈 강도
    - 중간 톤에서 노이즈 최대화
    - 붉은색 픽셀에 노이즈 감소
    - 노이즈 클리핑 방지
    """

    def __init__(self):
        super().__init__("SmartNoise", "스마트 노이즈")

        # Parameters
        self.noise_amount = 1.0  # 0.0 ~ 4.0, Noise amount

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply SmartNoise filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - noise_amount: Noise amount (0.0 ~ 4.0, default 1.0)

        Returns:
            Smart noised image (uint8, 0-255)
        """
        # Update parameters
        self.noise_amount = params.get("noise_amount", self.noise_amount)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Base amount
        base_amount = self.noise_amount * 0.08

        # Apply smart noise pixel by pixel
        result = np.zeros_like(img_float)

        for y in range(h):
            for x in range(w):
                r = img_float[y, x, 0]
                g = img_float[y, x, 1]
                b = img_float[y, x, 2]

                # Calculate luminance
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

                # Adaptive noise amount based on luminance
                # Maximum noise at mid-tones (luminance = 0.5)
                amount = base_amount
                if luminance < 0.5:
                    amount *= luminance / 0.5
                else:
                    amount *= (1.0 - luminance) / 0.5

                # Reduce noise on reddish pixels
                red_diff = r - ((g + b) / 2.0)
                if red_diff > 0.0:
                    amount *= 1.0 - (red_diff * 0.5)

                # Average noise luminance to subtract
                sub = 0.5 * amount

                # Noise clipping prevention
                if luminance - sub < 0.0:
                    factor = luminance / max(sub, 1e-6)
                    amount *= factor
                    sub *= factor
                elif luminance + sub > 1.0:
                    if luminance > sub:
                        factor = sub / max(luminance, 1e-6)
                    else:
                        factor = luminance / max(sub, 1e-6)
                    amount *= factor
                    sub *= factor

                # Generate unique seed per pixel
                # Using position and luminance for stable noise
                tx = float(x) / w
                ty = float(y) / h
                coord_scale = h * 2.0
                seed_val = (
                    (luminance * h) + (w * ty) + tx
                    # Depth would go here: + depth * h
                ) * 0.0001

                # Calculate golden noise
                noise = gold_noise(tx * coord_scale, ty * coord_scale, seed_val)

                # Apply noise
                result[y, x, 0] = r + (noise * amount - sub)
                result[y, x, 1] = g + (noise * amount - sub)
                result[y, x, 2] = b + (noise * amount - sub)

        # Clamp and convert
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)
