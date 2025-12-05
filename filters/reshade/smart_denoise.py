"""
SmartDeNoise.fx 구현

Edge-preserving Denoise Filter using circular kernel
Original: https://github.com/BrutPitt/glslSmartDeNoise/
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter


@njit(parallel=True, fastmath=True, cache=True)
def _smart_denoise_kernel(
    img,
    h,
    w,
    radius,
    inv_sigma_qx2,
    inv_sigma_qx2_pi,
    inv_threshold_sqx2,
    inv_threshold_sqrt2_pi,
):
    """
    SmartDeNoise JIT 커널

    Args:
        img: float32 이미지 (0-1)
        h, w: 이미지 크기
        radius: 커널 반경
        inv_sigma_qx2: 1.0 / (2 * sigma^2)
        inv_sigma_qx2_pi: 1.0 / (2 * PI * sigma^2)
        inv_threshold_sqx2: 1.0 / (2 * threshold^2)
        inv_threshold_sqrt2_pi: 1.0 / (sqrt(2*PI) * threshold)

    Returns:
        디노이즈된 이미지
    """
    out = np.empty((h, w, 3), dtype=np.float32)
    rad_q = radius * radius

    for y in prange(h):
        for x in range(w):
            # Center pixel
            centr_r = img[y, x, 0]
            centr_g = img[y, x, 1]
            centr_b = img[y, x, 2]

            z_buff = 0.0
            a_buff_r = 0.0
            a_buff_g = 0.0
            a_buff_b = 0.0

            # Circular kernel
            for dx in range(-radius, radius + 1):
                # pt = yRadius: circular trend
                pt_sq = rad_q - dx * dx
                if pt_sq < 0:
                    continue
                pt = int(np.sqrt(pt_sq))

                for dy in range(-pt, pt + 1):
                    # Clamp coordinates
                    ny = min(max(y + dy, 0), h - 1)
                    nx = min(max(x + dx, 0), w - 1)

                    # Sample neighbor
                    walk_r = img[ny, nx, 0]
                    walk_g = img[ny, nx, 1]
                    walk_b = img[ny, nx, 2]

                    # Color difference
                    dc_r = walk_r - centr_r
                    dc_g = walk_g - centr_g
                    dc_b = walk_b - centr_b
                    dc_sq = dc_r * dc_r + dc_g * dc_g + dc_b * dc_b

                    # Spatial distance
                    d_sq = dx * dx + dy * dy

                    # Delta factor (Gaussian weights)
                    # exp(-dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI *
                    # exp(-dot(d, d) * invSigmaQx2) * invSigmaQx2PI
                    delta_factor = (
                        np.exp(-dc_sq * inv_threshold_sqx2)
                        * inv_threshold_sqrt2_pi
                        * np.exp(-d_sq * inv_sigma_qx2)
                        * inv_sigma_qx2_pi
                    )

                    z_buff += delta_factor
                    a_buff_r += delta_factor * walk_r
                    a_buff_g += delta_factor * walk_g
                    a_buff_b += delta_factor * walk_b

            # Normalize
            if z_buff > 0:
                out[y, x, 0] = a_buff_r / z_buff
                out[y, x, 1] = a_buff_g / z_buff
                out[y, x, 2] = a_buff_b / z_buff
            else:
                out[y, x, 0] = centr_r
                out[y, x, 1] = centr_g
                out[y, x, 2] = centr_b

    return out


class SmartDeNoiseFilter(BaseFilter):
    """
    SmartDeNoise 필터

    Edge-preserving denoise filter using circular Gaussian kernel.
    원본: https://github.com/BrutPitt/glslSmartDeNoise/
    """

    def __init__(self):
        super().__init__("SmartDeNoise", "스마트 디노이즈")

        # Default parameters from original shader
        self.sigma = 1.25  # Standard Deviation Sigma Radius (0.001 ~ 8.0)
        self.threshold = 0.05  # Edge Sharpening Threshold (0.001 ~ 0.25)
        self.k_sigma = 1.5  # K Factor Sigma Coefficient (0.0 ~ 3.0)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply SmartDeNoise filter

        Args:
            image: RGB 이미지 (uint8, 0-255)
            **params:
                sigma (float): Standard deviation (0.001 ~ 8.0, default 1.25)
                threshold (float): Edge threshold (0.001 ~ 0.25, default 0.05)
                k_sigma (float): Kernel coefficient (0.0 ~ 3.0, default 1.5)

        Returns:
            디노이즈된 이미지 (uint8, 0-255)
        """
        # Update parameters
        self.sigma = params.get("sigma", self.sigma)
        self.threshold = params.get("threshold", self.threshold)
        self.k_sigma = params.get("k_sigma", self.k_sigma)

        # Convert to float32 [0, 1]
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Calculate kernel radius
        radius = int(round(self.k_sigma * self.sigma))
        if radius < 1:
            radius = 1

        # Pre-calculate constants
        INV_PI = 0.31830988618379067
        INV_SQRT_OF_2PI = 0.39894228040143268

        inv_sigma_qx2 = 0.5 / (self.sigma * self.sigma)  # 1/(2*sigma^2)
        inv_sigma_qx2_pi = INV_PI * inv_sigma_qx2  # 1/(2*PI*sigma^2)

        # 1/(2*threshold^2)
        inv_threshold_sqx2 = 0.5 / (self.threshold * self.threshold)
        # 1/(sqrt(2*PI)*threshold)
        inv_threshold_sqrt2_pi = INV_SQRT_OF_2PI / self.threshold

        # Apply denoise kernel
        result = _smart_denoise_kernel(
            img_float,
            h,
            w,
            radius,
            inv_sigma_qx2,
            inv_sigma_qx2_pi,
            inv_threshold_sqx2,
            inv_threshold_sqrt2_pi,
        )

        # Clamp and convert back to uint8
        result = np.clip(result, 0.0, 1.0)
        return (result * 255).astype(np.uint8)

    def warmup(self):
        """JIT 컴파일 유도"""
        print(f"[{self.name}] Warm-up started...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        self.apply(dummy)
        print(f"[{self.name}] Warm-up completed.")
