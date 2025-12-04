"""
qUINT_sharp.fx 구현

Depth Enhanced Local Contrast Sharpen (DELCS)
Original: Marty McFly / Pascal Gilcher
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter


@njit(fastmath=True, cache=True)
def color_to_luma(r, g, b):
    """Calculate luma"""
    return r * 0.3 + g * 0.59 + b * 0.11


@njit(fastmath=True, cache=True)
def blend_overlay(a, b):
    """Overlay blending mode"""
    if a < 0.5:
        return 2.0 * a * b
    else:
        c = 1.0 - a
        d = 1.0 - b
        return 1.0 - 2.0 * c * d


@njit(parallel=True, fastmath=True, cache=True)
def quint_sharp_kernel(img, h, w, strength, use_rms, sharpen_mode):
    """
    qUINT Sharp kernel
    
    Args:
        img: Image (H, W, 3)
        h, w: Dimensions
        strength: Sharpen strength
        use_rms: Use local contrast enhancer
        sharpen_mode: 0=Chroma, 1=Luma
    
    Returns:
        Sharpened image
    """
    result = np.empty((h, w, 3), dtype=np.float32)
    
    for y in prange(h):
        for x in range(w):
            # 3x3 neighborhood sampling
            # A B C
            # D E F
            # G H I
            
            # Get center pixel E
            E_r = img[y, x, 0]
            E_g = img[y, x, 1]
            E_b = img[y, x, 2]
            
            # Sample 8 neighbors with boundary clamping
            # A (top-left)
            ay = max(y - 1, 0)
            ax = max(x - 1, 0)
            A_r, A_g, A_b = img[ay, ax, 0], img[ay, ax, 1], img[ay, ax, 2]
            
            # B (top)
            by = max(y - 1, 0)
            B_r, B_g, B_b = img[by, x, 0], img[by, x, 1], img[by, x, 2]
            
            # C (top-right)
            cy = max(y - 1, 0)
            cx = min(x + 1, w - 1)
            C_r, C_g, C_b = img[cy, cx, 0], img[cy, cx, 1], img[cy, cx, 2]
            
            # D (left)
            dx = max(x - 1, 0)
            D_r, D_g, D_b = img[y, dx, 0], img[y, dx, 1], img[y, dx, 2]
            
            # F (right)
            fx = min(x + 1, w - 1)
            F_r, F_g, F_b = img[y, fx, 0], img[y, fx, 1], img[y, fx, 2]
            
            # G (bottom-left)
            gy = min(y + 1, h - 1)
            gx = max(x - 1, 0)
            G_r, G_g, G_b = img[gy, gx, 0], img[gy, gx, 1], img[gy, gx, 2]
            
            # H (bottom)
            hy = min(y + 1, h - 1)
            H_r, H_g, H_b = img[hy, x, 0], img[hy, x, 1], img[hy, x, 2]
            
            # I (bottom-right)
            iy = min(y + 1, h - 1)
            ix = min(x + 1, w - 1)
            I_r, I_g, I_b = img[iy, ix, 0], img[iy, ix, 1], img[iy, ix, 2]
            
            # Compute corners sum
            corners_r = A_r + C_r + G_r + I_r
            corners_g = A_g + C_g + G_g + I_g
            corners_b = A_b + C_b + G_b + I_b
            
            # Compute neighbors sum
            neighbours_r = B_r + D_r + F_r + H_r
            neighbours_g = B_g + D_g + F_g + H_g
            neighbours_b = B_b + D_b + F_b + H_b
            
            # Edge detection: corners + 2*neighbours - 12*center
            edge_r = corners_r + 2.0 * neighbours_r - 12.0 * E_r
            edge_g = corners_g + 2.0 * neighbours_g - 12.0 * E_g
            edge_b = corners_b + 2.0 * neighbours_b - 12.0 * E_b
            
            sharpen_r = edge_r
            sharpen_g = edge_g
            sharpen_b = edge_b
            
            # RMS mask for local contrast detection
            if use_rms:
                # Mean of all 9 pixels
                mean_r = (corners_r + neighbours_r + E_r) / 9.0
                mean_g = (corners_g + neighbours_g + E_g) / 9.0
                mean_b = (corners_b + neighbours_b + E_b) / 9.0
                
                # RMS calculation
                RMS_r = (mean_r - A_r) ** 2
                RMS_g = (mean_g - A_g) ** 2
                RMS_b = (mean_b - A_b) ** 2
                
                RMS_r += (mean_r - B_r) ** 2
                RMS_g += (mean_g - B_g) ** 2
                RMS_b += (mean_b - B_b) ** 2
                
                RMS_r += (mean_r - C_r) ** 2
                RMS_g += (mean_g - C_g) ** 2
                RMS_b += (mean_b - C_b) ** 2
                
                RMS_r += (mean_r - D_r) ** 2
                RMS_g += (mean_g - D_g) ** 2
                RMS_b += (mean_b - D_b) ** 2
                
                RMS_r += (mean_r - E_r) ** 2
                RMS_g += (mean_g - E_g) ** 2
                RMS_b += (mean_b - E_b) ** 2
                
                RMS_r += (mean_r - F_r) ** 2
                RMS_g += (mean_g - F_g) ** 2
                RMS_b += (mean_b - F_b) ** 2
                
                RMS_r += (mean_r - G_r) ** 2
                RMS_g += (mean_g - G_g) ** 2
                RMS_b += (mean_b - G_b) ** 2
                
                RMS_r += (mean_r - H_r) ** 2
                RMS_g += (mean_g - H_g) ** 2
                RMS_b += (mean_b - H_b) ** 2
                
                RMS_r += (mean_r - I_r) ** 2
                RMS_g += (mean_g - I_g) ** 2
                RMS_b += (mean_b - I_b) ** 2
                
                # rsqrt(RMS + 0.001) * 0.1
                sharpen_r *= 1.0 / np.sqrt(RMS_r + 0.001) * 0.1
                sharpen_g *= 1.0 / np.sqrt(RMS_g + 0.001) * 0.1
                sharpen_b *= 1.0 / np.sqrt(RMS_b + 0.001) * 0.1
            
            # Luma mode: convert to grayscale
            if sharpen_mode == 1:
                luma_sharpen = color_to_luma(sharpen_r, sharpen_g, sharpen_b)
                sharpen_r = luma_sharpen
                sharpen_g = luma_sharpen
                sharpen_b = luma_sharpen
            
            # Apply strength and negate
            sharpen_r = -sharpen_r * strength * 0.1
            sharpen_g = -sharpen_g * strength * 0.1
            sharpen_b = -sharpen_b * strength * 0.1
            
            # Smooth falloff: sign * log(abs * 10 + 1) * 0.3
            sharpen_r = (
                (1.0 if sharpen_r >= 0 else -1.0)
                * np.log(abs(sharpen_r) * 10.0 + 1.0)
                * 0.3
            )
            sharpen_g = (
                (1.0 if sharpen_g >= 0 else -1.0)
                * np.log(abs(sharpen_g) * 10.0 + 1.0)
                * 0.3
            )
            sharpen_b = (
                (1.0 if sharpen_b >= 0 else -1.0)
                * np.log(abs(sharpen_b) * 10.0 + 1.0)
                * 0.3
            )
            
            # Overlay blend
            result[y, x, 0] = min(
                max(blend_overlay(E_r, 0.5 + sharpen_r), 0.0), 1.0
            )
            result[y, x, 1] = min(
                max(blend_overlay(E_g, 0.5 + sharpen_g), 0.0), 1.0
            )
            result[y, x, 2] = min(
                max(blend_overlay(E_b, 0.5 + sharpen_b), 0.0), 1.0
            )
    
    return result


class QUINTSharpFilter(BaseFilter):
    """
    qUINT Sharp - Depth Enhanced Local Contrast Sharpen

    Features:
    - 고급 edge detection (3x3 커널)
    - RMS 기반 local contrast detection
    - Chroma/Luma 샤프닝 모드
    - Overlay 블렌딩
    """

    def __init__(self):
        super().__init__("qUINT_Sharp", "qUINT 샤프")

        # Parameters
        self.sharp_strength = 0.7  # 0.0 ~ 1.0
        self.rms_mask_enable = True  # Local contrast enhancer
        self.sharpen_mode = 1  # 0=Chroma, 1=Luma

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply qUINT Sharp filter

        Args:
            image: RGB image (uint8, 0-255)
            **params: Filter parameters
                - sharp_strength: Sharpen strength (0.0 ~ 1.0, default 0.7)
                - rms_mask_enable: Use local contrast enhancer
                    (True/False, default True)
                - sharpen_mode: 0=Chroma, 1=Luma (default 1)

        Returns:
            Sharpened image (uint8, 0-255)
        """
        # Update parameters
        self.sharp_strength = params.get("sharp_strength", self.sharp_strength)
        self.rms_mask_enable = params.get(
            "rms_mask_enable", self.rms_mask_enable
        )
        self.sharpen_mode = params.get("sharpen_mode", self.sharpen_mode)

        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # Apply sharpening
        result = quint_sharp_kernel(
            img_float,
            h,
            w,
            self.sharp_strength,
            self.rms_mask_enable,
            self.sharpen_mode,
        )

        # Convert back
        return (result * 255).astype(np.uint8)
