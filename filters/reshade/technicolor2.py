"""
Technicolor2.fx 구현
테크니컬러 v2 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import get_luma_bt709, lerp, saturate


@njit(parallel=True, fastmath=True, cache=True)
def _technicolor2_kernel(image, strength, saturation, brightness):
    h, w, c = image.shape
    out = np.empty((h, w, c), dtype=np.float32)

    for y in prange(h):
        for x in range(w):
            # BGR Input
            b_in = image[y, x, 0]
            g_in = image[y, x, 1]
            r_in = image[y, x, 2]

            # 1. Technicolor 3-strip
            # Red channel (Cyan filter): (G + B) * 0.5
            cyan = (g_in + b_in) * 0.5

            # Green channel (Magenta filter): (R + B) * 0.5
            magenta = (r_in + b_in) * 0.5

            # Blue channel (Yellow filter): (R + G) * 0.5
            yellow = (r_in + g_in) * 0.5

            # Combine
            r_out = cyan + magenta
            g_out = cyan + yellow
            b_out = magenta + yellow

            # Saturate
            r_out = saturate(r_out)
            g_out = saturate(g_out)
            b_out = saturate(b_out)

            # 2. Saturation Adjustment
            luma = get_luma_bt709(r_out, g_out, b_out)

            r_out = lerp(luma, r_out, saturation)
            g_out = lerp(luma, g_out, saturation)
            b_out = lerp(luma, b_out, saturation)

            # 3. Brightness Adjustment
            r_out += brightness
            g_out += brightness
            b_out += brightness

            r_out = saturate(r_out)
            g_out = saturate(g_out)
            b_out = saturate(b_out)

            # 4. Blend with Original
            r_final = lerp(r_in, r_out, strength)
            g_final = lerp(g_in, g_out, strength)
            b_final = lerp(b_in, b_out, strength)

            # Output (BGR)
            out[y, x, 0] = saturate(b_final)
            out[y, x, 1] = saturate(g_final)
            out[y, x, 2] = saturate(r_final)

    return out


class Technicolor2Filter(BaseFilter):
    """
    Technicolor2 (Numba Accelerated)
    """

    def __init__(self):
        super().__init__("Technicolor2", "테크니컬러 3-strip")
        self.strength = 1.0  # 0.0 ~ 1.0
        self.saturation = 1.0  # 0.0 ~ 2.0
        self.brightness = 0.0  # -0.5 ~ 0.5

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        테크니컬러 v2 효과 적용
        """
        self.strength = params.get("strength", self.strength)
        self.saturation = params.get("saturation", self.saturation)
        self.brightness = params.get("brightness", self.brightness)

        img_float = image.astype(np.float32) / 255.0

        result = _technicolor2_kernel(
            img_float, self.strength, self.saturation, self.brightness
        )

        return (result * 255).astype(np.uint8)
