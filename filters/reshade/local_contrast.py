"""
LocalContrastCS - 로컬 대비 향상

언샵 마스크 기반 적응형 대비 강화
"""

import cv2
import numpy as np
from numba import njit, prange

from filters.base_filter import BaseFilter
from filters.reshade.numba_helpers import pow_safe, saturate


@njit(parallel=True, fastmath=True, cache=True)
def _local_contrast_enhance(image, blurred, strength, weight_exponent):
    h, w, c = image.shape
    out = np.empty((h, w, c), dtype=np.float32)

    # Pre-calculate exponent factor
    exp_factor = 1.0 / max(0.1, weight_exponent)

    for y in prange(h):
        for x in range(w):
            for k in range(c):
                val = image[y, x, k]
                blur_val = blurred[y, x, k]

                detail = val - blur_val
                detail_mag = abs(detail)

                # weight = pow(detail_mag, exp_factor)
                weight = pow_safe(detail_mag, exp_factor)

                enhanced = detail * weight * strength

                out[y, x, k] = saturate(val + enhanced)

    return out


class LocalContrastCSFilter(BaseFilter):
    """Local Contrast 필터 (대비 향상, Numba Accelerated)"""

    def __init__(self):
        super().__init__("LocalContrastCS", "로컬 대비")
        self.strength = 1.0
        self.weight_exponent = 5.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        로컬 대비 향상 적용
        언샵 마스크 기반 로컬 대비 강화
        """
        self.strength = params.get("Strength", self.strength)
        self.weight_exponent = params.get("WeightExponent", self.weight_exponent)

        # OpenCV GaussianBlur는 이미 최적화되어 있으므로 그대로 사용
        # 단, Numba 커널로 넘기기 위해 float 변환
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_float = img_bgr.astype(np.float32) / 255.0

        radius = int(5 * self.strength)
        if radius % 2 == 0:
            radius += 1
        radius = max(3, radius)

        # 블러링은 OpenCV가 빠름
        blurred = cv2.GaussianBlur(img_float, (radius, radius), 0)

        # 디테일 강화 및 합성은 Numba로 처리
        result = _local_contrast_enhance(
            img_float, blurred, self.strength, self.weight_exponent
        )

        result_uint8 = (result * 255).astype(np.uint8)
        return cv2.cvtColor(result_uint8, cv2.COLOR_BGR2RGB)
