"""
AMD FidelityFX Contrast Adaptive Sharpening (CAS) 필터

CAS는 낮은 오버헤드로 적응형 샤프닝을 수행하는 알고리즘입니다.
픽셀마다 샤프닝 양을 조정하여 이미지 전체에 균일한 선명도를 목표로 합니다.
이미 선명한 영역은 덜 샤프닝하고, 디테일이 부족한 영역은 더 샤프닝합니다.
"""

import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import lerp, rcp, rsqrt, saturate


class CASFilter(BaseFilter):
    """AMD Contrast Adaptive Sharpening 필터"""

    def __init__(self):
        super().__init__("CAS", "AMD Contrast Adaptive Sharpening")
        self.contrast = 0.0  # 0.0 ~ 1.0
        self.sharpening = 1.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """CAS 필터 적용"""
        # 파라미터 업데이트
        self.contrast = params.get("contrast", self.contrast)
        self.sharpening = params.get("sharpening", self.sharpening)

        img_float = image.astype(np.float32) / 255.0

        # 3x3 이웃 샘플링 패턴:
        #  a b c
        #  d e f
        #  g h i

        # 패딩 추가 (border replication)
        padded = np.pad(img_float, ((1, 1), (1, 1), (0, 0)), mode="edge")

        h, w = img_float.shape[:2]
        result = np.zeros_like(img_float)

        # 3x3 이웃 추출
        a = padded[0:h, 0:w]  # top-left
        b = padded[0:h, 1 : w + 1]  # top
        c = padded[0:h, 2 : w + 2]  # top-right
        d = padded[1 : h + 1, 0:w]  # left
        e = img_float  # center
        f = padded[1 : h + 1, 2 : w + 2]  # right
        g = padded[2 : h + 2, 0:w]  # bottom-left
        h_pixel = padded[2 : h + 2, 1 : w + 1]  # bottom
        i = padded[2 : h + 2, 2 : w + 2]  # bottom-right

        # Soft min and max (2.0x bigger, factored out the extra multiply)
        # mnRGB = min(min(d,e,f,b), h) + min of corners
        mnRGB = np.minimum(np.minimum(np.minimum(np.minimum(d, e), f), b), h_pixel)
        mnRGB2 = np.minimum(mnRGB, np.minimum(np.minimum(np.minimum(a, c), g), i))
        mnRGB = mnRGB + mnRGB2

        mxRGB = np.maximum(np.maximum(np.maximum(np.maximum(d, e), f), b), h_pixel)
        mxRGB2 = np.maximum(mxRGB, np.maximum(np.maximum(np.maximum(a, c), g), i))
        mxRGB = mxRGB + mxRGB2

        # Smooth minimum distance to signal limit divided by smooth max
        rcpMRGB = rcp(mxRGB)
        ampRGB = saturate(np.minimum(mnRGB, 2.0 - mxRGB) * rcpMRGB)

        # Shaping amount of sharpening
        ampRGB = rsqrt(ampRGB)

        peak = -3.0 * self.contrast + 8.0
        wRGB = -rcp(ampRGB * peak)

        rcpWeightRGB = rcp(4.0 * wRGB + 1.0)

        # Filter shape:
        #   0 w 0
        #   w 1 w
        #   0 w 0
        window = (b + d) + (f + h_pixel)
        outColor = saturate((window * wRGB + e) * rcpWeightRGB)

        # Blend between original and sharpened based on sharpening intensity
        result = lerp(e, outColor, self.sharpening)

        return (result * 255).astype(np.uint8)
