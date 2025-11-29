"""
PD80_04_Color_Balance.fx 구현
PD80 색상 균형 조정

원본 셰이더: https://github.com/prod80/prod80-ReShade-Repository
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80ColorBalanceFilter(BaseFilter):
    def __init__(self):
        super().__init__("PD80ColorBalance", "색상 균형 (그림자/중간톤/하이라이트)")
        self.shadow_red = 0.0  # -1.0 ~ 1.0
        self.shadow_green = 0.0  # -1.0 ~ 1.0
        self.shadow_blue = 0.0  # -1.0 ~ 1.0
        self.midtone_red = 0.0  # -1.0 ~ 1.0
        self.midtone_green = 0.0  # -1.0 ~ 1.0
        self.midtone_blue = 0.0  # -1.0 ~ 1.0
        self.highlight_red = 0.0  # -1.0 ~ 1.0
        self.highlight_green = 0.0  # -1.0 ~ 1.0
        self.highlight_blue = 0.0  # -1.0 ~ 1.0  # -1.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        색상 균형 적용

        그림자, 중간톤, 하이라이트 각각에 대해 RGB 색상 조정

        Parameters:
        -----------
        shadows_cyan_red : float
            그림자 Cyan(-) / Red(+) 조정
        shadows_magenta_green : float
            그림자 Magenta(-) / Green(+) 조정
        shadows_yellow_blue : float
            그림자 Yellow(-) / Blue(+) 조정
        (midtones, highlights 동일)
        """
        for key in params:
            if hasattr(self, key):
                setattr(self, key, params[key])

        img_float = image.astype(np.float32) / 255.0

        # Luminance 계산
        luma = np.dot(img_float, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)

        # 그림자/중간톤/하이라이트 가중치 계산
        # Shadows: 어두운 영역에 강하게 적용
        shadow_weight = 1.0 - luma
        shadow_weight = pow_safe(shadow_weight, 2.0)

        # Highlights: 밝은 영역에 강하게 적용
        highlight_weight = luma
        highlight_weight = pow_safe(highlight_weight, 2.0)

        # Midtones: 중간 영역에 강하게 적용 (bell curve)
        midtone_weight = 4.0 * luma * (1.0 - luma)

        # 색상 조정 벡터 (BGR 순서)
        shadows_adj = np.array(
            [
                self.shadows_yellow_blue,  # Blue
                self.shadows_magenta_green,  # Green
                self.shadows_cyan_red,  # Red
            ],
            dtype=np.float32,
        )

        midtones_adj = np.array(
            [
                self.midtones_yellow_blue,
                self.midtones_magenta_green,
                self.midtones_cyan_red,
            ],
            dtype=np.float32,
        )

        highlights_adj = np.array(
            [
                self.highlights_yellow_blue,
                self.highlights_magenta_green,
                self.highlights_cyan_red,
            ],
            dtype=np.float32,
        )

        # 각 톤 영역에 색상 조정 적용
        result = img_float.copy()
        result += shadow_weight * shadows_adj * 0.5
        result += midtone_weight * midtones_adj * 0.5
        result += highlight_weight * highlights_adj * 0.5

        result = saturate(result)

        return (result * 255).astype(np.uint8)


def pow_safe(x, power):
    """안전한 거듭제곱 (음수 방지)"""
    return np.power(np.maximum(x, 0.0), power)
