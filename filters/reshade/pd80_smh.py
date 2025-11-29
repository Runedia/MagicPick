"""
PD80_03_Shadows_Midtones_Highlights.fx 구현
PD80 그림자/중간톤/하이라이트 조정

원본 셰이더: https://github.com/prod80/prod80-ReShade-Repository
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import pow_safe, saturate


class PD80SMHFilter(BaseFilter):
    def __init__(self):
        super().__init__("PD80SMH", "그림자/중간톤/하이라이트 RGB 조정")
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
        그림자/중간톤/하이라이트 조정 적용

        Parameters:
        -----------
        shadows_red/green/blue : float
            그림자 RGB 조정 (-1.0 ~ 1.0)
        midtones_red/green/blue : float
            중간톤 RGB 조정 (-1.0 ~ 1.0)
        highlights_red/green/blue : float
            하이라이트 RGB 조정 (-1.0 ~ 1.0)
        """
        for key in params:
            if hasattr(self, key):
                setattr(self, key, params[key])

        img_float = image.astype(np.float32) / 255.0

        # Luminance 계산
        luma = np.dot(img_float, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)

        # 그림자/중간톤/하이라이트 가중치 계산
        # Shadows: 어두운 영역 (0.0 ~ 0.333)
        shadow_weight = np.clip((0.333 - luma) / 0.333, 0.0, 1.0)
        shadow_weight = pow_safe(shadow_weight, 2.0)

        # Highlights: 밝은 영역 (0.667 ~ 1.0)
        highlight_weight = np.clip((luma - 0.667) / 0.333, 0.0, 1.0)
        highlight_weight = pow_safe(highlight_weight, 2.0)

        # Midtones: 중간 영역 (bell curve)
        # 피크는 0.5, 0.333과 0.667에서 0
        midtone_weight = 1.0 - shadow_weight - highlight_weight
        midtone_weight = np.clip(midtone_weight, 0.0, 1.0)

        # 조정 벡터 생성 (BGR 순서)
        shadows_adj = np.array(
            [self.shadows_blue, self.shadows_green, self.shadows_red],
            dtype=np.float32,
        )

        midtones_adj = np.array(
            [self.midtones_blue, self.midtones_green, self.midtones_red],
            dtype=np.float32,
        )

        highlights_adj = np.array(
            [self.highlights_blue, self.highlights_green, self.highlights_red],
            dtype=np.float32,
        )

        # 각 톤 영역에 조정 적용
        result = img_float.copy()
        result += shadow_weight * shadows_adj * 0.5
        result += midtone_weight * midtones_adj * 0.5
        result += highlight_weight * highlights_adj * 0.5

        result = saturate(result)

        return (result * 255).astype(np.uint8)
