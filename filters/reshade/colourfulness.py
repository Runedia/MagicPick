"""
Colourfulness.fx 구현
채도 강화 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class ColourfulnessFilter(BaseFilter):
    def __init__(self):
        super().__init__("Colourfulness", "채도 강화")
        self.colourfulness = 0.4  # 0.0 ~ 2.5
        self.lim_luma = 0.7  # 0.1 ~ 1.0  # 0.1 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        채도 강화 적용

        Parameters:
        -----------
        colourfulness : float
            채도 강화 정도 (0.0 ~ 2.5)
        lim_luma : float
            밝기 제한 (0.1 ~ 1.0)
        """
        self.colourfulness = params.get("colourfulness", self.colourfulness)
        self.lim_luma = params.get("lim_luma", self.lim_luma)

        img_float = image.astype(np.float32) / 255.0

        # RGB를 Luma로 변환
        luma = np.dot(img_float, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)

        # 채도 강화 계산
        # 원본 색상과 밝기의 차이를 증폭
        chroma = img_float - luma

        # colourfulness 파라미터로 채도 조정
        chroma = chroma * (1.0 + self.colourfulness)

        # 재결합
        result = luma + chroma

        # lim_luma로 밝기 제한 적용
        max_luma = self.lim_luma
        luma_scale = np.where(luma > max_luma, max_luma / np.maximum(luma, 1e-8), 1.0)
        result = result * luma_scale

        result = saturate(result)

        return (result * 255).astype(np.uint8)
