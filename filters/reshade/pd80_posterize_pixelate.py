"""
PD80_06_Posterize_Pixelate.fx 구현
PD80 포스터화/픽셀화 효과

원본 셰이더: https://github.com/prod80/prod80-ReShade-Repository
"""

import cv2
import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class PD80PosterizePixelateFilter(BaseFilter):
    def __init__(self):
        super().__init__("PD80PosterizePixelate", "포스터화/픽셀화")
        self.color_levels = 8.0  # 2.0 ~ 32.0
        self.pixel_size = 1.0  # 1.0 ~ 16.0  # 1 ~ 32

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        포스터화/픽셀화 효과 적용

        Parameters:
        -----------
        enable_posterize : bool
            포스터화 활성화
        enable_pixelate : bool
            픽셀화 활성화
        color_levels : int
            포스터화 색상 레벨 (2 ~ 256)
        pixel_size : int
            픽셀 크기 (1 ~ 32)
        """
        self.enable_posterize = params.get("enable_posterize", self.enable_posterize)
        self.enable_pixelate = params.get("enable_pixelate", self.enable_pixelate)
        self.color_levels = params.get("color_levels", self.color_levels)
        self.pixel_size = params.get("pixel_size", self.pixel_size)

        img_float = image.astype(np.float32) / 255.0
        result = img_float.copy()

        # 1. 픽셀화 (먼저 적용)
        if self.enable_pixelate and self.pixel_size > 1:
            h, w = result.shape[:2]

            # 다운샘플링
            small_h = max(1, h // self.pixel_size)
            small_w = max(1, w // self.pixel_size)
            small = cv2.resize(result, (small_w, small_h), interpolation=cv2.INTER_AREA)

            # 업샘플링 (Nearest neighbor로 픽셀 효과)
            result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. 포스터화
        if self.enable_posterize and self.color_levels < 256:
            # 색상 양자화
            result = np.floor(result * self.color_levels) / self.color_levels

        result = saturate(result)

        return (result * 255).astype(np.uint8)
