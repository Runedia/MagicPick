"""
ExtendedLevels.fx 구현
확장 레벨 조정 (입력/출력 레벨, 감마)

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import pow_safe, saturate


class ExtendedLevelsFilter(BaseFilter):
    def __init__(self):
        super().__init__("ExtendedLevels", "확장 레벨 조정")
        self.input_black = 0.0  # 0.0 ~ 1.0
        self.input_white = 1.0  # 0.0 ~ 1.0
        self.gamma = 1.0  # 0.1 ~ 3.0
        self.output_black = 0.0  # 0.0 ~ 1.0
        self.output_white = 1.0  # 0.0 ~ 1.0  # 0.1 ~ 10.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        확장 레벨 조정 적용

        Parameters:
        -----------
        input_black_lvl : int
            입력 블랙 레벨 (0 ~ 255)
        input_white_lvl : int
            입력 화이트 레벨 (0 ~ 255)
        output_black_lvl : int
            출력 블랙 레벨 (0 ~ 255)
        output_white_lvl : int
            출력 화이트 레벨 (0 ~ 255)
        gamma : float
            감마 값 (0.1 ~ 10.0)
        """
        self.input_black_lvl = params.get("input_black_lvl", self.input_black_lvl)
        self.input_white_lvl = params.get("input_white_lvl", self.input_white_lvl)
        self.output_black_lvl = params.get("output_black_lvl", self.output_black_lvl)
        self.output_white_lvl = params.get("output_white_lvl", self.output_white_lvl)
        self.gamma = params.get("gamma", self.gamma)

        img_float = image.astype(np.float32) / 255.0

        # 1. Input Levels 적용
        input_black = self.input_black_lvl / 255.0
        input_white = self.input_white_lvl / 255.0
        input_range = input_white - input_black

        if input_range > 0:
            result = (img_float - input_black) / input_range
        else:
            result = img_float

        result = saturate(result)

        # 2. Gamma 적용
        result = pow_safe(result, 1.0 / self.gamma)

        # 3. Output Levels 적용
        output_black = self.output_black_lvl / 255.0
        output_white = self.output_white_lvl / 255.0
        output_range = output_white - output_black

        result = result * output_range + output_black

        result = saturate(result)

        return (result * 255).astype(np.uint8)
