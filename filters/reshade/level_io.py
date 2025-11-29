"""
ReShade LevelIO 필터
"""

import numpy as np

from ..base_filter import BaseFilter


class LevelIOFilter(BaseFilter):
    def __init__(self):
        super().__init__(
            "LevelIO",
            "ReShade의 LevelIO 효과를 적용하여 입력/출력 레벨과 감마를 조절합니다.",
        )
        self.set_default_params(
            {
                "input_black_point": 0.0,  # 0-255
                "input_white_point": 255.0,  # 0-255
                "gamma": 1.0,  # 0.1-10.0
                "saturation": 1.0,  # 0.0-2.0
                "output_black_point": 0.0,  # 0-255
                "output_white_point": 255.0,  # 0-255
            }
        )

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        params = self.validate_params(params)
        img_float = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Normalize input parameters to [0, 1] range
        ib = params["input_black_point"] / 255.0
        iw = params["input_white_point"] / 255.0
        g = params["gamma"]
        s = params["saturation"]
        ob = params["output_black_point"] / 255.0
        ow = params["output_white_point"] / 255.0

        color = img_float.copy()

        # Input Levels
        # color.rgb=min(max(color.rgb-ib, 0)/(iw-ib), 1);
        numerator = color - ib
        denominator = iw - ib

        # 0으로 나누기 방지
        denominator[denominator == 0] = 1.0  # If iw == ib, result is 1

        color = np.clip(numerator / denominator, 0.0, 1.0)

        # Gamma
        # if(lin_g != 1) color.rgb=pow(abs(color.rgb), 1/lin_g);
        if g != 1.0:
            # abs is not strictly needed as color is already [0,1]
            color = np.power(color, 1.0 / g)

        # Output Levels
        # color.rgb=min( max(color.rgb*(ow-ob)+ob, ob), ow);
        color = color * (ow - ob) + ob
        color = np.clip(color, ob, ow)  # ReShade shader uses ob and ow as clamp limits

        # Saturation
        # if (lio_s != 1) { const float cm=(color.r+color.g+color.b)/3; color.rgb=cm-(cm-color.rgb)*lio_s; }
        if s != 1.0:
            cm = np.mean(color, axis=-1, keepdims=True)
            color = cm - (cm - color) * s

        return np.clip(color * 255.0, 0, 255).astype(np.uint8)
