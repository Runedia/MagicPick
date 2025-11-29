"""
Lift Gamma Gain 필터

색보정의 기본 도구로, 그림자(Lift), 중간톤(Gamma), 하이라이트(Gain)를 각각 조정합니다.
RGB 채널별로 독립적으로 조정 가능합니다.
"""

import numpy as np

from filters.base_filter import BaseFilter

from .hlsl_helpers import pow_safe, saturate


class LiftGammaGainFilter(BaseFilter):
    """Lift Gamma Gain 색보정 필터"""

    def __init__(self):
        super().__init__("LiftGammaGain", "리프트/감마/게인 조정")
        # 기본값: [1.0, 1.0, 1.0] (변화 없음)
        self.rgb_lift = [1.0, 1.0, 1.0]  # 그림자 조정 (0.0 ~ 2.0)
        self.rgb_gamma = [1.0, 1.0, 1.0]  # 중간톤 조정 (0.0 ~ 2.0)
        self.rgb_gain = [1.0, 1.0, 1.0]  # 하이라이트 조정 (0.0 ~ 2.0)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Lift Gamma Gain 필터 적용"""
        # 파라미터 업데이트
        if "rgb_lift" in params:
            self.rgb_lift = params["rgb_lift"]
        if "rgb_gamma" in params:
            self.rgb_gamma = params["rgb_gamma"]
        if "rgb_gain" in params:
            self.rgb_gain = params["rgb_gain"]

        img_float = image.astype(np.float32) / 255.0

        # NumPy 배열로 변환 (BGR -> RGB)
        lift = np.array(
            [self.rgb_lift[2], self.rgb_lift[1], self.rgb_lift[0]],
            dtype=np.float32,
        )
        gamma = np.array(
            [self.rgb_gamma[2], self.rgb_gamma[1], self.rgb_gamma[0]],
            dtype=np.float32,
        )
        gain = np.array(
            [self.rgb_gain[2], self.rgb_gain[1], self.rgb_gain[0]],
            dtype=np.float32,
        )

        # 1. Lift: 그림자 조정
        # color = color * (1.5 - 0.5 * RGB_Lift) + 0.5 * RGB_Lift - 0.5
        color = img_float * (1.5 - 0.5 * lift) + 0.5 * lift - 0.5
        color = saturate(color)

        # 2. Gain: 하이라이트 조정
        color = color * gain

        # 3. Gamma: 중간톤 조정
        # color = pow(abs(color), 1.0 / RGB_Gamma)
        color = pow_safe(np.abs(color), 1.0 / gamma)

        color = saturate(color)

        return (color * 255).astype(np.uint8)
