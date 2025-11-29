"""
SimpleGrain.fx 구현
심플 그레인 효과

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class SimpleGrainFilter(BaseFilter):
    def __init__(self):
        super().__init__("SimpleGrain", "심플 그레인")
        self.intensity = 0.2  # 0.0 ~ 1.0
        self.mean = 0.0  # 노이즈 평균
        self.variance = 0.05  # 노이즈 분산  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        심플 그레인 적용

        가우시안 노이즈를 이용한 필름 그레인 효과

        Parameters:
        -----------
        intensity : float
            그레인 강도 (0.0 ~ 1.0)
        variance : float
            그레인 분산 (0.0 ~ 1.0)
        mean : float
            그레인 평균 (0.0 ~ 1.0)
        """
        self.intensity = params.get("intensity", self.intensity)
        self.variance = params.get("variance", self.variance)
        self.mean = params.get("mean", self.mean)

        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 가우시안 노이즈 생성
        np.random.seed(42)  # 일관성을 위한 고정 시드
        noise = np.random.normal(self.mean, self.variance, (h, w))
        noise = np.clip(noise, 0.0, 1.0)

        # 노이즈를 3채널로 확장
        noise_rgb = np.stack([noise, noise, noise], axis=2)

        # 루마 기반 가중치 (어두운 부분에 더 강한 그레인)
        luma = np.dot(img_float, [0.2126, 0.7152, 0.0722])
        luma = np.expand_dims(luma, axis=2)

        # 어두운 영역에 더 강하게 적용
        grain_strength = self.intensity * (1.0 - luma * 0.5)

        # 노이즈 적용
        result = img_float + (noise_rgb - self.mean) * grain_strength

        result = saturate(result)

        return (result * 255).astype(np.uint8)
