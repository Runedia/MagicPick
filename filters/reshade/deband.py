"""
Deband.fx 구현
디밴딩 효과 (밴딩 제거)

원본 셰이더: https://github.com/crosire/reshade-shaders
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import saturate


class DebandFilter(BaseFilter):
    def __init__(self):
        super().__init__("Deband", "디밴딩 (밴딩 제거)")
        self.threshold = 0.003  # 0.0 ~ 0.1
        self.iterations = 1  # 1 ~ 4
        self.range = 16.0  # 1.0 ~ 64.0  # 1 ~ 4

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        디밴딩 적용

        그라디언트 밴딩 아티팩트를 제거하기 위해
        랜덤 샘플링과 블렌딩 사용

        Parameters:
        -----------
        range : float
            샘플링 범위 (1.0 ~ 64.0)
        threshold : float
            블렌딩 임계값 (0.0 ~ 0.1)
        iterations : int
            반복 횟수 (1 ~ 4)
        """
        self.range = params.get("range", self.range)
        self.threshold = params.get("threshold", self.threshold)
        self.iterations = params.get("iterations", self.iterations)

        img_float = image.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        result = img_float.copy()

        for iteration in range(self.iterations):
            result = self._deband_iteration(result, h, w)

        result = saturate(result)

        return (result * 255).astype(np.uint8)

    def _deband_iteration(self, img, h, w):
        """
        디밴딩 1회 반복
        """
        # 랜덤 오프셋 생성 (deterministic for static images)
        np.random.seed(42)

        # 각 픽셀에 대해 랜덤 샘플 4개 추출
        angles = np.array([0, 90, 180, 270]) * (np.pi / 180.0)

        samples = []
        for angle in angles:
            # 랜덤 거리
            distance = np.random.uniform(1.0, self.range)

            # 오프셋 계산
            offset_x = int(np.cos(angle) * distance)
            offset_y = int(np.sin(angle) * distance)

            # 샘플 추출 (경계 처리)
            sample = np.roll(img, (offset_y, offset_x), axis=(0, 1))
            samples.append(sample)

        # 샘플 평균
        avg_sample = np.mean(samples, axis=0)

        # 원본과의 차이 계산
        diff = np.abs(img - avg_sample)
        diff_luma = np.max(diff, axis=2, keepdims=True)

        # 임계값 이하일 때만 블렌딩 (밴딩 영역)
        blend_mask = (diff_luma < self.threshold).astype(np.float32)

        # 블렌딩
        result = img * (1.0 - blend_mask) + avg_sample * blend_mask

        return result
