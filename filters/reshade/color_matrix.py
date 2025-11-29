"""
Color Matrix 필터

3x3 색상 행렬을 사용하여 색상을 변환합니다.
각 출력 색상 채널이 입력 RGB의 조합으로 구성됩니다.
"""

import numpy as np

from filters.base_filter import BaseFilter
from filters.reshade.hlsl_helpers import lerp, saturate


class ColorMatrixFilter(BaseFilter):
    """색상 행렬 변환 필터"""

    def __init__(self):
        super().__init__("ColorMatrix", "색상 행렬 변환")
        # 기본 행렬 (약간의 색상 변화)
        self.matrix_red = [
            0.817,
            0.183,
            0.000,
        ]  # 새로운 R = 0.817*R + 0.183*G + 0.000*B
        self.matrix_green = [0.333, 0.667, 0.000]  # 새로운 G
        self.matrix_blue = [0.000, 0.125, 0.875]  # 새로운 B
        self.strength = 1.0  # 0.0 ~ 1.0

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Color Matrix 필터 적용"""
        # 파라미터 업데이트
        if "matrix_red" in params:
            self.matrix_red = params["matrix_red"]
        if "matrix_green" in params:
            self.matrix_green = params["matrix_green"]
        if "matrix_blue" in params:
            self.matrix_blue = params["matrix_blue"]
        self.strength = params.get("strength", self.strength)

        img_float = image.astype(np.float32) / 255.0

        # 3x3 색상 행렬 생성 (행 우선)
        # HLSL의 float3x3(row0, row1, row2)는 행렬의 각 행을 나타냄
        # NumPy에서는 열 우선이므로 전치 필요
        color_matrix = np.array(
            [
                self.matrix_red,  # 새로운 R을 만드는 계수
                self.matrix_green,  # 새로운 G를 만드는 계수
                self.matrix_blue,  # 새로운 B를 만드는 계수
            ],
            dtype=np.float32,
        ).T  # 전치하여 올바른 형태로 변환

        # 이미지를 (H*W, 3) 형태로 reshape
        h, w = img_float.shape[:2]
        pixels = img_float.reshape(-1, 3)

        # 행렬 곱셈: (H*W, 3) @ (3, 3) = (H*W, 3)
        transformed = np.dot(pixels, color_matrix)

        # 원래 형태로 복원
        transformed = transformed.reshape(h, w, 3)

        # 강도 조정
        result = lerp(img_float, transformed, self.strength)
        result = saturate(result)

        return (result * 255).astype(np.uint8)
