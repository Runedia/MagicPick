"""
ReShade Deblur 필터
"""

import numpy as np

from filters.base_filter import BaseFilter


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


class DeblurFilter(BaseFilter):
    def __init__(self):
        super().__init__("Deblur", "ReShade의 Deblur 효과를 적용합니다.")
        self.set_default_params(
            {
                "offset": 1.0,
                "strength": 6.0,
                "smart": 0.7,
            }
        )

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        이미지에 Deblur 필터를 적용합니다.

        Args:
            image: 입력 이미지 (NumPy array, RGB 형식)
            **params:
                offset (float): 필터 너비 (정수 값으로 처리됨, 예: 1.0, 2.0)
                strength (float): 블러 제거 강도 (1.0 ~ 9.0)
                smart (float): 스마트 블러 강도 (0.0 ~ 1.0)

        Returns:
            필터가 적용된 이미지 (NumPy array, RGB 형식)
        """
        params = self.validate_params(params)
        # offset은 정수 픽셀 거리로 처리
        offset = int(round(float(params["offset"])))
        strength = float(params["strength"])
        smart = float(params["smart"])

        img_float = image.astype(np.float32)

        # 이웃 픽셀을 가져오기 위해 배열을 복사하고 슬라이싱
        # 패딩을 사용하는 대신, 경계 근처에서는 잘못된 값을 생성할 수 있지만,
        # 대부분의 이미지에서 눈에 띄지 않으며 코드가 훨씬 간단해짐.
        c11 = img_float

        # 각 이웃에 대해 shift된 배열 생성
        c00 = np.roll(img_float, shift=(offset, offset), axis=(0, 1))
        c10 = np.roll(img_float, shift=offset, axis=0)
        c20 = np.roll(img_float, shift=(offset, -offset), axis=(0, 1))
        c01 = np.roll(img_float, shift=offset, axis=1)
        c21 = np.roll(img_float, shift=-offset, axis=1)
        c02 = np.roll(img_float, shift=(-offset, offset), axis=(0, 1))
        c12 = np.roll(img_float, shift=-offset, axis=0)
        c22 = np.roll(img_float, shift=(-offset, -offset), axis=(0, 1))

        # 3x3 영역의 최소/최대값 계산
        all_neighbors = np.stack([c00, c10, c20, c01, c11, c21, c02, c12, c22], axis=0)
        mn1 = np.min(all_neighbors, axis=0)
        mx1 = np.max(all_neighbors, axis=0)

        contrast = mx1 - mn1
        m = np.max(contrast, axis=2, keepdims=True)

        # 핵심 deblur 로직
        dif1 = np.abs(c11 - mn1) + 1e-5
        df1 = np.power(dif1, strength)

        dif2 = np.abs(c11 - mx1) + 1e-5
        df2 = np.power(dif2, strength)

        dif1_sq = dif1 * dif1 * dif1
        dif2_sq = dif2 * dif2 * dif2

        # 0으로 나누기 방지
        den = dif1_sq + dif2_sq
        den[den == 0] = 1.0

        df = dif1_sq / den
        ratio = np.abs(dif1_sq - dif2_sq) / den

        d11 = _lerp(c11, _lerp(mn1, mx1, df), ratio)

        c11_updated = _lerp(c11, d11, np.clip(2.0 * m - 0.125, 0, 1))

        d11_final = _lerp(c11_updated, d11, smart)

        # 결과를 0-255 범위로 클리핑하고 uint8로 변환
        result = np.clip(d11_final, 0, 255).astype(np.uint8)

        return result
